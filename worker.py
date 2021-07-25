import logging
import numpy as np
import torch

from hpbandster.core.worker import Worker
from hpbandster.examples.commons import MyWorker
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils.utils import get_train_val_loaders
from naslib.utils import utils
import naslib.utils.logging as naslib_logging

class ModelTrainer(Worker):

    def __init__(self, search_space: NasBench201SearchSpace, seed, job_config, **kwargs):
        super(ModelTrainer, self).__init__(**kwargs)
        self.search_space = search_space.clone()
        self.random_seed = seed
        self.num_epochs = int(job_config.search.epochs)
        self.job_config = job_config.sampler
        self.data_loaders = get_train_val_loaders(job_config, mode='train')


    def compute(self, config_id, config, budget, working_directory):
        """
        Here, budget will refer to the number of epochs that the model is to be trained for. Returns the test loss and
        the complete info dict.
        """

        start_time = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model: NasBench201SearchSpace = self.search_space.clone()
        model.set_op_indices(config)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        model.parse()
        model = model.to(device)

        errors_dict, metrics = self.get_metrics(model)
        train_queue, valid_queue, test_queue, _, _ = self.data_loaders

        # TODO: Fix references to config for learning rate, weight decay
        optim = torch.optim.Adam(model.parameters(), lr=self.job_config.learning_rate,
                                 weight_decay=self.job_config.weight_decay)
        loss = torch.nn.CrossEntropyLoss()

        naslib_logging.log_first_n(logging.INFO, f"Training config {config} for {self.num_epochs} epochs.", 3)
        train_start_time = time.time()
        for e in range(self.num_epochs):
            for step, ((train_inputs, train_labels), (val_inputs, val_labels)) in \
                    enumerate(zip(train_queue, valid_queue)):
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device)
                optim.zero_grad()
                logits_train = model(train_inputs)
                train_loss = loss(logits_train, train_labels)
                train_loss.backward()
                optim.step()

                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                logits_val = model(val_inputs)
                val_loss = loss(logits_val, val_labels)

                naslib_logging.log_every_n_seconds(
                    logging.INFO,
                    "Epoch {}-{}, Train loss: {:.5f}, validation loss: {:.5f}".format(e, step, train_loss, val_loss),
                    n=5
                )

                metrics.train_loss.update(float(train_loss.detach().cpu()))
                metrics.val_loss.update(float(val_loss.detach().cpu()))
                update_accuracies(logits_train, train_labels, "train")
                update_accuracies(logits_val, val_labels, "val")

            errors_dict.train_acc.append(metrics.train_acc.avg)
            errors_dict.train_loss.append(metrics.train_loss.avg)
            errors_dict.val_acc.append(metrics.val_acc.avg)
            errors_dict.val_loss.append(metrics.val_loss.avg)

        train_end_time = time.time()
        # logging.info("Finished training.")

        for (test_inputs, test_labels) in test_queue:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            logits_test = model(test_inputs)
            test_loss = loss(logits_test, test_labels)
            metrics.test_loss.update(float(test_loss.detach().cpu()))
            update_accuracies(logits_test, test_labels, "test")

        end_time = time.time()

        errors_dict.test_acc = metrics.test_acc.avg
        errors_dict.test_loss = metrics.test_loss.avg
        errors_dict.runtime = end_time - start_time
        errors_dict.train_time = train_end_time - train_start_time

        return errors_dict.test_loss, errors_dict

    def get_metrics(self, model):
        errors_dict = utils.AttrDict(
            {'train_acc': [],
             'train_loss': [],
             'val_acc': [],
             'val_loss': [],
             'test_acc': None,
             'test_loss': None,
             'runtime': None,
             'train_time': None,
             'params': utils.count_parameters_in_MB(model)}
        )

        metrics = utils.AttrDict({
            'train_acc': utils.AverageMeter(),
            'train_loss': utils.AverageMeter(),
            'val_acc': utils.AverageMeter(),
            'val_loss': utils.AverageMeter(),
            'test_acc': utils.AverageMeter(),
            'test_loss': utils.AverageMeter(),
        })

        return errors_dict, metrics

    def update_accuracies(metrics, logits, target, split):
        """Update the accuracy counters"""
        logits = logits.clone().detach().cpu()
        target = target.clone().detach().cpu()
        acc, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = logits.size(0)

        if split == 'train':
            metrics.train_acc.update(acc.data.item(), n)
        elif split == 'val':
            metrics.val_acc.update(acc.data.item(), n)
        elif split == 'test':
            metrics.test_acc.update(acc.data.item(), n)
        else:
            raise ValueError("Unknown split: {}. Expected either 'train' or 'val'")
