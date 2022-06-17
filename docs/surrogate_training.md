# Training a Surrogate Model

__Note:__ In order to use our surrogate training scripts, users must install the package Neural Pipeline Search (NePS)
as an extra dependency. This can be done easily by running the following shell command in an environment already
containing all the dependencies for JAHS-Bench-201:

```bash
pip install neural-pipeline-search
```

In case of any issues, users should consult the Git repo and documentation for NePS, which can be found
[here](https://github.com/automl/neps).

## Downloading the Data

The data splits for training a model can be downloaded by running the following bash script:
```bash
python -m jahs_bench.download --target=metric_data --save_dir=$save_dir
```

where `save_dir` is a directory where the data will be stored.

## Training the Models

We train one surrogate model per metric. The downloaded data files are already arranged in a directory structure with
the format `<save_dir>/metric_data/<task>/<metric>/{train,valid,test}_set.pkl.gz`. Thus, for training a surrogate for
predicting, for instance, the validation accuracy scores on CIFAR-10 over 20 HPO evaluations, one would then run the
following shell script:

```bash
python -m jahs_bench.surrogate_training.pipeline
  --working_directory=$root_dir/$metric
  --datadir=$save_dir/metric_data/$task
  --output=$metric
  --max_evaluations_total=20
```

Here, `root_dir` is a directory where all the interim data files and logs generated during the HPO loop,
including the trained interim surrogate models, will be stored for one given task. The above command can be executed by multiple parallel
workers as long as all the input arguments remain fixed in order to achieve parallelization (i.e. when launched with 4
workers, all workers will stop once 20 total evaluations are reached).

## Assembling the Final Ensemble

Once the above HPO loop finishes, it is necessary to extract the best model for each metric and put them together into
an ensemble that can be used by the Benchmark API. In order to do this, run the following shell script:

```bash
python -m jahs_bench.surrogate_training.assemble_models
  --final_dir=$final_dir
  --root_dir=$root_dir
```

where `final_dir` is the directory where the ensemble for one given task should be saved and `root_dir` is the same as
in the previous step.

## Evaluating the Trained Surrogates

Finally, the trained surrogates can be evaluated on the test set to obtain the final correlation and regression scores.
This can be achieved by running the following shell script:

```bash
outputs=(latency runtime valid-acc)
python -m jahs_bench.surrogate_training.evaluation
  --testset-file=$path_to_test_set
  --model-dir=$final_dir
  --save-dir=$save_dir
  --outputs ${outputs[@]}
```

Where `path_to_test_set` is the full path to the relevant `test_set.pkl.gz` file of the task the surrogate was trained
for, `final_dir` is the directory where a surrogate ensemble was saved in the previous step, `save_dir` is a directory
where the results of running this script - including surrogate predictions on the test set and the generated scores -
should be saved. Finally, depending on which metrics the surrogate was trained to predict, the array `outputs` should
be set accordingly. Here, we show an example using the metrics "latency", "runtime" and "valid-acc"
(validation accuracy).

## Cluster-based Computation

We note that our own experiments were run on a distributed computing cluster, which entails cluster-specific details
and configurations, but used the same scripts as described here at their heart. Depending on their own local setup,
users may thus have to tweak their usage of the scripts appropriately. These scripts may also contain additional
options to help users with this setup, accessible by running `python -m $script_name.py --help`.
