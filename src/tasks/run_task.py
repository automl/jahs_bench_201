from path import Path
import os
import pickle

from wrapper.jahs_bench_wrapper import JAHS_Bench_wrapper
from optimizer.random_search import RandomSearch
from optimizer.successive_halving import SuccessiveHalving
from hpbandster.optimizers.bohb import BOHB
from jahs_bench_201.src.tasks.utils.exp_setup import set_seed, args

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

set_seed(args.seed)

if args.fidelity is None:
    experiment = "RS"
elif args.use_model:
    experiment = f"BOHB_{args.fidelity}"
else:
    experiment = f"SH_{args.fidelity}"
if args.use_default_hps:
    experiment += "_just_nas"
elif args.use_default_arch:
    experiment += "_just_hpo"
args.working_directory = os.path.join(args.working_directory, f"{args.dataset}")
args.working_directory = os.path.join(args.working_directory, experiment, str(args.seed))

working_dir = Path(args.working_directory)
working_dir.makedirs_p()
# with open(working_dir / "args.json", "w") as f:
#     json.dump(args.__dict__, f, indent=4)

model_path = args.model_path / f"{args.dataset}"

result_logger = hpres.json_result_logger(directory=args.working_directory,
                                         overwrite=False)

NS = hpns.NameServer(
    run_id=args.run_id,
    host=args.host,
    port=0,
    working_directory=args.working_directory
)
ns_host, ns_port = NS.start()

worker = JAHS_Bench_wrapper(
    dataset=args.dataset,
    model_path=model_path,
    use_default_hps=args.use_default_hps,
    use_default_arch=args.use_default_arch,
    fidelity=args.fidelity,
    seed=args.seed,
    run_id=args.run_id,
    host=args.host,
    nameserver=ns_host,
    nameserver_port=ns_port,
    timeout=120
)

worker.run(background=True)

if args.fidelity is None:
    rs = RandomSearch(
        configspace=worker.joint_config_space,
        run_id=args.run_id, host=args.host, nameserver=ns_host,
        nameserver_port=ns_port, result_logger=result_logger,
        min_budget=args.max_budget, max_budget=args.max_budget,
    )
elif args.use_model:
    rs = BOHB(
        configspace=worker.joint_config_space,
        run_id=args.run_id, host=args.host, nameserver=ns_host,
        nameserver_port=ns_port, result_logger=result_logger,
        min_budget=args.min_budget, max_budget=args.max_budget, eta=args.eta,
    )
else:
    rs = SuccessiveHalving(
        configspace=worker.joint_config_space,
        run_id=args.run_id, host=args.host, nameserver=ns_host,
        nameserver_port=ns_port, result_logger=result_logger,
        min_budget=args.min_budget, max_budget=args.max_budget, eta=args.eta,
        fidelity=args.fidelity
    )

res = rs.run(n_iterations=args.n_iterations)

with open(os.path.join(args.working_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)

rs.shutdown(shutdown_workers=True)
NS.shutdown()

id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.' % (
        sum([r.budget for r in res.get_all_runs()]) / args.max_budget))
