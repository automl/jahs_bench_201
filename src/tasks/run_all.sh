#!/bin/bash
#/home/maczek/Documents/jahs_bench
export PYTHONPATH=/home/maczek/Documents/jahs_bench/jahs_bench_mf/tabular_sampling/:$PYTHONPATH
export PYTHONPATH=/home/maczek/Documents/jahs_bench/ICGen/:$PYTHONPATH
export PYTHONPATH=/home/maczek/Documents/jahs_bench/ICGen/icgen/:$PYTHONPATH
export PYTHONPATH=/home/maczek/Documents/jahs_bench/NASLib/:$PYTHONPATH
export PYTHONPATH=/home/maczek/Documents/jahs_bench/NASLib/naslib/:$PYTHONPATH
export PYTHONPATH=/home/maczek/Documents/jahs_bench/jahs_bench_mf/src/tasks/:$PYTHONPATH
export PYTHONPATH=/home/maczek/Documents/jahs_bench/:$PYTHONPATH
export PYTHONPATH=/home/maczek/Documents/jahs_bench/jahs_bench_mf/:$PYTHONPATH


export PYTHONPATH=/home/janowski/jahs_bench/jahs_bench_mf/tabular_sampling/:$PYTHONPATH
export PYTHONPATH=/home/janowski/jahs_bench/ICGen/:$PYTHONPATH
export PYTHONPATH=/home/janowski/jahs_bench/ICGen/icgen/:$PYTHONPATH
export PYTHONPATH=/home/janowski/jahs_bench/NASLib/:$PYTHONPATH
export PYTHONPATH=/home/janowski/jahs_bench/NASLib/naslib/:$PYTHONPATH
export PYTHONPATH=/home/janowski/jahs_bench/jahs_bench_mf/src/tasks/:$PYTHONPATH
export PYTHONPATH=/home/janowski/jahs_bench/:$PYTHONPATH
export PYTHONPATH=/home/janowski/jahs_bench/jahs_bench_mf/:$PYTHONPATH

declare -a tasks=(
                "--fidelity None --n_iterations 100" "--fidelity None --use_default_hps --n_iterations 100" "--fidelity None --use_default_arch --n_iterations 100"

                "--fidelity Epochs" #"--fidelity Epochs --use_default_hps" "--fidelity Epochs --use_default_arch"
                "--fidelity N" # "--fidelity N --use_default_hps" "--fidelity N --use_default_arch"
                "--fidelity W" # "--fidelity W --use_default_hps" "--fidelity W --use_default_arch"
                "--fidelity Resolution" # "--fidelity Resolution --use_default_hps" "--fidelity Resolution --use_default_arch"
#                "--fidelity Epochs --use_model --n_iterations 30"
#                "--fidelity N --use_model --n_iterations 30"
#                "--fidelity W --use_model --n_iterations 30"
#                "--fidelity Resolution --use_model --n_iterations 30"

                )
declare -a datasets=(
  "cifar10"
#  "colorectal_histology"
#  "fashionMNIST"
)

for d in "${datasets[@]}"
do
  for i in {1..100}
  do
    for j in "${tasks[@]}"
    do
      python run_task.py --seed $i $j --dataset $d
    done
  done
done

python analysis.py
