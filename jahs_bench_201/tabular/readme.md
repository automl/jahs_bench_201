# NEPSBench-101
This repository contains the original code for the Neural Pipeline Search 
Benchmark 101. 
It has been designed to allow a user to: 
1. Re-generate the original benchmark data. 
2. Verify the experiments performed using the original data.
3. Generate data for new configurations using the same pipeline as was used for the original data.
4. Use the surrogate benchmark based on the original data as a drop-in replacement for the objective function in a 
   NAS/HPO/NAS+HPO search algorithm.
5. Use the training pipeline to live train models for a NAS/HPO/NAS+HPO search algorithm.

## 1. Generating Benchmark Data
The following is an overview of the process for generating any data using the benchmarking pipeline. This assumes that 
the benchmarking process is being run over a SLURM CPU cluster.

1. **Generate a sampling profile:** Use the [pre-scheduling library](clusterlib/prescheduler.py) to generate a sampling 
   profile. A sampling profile defines the number of independent, parallel bit-streams that will be used to generate 
   model configurations, the number of samples that will be drawn from each bit-stream, and some estimates of the SLURM 
   job resource requirements and directory structure. The resource requirements here are expected to be rough estimates 
   and can be updated later (read below).
2. **Initialize the sampling profile:** Use the [pre-scheduler script](scripts/preschedule) in the `initialize` mode to 
   prepare the directory where the benchmark data will be stored and sample the actual model configurations.
3. **Generate training metrics:** Use the [pre-scheduler script](scripts/preschedule) in the `generate` mode to 
   generate the job scripts. This can be used to define arbitrary SLURM node and job setups across which the sampling 
   profile is distributed based on estimates of how long sampling will take and the intended training configuration. It 
   is recommended to first perform a small run, say 10 epochs long, in order to generate a good estimate of each 
   model's per-epoch runtime, before updating the sampling profile and generating a fresh set of jobs for longer 
   evaluations, e.g. 50, 100, ... epochs. Increasing the number of epochs incrementally also enables compute resource 
   requirements to be continually updated and controlled.

### 1a. Reproducibility of generated data:
When referring to reproducibility, there are two separate aspects that need to be addressed:
1. The generation of the sampling profile
2. The generation of the metrics of the model training runs

When reproducing the sampling profile, the configurations must  retain statistical independent between random samples. 
When reproducing the model training metrics for each configuration, the loss and accuracy should be reproduced 
exactly epoch-per-epoch (this does not apply to hardware dependent metrics such as training duration and runtime). 
Given a pre-generated sampling profile, which includes randomly generated seeds for model training, the model training 
can easily be reproduced in terms of losses and accuracies as long as the PyTorch versions allow this. Reproducing the 
sampling profile may be dependent on specific versions of this code repository. Nevertheless, new data can be added to 
existing data from older versions of the code as long as the sampling guidelines for statistical independence (see 
section 2) are followed.


### 1b. Re-creating the original benchmark data
The same procedure as above can be used to do so. The relevant sampling profiles have been uploaded, which should 
enable model trainings to always be reproduced barring future changes to PyTorch that break this functionality. For 
guaranteed results, the exact PyTorch version from the time of release of certain data should be used. 

If the sampling profile itself needs to be reproduced in addition to re-running the original benchmarking experiments 
on CIFAR-10 data, the original sampling profile should be used along with the development-only version **0.1.0** of 
this repository, although this should be an entirely unncessary use case. This restriction is because some changes 
introduced after this point could potentially break the reproducibility of the sampling profile (and not the model 
training) for the CIFAR-10 data. 


### 2. Statistical Independence of Random Samples
Statistical independence of random samples drawn for generating any given sampling profile is guaranteed by drawing 
multiple parallel pseudo-random number generators (PRNGs) from a common seed that can then be split up into (nearly) as 
many parallel PRNGs that still retain statistical independence as needed due to the use of a Philox PRNG 
**[needs reference]**. Offsets to the Philox PRNG can be used to generate more statistically independent samples for 
the profile, independent of the code version, by choosing an offset value that has not already been used. The only 
restriction is that the original common seed stored in [distributed_nas_sampling.py](distributed_nas_sampling.py) must 
remain unchanged. 

These offset values are referred to as the "taskid" in the metrics DataFrame's index and are also used for 
enabling some level of paralllel operation on the storage systems in cases when a large number of parallel processes 
require simultaneous read/write access. It is, thus, a good idea to balance the number of individual bit-streams 
(taskid) that are running in parallel and number of samples drawn from each bit stream (model_idx) when training a 
large number of models in parallel.