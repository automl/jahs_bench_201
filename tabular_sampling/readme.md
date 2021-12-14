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

1. Use `tabular_sampling.clusterlib.prescheduler` to generate a sampling profile. A sampling profile defines the number 
   of independent, parallel bit-streams that will be used to generate model configurations, the number of samples that 
   will be drawn from each bit-stream, and some estimates of the SLURM job resource requirements and directory 
   structure. The resource requirements here are expected to be rough estimates and can be updated later (read below).
2. Use `tabular_sampling.scripts.preschedule` in the mode `initialize` to prepare the directory where the benchmark 
   data will be stored and sample the actual model configurations.
3. Use `tabular_sampling.scripts.preschedule` in the mode `generate` to generate the job scripts. This can be used to 
   define arbitrary SLURM node and job setups across which the sampling profile is distributed based on estimates of 
   how long sampling will take and the intended training configuration. It is recommended to first perform a small run, 
   say 10 epochs long, in order to generate a good estimate of each model's per-epoch runtime, before updating the 
   sampling profile and generating a fresh set of jobs for longer evaluation, e.g. 50, 100, ... epochs.

### 1a. Re-creating the original benchmark data

The same procedure as above can be used to do so. The relevant sampling profiles have been uploaded.

