We accompany JAHS-Bench-201 with evaluation protocols and code to aid 
following these protocols, to facilitate fair, reproducible, and methodologically sound comparisons.
Studies with JAHS-Bench-201 should report on all three datasets and present results as trajectories of
best validation error (single-objective) or hypervolume (multi-objective). To allow using the same
evaluation protocol and comparisons across the various algorithmic settings above, these trajectories
should be reported across total runtime taking into account the training and evaluation costs predicted
by the surrogate benchmark. We suggest to report until a runtime corresponding to approximately
100 evaluations and, for interpretability, show the total runtime divided by the
mean runtime of one evaluation. Further, in (multi) multi-fidelity runs utilizing epochs, we support
both using continuations from few to many epochs (to simulate the checkpointing and continuation of
the model) as well as retraining from scratch for optimizers that do not support continuations. Results
should report the mean and standard error around the mean and feature a minimum of 10 seeds.
