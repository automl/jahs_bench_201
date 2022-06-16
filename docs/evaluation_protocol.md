For scientific studies we recommend an evaluation protocol when using JAHS-Bench-201 here to facilitate fair, reproducible, and methodologically sound comparisons.


## Optimization Tasks


Studies with JAHS-Bench-201 should report on all three datasets and present results as trajectories of
best validation error (single-objective) or hypervolume (multi-objective).


## Seeds and Errors Bounds

Results should report the mean and standard error around the mean and feature a minimum of 10 seeds.

## Reporting Across Runtimes

To allow using the same evaluation protocol and comparisons across various algorithmic settings the objective trajectories
should be reported across total runtime taking into account the training and evaluation costs predicted
by the surrogate benchmark. We suggest to report until a runtime corresponding to approximately
100 evaluations and, for interpretability, show the total runtime divided by the
mean runtime of one evaluation.

## (Multi) Multi-fidelity

For (multi) multi-fidelity runs utilizing epochs, we support both using continuations from few to many epochs (to simulate the checkpointing and continuation of
the model) as well as retraining from scratch for optimizers that do not support continuations.
