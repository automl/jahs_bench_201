# Downloading the Surrogate Models

The current hosting solution is a transitory one as we work towards setting up a more robust solution using
[Figshare+](https://figshare.com/), which provides perpetual data storage guarantees, a DOI and a web API for
querying the dataset as well as the metadata.

We share our trained models as compressed tarballs that are readable using our code base.

The most convenient method for downloading our models is through our [API](https://automl.github.io/jahs_bench_201/).
Nevertheless, interested users may directly download our DataFrames using a file transfer software of their choice,
such as `wget`, from our archive.

To download the full set of all surrogates models, run

```bash
wget --no-parent -r https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.0.0/assembled_surrogates.tar -O assembled_surrogates.tar
tar -xf assembled_surrogates.tar
```

## Archive Structure

For each of the three tasks, "cifar10", "colorectal_histology" and "fashion_mnist", the name of the task is the
immediate sub-directory within "metric_data" and contains all the models pertaining to that task.
Immediately under each task's directory, are further sub-directories named after the individual metrics the models they
contains were trained to predict. These sub-directories can be directly passed to `jahs_bench.surrogate.model.XGBSurrogate.load()`
in order to load the respective models into memory.

The downloaded models can be individually loaded into memory as:

```python
from jahs_bench.surrogate.model import XGBSurrogate

pth = "assembled_surrogates/cifar10/latency"  # Path to a model directory
model = XGBSurrogate.load(pth)
```

The advantage to directly loading a model in this manner is that the `model.predict()` method is able to process
entire DataFrames of queries and return a corresponding DataFrame of predicted metrics.
