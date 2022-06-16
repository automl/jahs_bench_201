# Downloading the Surrogate Models

The current hosting solution is a transitory one as we work towards setting up a more robust solution using
[Figshare+](https://figshare.com/), which provides perpetual data storage guarantees, a DOI and a web API for
querying the dataset as well as the metadata.

We share our trained models as g-zipped tarballs that are readable using our code base.

The most convenient method for downloading our datasets is through our API. Nevertheless, interested users may directly
download our DataFrames using file transfer software, such as `wget`, from our archive
[here](https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/data/aadsqldb/jahs_bench_201/v1.0.0/assembled_surrogates)

For example, to download the full set of surrogates for CIFAR-10, run

```bash
wget ...
```

## Archive Structure

For each of the three tasks, "cifar10", "colorectal_histology" and "fashion_mnist", the name of the task should be
appended to the above archive link, as "...metric_data/cifar10", in order to access the sub-directory of that dataset.
Following this, users can either recursively download the entire directory tree rooted at that sub-directory and pass
it to our top-level API in order to load all the models for a particular task or downloaded individual sub-directories
and use the more granular API to load surrogates for individual metrics. Each sub-sub-directory is named after the
particular metric the model contained within was trained to predict and contains a number of tarballs that contain the
relevant data needed to load a trained model into memory.

The downloaded models can be individually loaded into memory as:

```python
from jahs_bench_201.surrogate.model import XGBSurrogate
pth = ""  # Full path to a model directory (e.g. "../cifar10/latency")
model = XGBSurrogate.load()
```

The above code snippet, when filled in with a local path to the downloaded tarball, will display the first five rows in
that table.
