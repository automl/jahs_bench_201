# Downloading the Performance Dataset

The current hosting solution is a transitory one as we work towards setting up a more robust solution using
[Figshare+](https://figshare.com/), which provides perpetual data storage guarantees, a DOI and a web API for
querying the dataset as well as the metadata.

Currently, we share all our data in the form of Pandas DataFrames which are very efficient for handling large tables of
data, stored as compressed pickle files using pickle protocol 4. We are aware of the inherent limitations with sharing
pickle files and therefore are investigating the most appropriate data format. Current candidates include CSV, HDF5 and
Feather.

The most convenient method for downloading our datasets is through our API. Nevertheless, interested users may directly
download our DataFrames using file transfer software, such as `wget`, from our archive
[here](https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/data/aadsqldb/jahs_bench_201/v1.0.0/metric_data)

For example, to download the full raw metrics data for CIFAR-10, run

```bash
wget ...
```

## Archive Structure

For each of the three tasks, "cifar10", "colorectal_histology" and "fashion_mnist", the name of the task should be
appended to the above archive link, as "...metric_data/cifar10", in order to access the sub-directory of that dataset.
Following this, users may then append the exact filename they wish to acces to the link. The following 4 files are
available:
* "raw.pkl.gz": This contains the full set of raw performance metrics sans any post-processing or filteration.
* "train_set.pkl.gz": This is the actual training data used for training our surrogate models.
* "valid_set.pkl.gz": This is the actual validation data used for validating the fitness of any given configuration
during HPO.
* "test_set.pkl.gz": This is the actual testing data used for generating the final performance scores of our surrogates
reported in the paper.

For each of these files, users can directly load them into memory as pandas DataFrames, as:

```python
import pandas as pd
pth = ""  # Full path to a downloaded file, ending in ".pkl.gz"
df = pd.read_pickle(pth)
df.head(5)
```

The above code snippet, when filled in with a local path to the downloaded tarball, will display the first five rows in
that table.
