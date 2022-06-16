Currently, we share all our data in the form of Pandas DataFrames which are very efficient for handling large tables of
data, stored as compressed pickle files using pickle protocol 4. The current hosting solution is a transitory one as we
work towards setting up a more robust solution using [Figshare+](https://figshare.com/), which provides perpetual data
storage guarantees, a DOI and a web API for querying the dataset as well as the metadata. Additionally, we are aware of
the inherent isssues with sharing pickle files and therefore are investigating the most appropriate data format.
Current candidates include CSV, HDF5 and Feather.
