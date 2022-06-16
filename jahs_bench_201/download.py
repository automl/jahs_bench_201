import requests
import tarfile
from pathlib import Path

surrogate_url = "https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.0.0/assembled_surrogates.tar"
metric_url = "https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.0.0/metric_data.tar"


def download_and_extract_url(url, save_dir, filename):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_tar_file = save_dir / filename

    print(f"Starting download of {url}, this might take a while.")
    with requests.get(url, stream=True) as response:
        with open(save_tar_file, 'wb') as f:
            f.write(response.raw.read())

    print("Download finished, extracting now")
    with tarfile.open(save_tar_file, 'r') as f:
        f.extractall(path=save_dir)
    print("Done extracting")


if __name__ == "__main__":
    download_and_extract_url(surrogate_url, "jahs_bench_201_data", "assembled_surrogates.tar")
    download_and_extract_url(metric_url, "jahs_bench_201_data", "metric_data.tar")
