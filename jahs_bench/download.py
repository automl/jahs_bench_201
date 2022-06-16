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


def download_surrogates(save_dir="jahs_bench_data"):
    download_and_extract_url(surrogate_url, save_dir, "assembled_surrogates.tar")


def download_metrics(save_dir="jahs_bench_data"):
    download_and_extract_url(metric_url, save_dir, "metric_data.tar")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="surrogates", choices=["surrogates", "metrics", "all"])
    parser.add_argument("--save_dir", default="jahs_bench_data")
    args = parser.parse_args()

    if args.target == "surrogates":
        download_surrogates(args.save_dir)
    elif args.target == "metrics":
        download_metrics(args.save_dir)
    else:
        download_surrogates(args.save_dir)
        download_metrics(args.save_dir)
