import requests
import tarfile
from pathlib import Path
from enum import Enum

class KnownVersions(Enum):
    v1_0_0 = "v1.0.0"
    v1_1_0 = "v1.1.0"


class FileNames(Enum):
    surrogates = "assembled_surrogates.tar"
    metrics = "metric_data.tar"
    tasks = "datasets.tar"


latest_version = KnownVersions.v1_1_0
BASE_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/"


def _construct_url(filename: FileNames, version: KnownVersions = latest_version):
    return "/".join([BASE_URL, version.value, filename.value])


def _download_and_extract_url(url, save_dir, filename):
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
    print("Done extracting.")


def download(filename: FileNames, version: KnownVersions, save_dir: Path):
    url = _construct_url(filename=filename, version=KnownVersions(version))
    _download_and_extract_url(url=url, save_dir=save_dir, filename=filename.value)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="surrogates",
                        choices=[m.name for m in FileNames] + ["all"])
    parser.add_argument("--save_dir", default="jahs_bench_data")
    parser.add_argument("--version", default=latest_version,
                        choices=[m.value for m in KnownVersions])
    args = parser.parse_args()

    targets = list(FileNames) if args.target == "all" else [FileNames[args.target]]
    for t in targets:
        download(filename=t, version=KnownVersions(args.version), save_dir=args.save_dir)
