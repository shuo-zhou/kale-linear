from pathlib import Path
from urllib.request import urlretrieve


def download_path():
    path = Path("tests").joinpath("test_data")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def download_gait_gallery_data(save_path):
    # Downloading gait gallery data for tests/conftest.py test
    url = "https://github.com/pykale/data/raw/main/videos/gait/gait_gallery_data.mat"
    # Download the file and save it to the specified path
    output_file = Path(save_path).joinpath("gait.mat")
    urlretrieve(url, str(output_file))


def download_mpca_data(save_path):
    # Downloading MPCA data for tests/embed/test_factorization.py test
    url = "https://github.com/pykale/data/raw/main/videos/gait/mpca_baseline.mat"
    # Download the file and save it to the specified path
    output_file = Path(save_path).joinpath("baseline.mat")
    urlretrieve(url, str(output_file))
