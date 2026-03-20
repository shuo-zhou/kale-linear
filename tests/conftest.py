import os

import pytest
from torch.nn.functional import one_hot

from kale.embed.image_cnn import ResNet18Feature
from kale.loaddata.image_access import ImageAccess
from kale.loaddata.multi_domain import MultiDomainAdapDataset


@pytest.fixture(scope="session")
def download_path():
    path = os.path.join("tests", "test_data")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def office_path(download_path):
    office_path = os.path.join(download_path, "office")
    os.makedirs(office_path, exist_ok=True)
    return office_path


@pytest.fixture(scope="session")
def office_test_data(office_path):
    office_caltech_access = ImageAccess.get_multi_domain_images(
        "OFFICE_CALTECH", office_path, download=True, return_domain_label=True
    )
    dataset = MultiDomainAdapDataset(office_caltech_access)
    dataset.prepare_data_loaders()
    dataloader = dataset.get_domain_loaders(split="train", batch_size=100)
    feature_network = ResNet18Feature()
    x, y, z = next(iter(dataloader))
    x = feature_network(x)
    covariate_mat = one_hot(z)
    out = [x, y, z, covariate_mat]
    for i in range(len(out)):
        out[i] = out[i].detach().numpy()

    return out
