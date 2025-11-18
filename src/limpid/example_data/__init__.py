import os
import numpy as np
from importlib import resources


ddir = resources.files("limpid").joinpath("example_data")
datasets = [d.stem for d in ddir.iterdir() if d.suffix == '.csv']

def load_example_data(dname: str):
    """Load example dataset.

    Args:
      dname: Name of the dataset to read (without extension). Must be a valid
        dataset located in limpid.example_data.

    Returns:
      Three lists in the order [energy, lineshape, lineshape_delta].
    """

    # Check extension
    d, ext = os.path.splitext(dname)
    if ext.lower() == ".csv":
        dname = d
    # Check that dataset exist
    if dname not in datasets:
        err = (f'Dataset does not exist. Valid datasets names are: {datasets}')
        raise ValueError(err)
    # Load dataset
    with resources.as_file(
        resources.files("limpid").joinpath("example_data", dname + ".csv")
    ) as filepath:
        data =  np.transpose(np.genfromtxt(filepath, delimiter=","))
    return data

def list_example_data():
    """List available example datasets.

    Returns:
      A list containing names of all datasets present in limpid.example_data.
    """

    return datasets
