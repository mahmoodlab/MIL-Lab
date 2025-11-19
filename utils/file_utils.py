"""
File I/O utilities for MIL-Lab

Handles saving and loading various file formats including HDF5, pickle, etc.
"""

import h5py
import pickle
import os
from pathlib import Path


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    """
    Save assets to HDF5 file

    Args:
        output_path: Path to output HDF5 file
        asset_dict: Dictionary of assets to save {name: data}
        attr_dict: Dictionary of attributes {name: attrs}
        mode: File mode ('w' for write, 'a' for append)

    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, mode) as f:
        for key, val in asset_dict.items():
            if key in f:
                del f[key]

            data_shape = val.shape
            data_type = val.dtype

            chunk_shape = None
            if data_shape[0] > 1:
                chunk_shape = (1,) + data_shape[1:]

            if chunk_shape is not None:
                maxshape = (None,) + data_shape[1:]
                dset = f.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=maxshape,
                    chunks=chunk_shape,
                    dtype=data_type
                )
            else:
                dset = f.create_dataset(key, shape=data_shape, dtype=data_type)

            dset[:] = val

            if attr_dict is not None and key in attr_dict:
                for attr_key, attr_val in attr_dict[key].items():
                    dset.attrs[attr_key] = attr_val

    return output_path


def load_hdf5(file_path, keys=None):
    """
    Load data from HDF5 file

    Args:
        file_path: Path to HDF5 file
        keys: List of keys to load (None = load all)

    Returns:
        Dictionary of loaded data
    """
    data = {}

    with h5py.File(file_path, 'r') as f:
        if keys is None:
            keys = list(f.keys())

        for key in keys:
            if key in f:
                data[key] = f[key][:]

    return data


def save_pkl(file_path, obj):
    """
    Save object to pickle file

    Args:
        file_path: Path to output pickle file
        obj: Object to save
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(file_path):
    """
    Load object from pickle file

    Args:
        file_path: Path to pickle file

    Returns:
        Loaded object
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_slide_id(slide_path, slide_ext='.svs'):
    """
    Extract slide ID from file path

    Args:
        slide_path: Path to slide file
        slide_ext: Slide file extension

    Returns:
        Slide ID (filename without extension)
    """
    filename = os.path.basename(slide_path)
    return filename.replace(slide_ext, '')
