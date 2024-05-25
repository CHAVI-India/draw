import json
import os
import os.path
import shutil
from glob import glob
from pathlib import Path
import subprocess as sp

from pydicom import dcmread

from draw.config import RTSTRUCT_STRING, DCM_REGEX, LOG, DicomKeyToTag


def normpath(path):
    return os.path.normpath(path)


def get_all_folders_from_raw_dir(dir_path):
    return [f.path for f in os.scandir(dir_path) if f.is_dir()]


def read_json(file_path):
    with open(file_path, "r") as fp:
        j = json.load(fp)
    return j


def write_json(dict_obj, file_path):
    json.dump(dict_obj, open(file_path, "w"), sort_keys=False, indent=4)


def is_rt_file(file_name):
    f = dcmread(file_name)
    return f.Modality == RTSTRUCT_STRING


def get_rt_file_path(dicom_dir) -> str:
    # We expect only 1 RS per dicom series
    return [
        file_name
        for file_name in glob(normpath(f"{dicom_dir}/{DCM_REGEX}"), recursive=True)
        if is_rt_file(file_name)
    ][0]


def get_files_not_rt(dicom_dir) -> list[str]:
    return [
        file_name
        for file_name in glob(normpath(f"{dicom_dir}/{DCM_REGEX}"), recursive=True)
        if not is_rt_file(file_name)
    ]


def remove_stuff(path):
    if os.path.exists(path):
        LOG.info(f"Deleting {path}")
        shutil.rmtree(path)


def get_immediate_dicom_parent_dir(dicom_dir):
    one_dcm_path = get_one_dcm_path(dicom_dir)
    return str(one_dcm_path.parent)


def get_one_dcm_path(dicom_dir):
    one_dcm_path = glob(normpath(f"{dicom_dir}/{DCM_REGEX}"), recursive=True)[0]
    one_dcm_path = Path(one_dcm_path)
    return one_dcm_path


def get_all_dicom_dirs(root_dir):
    all_dicom_dirs = get_all_folders_from_raw_dir(root_dir)
    return all_dicom_dirs


def assert_env_key_set(key):
    base_dir = os.environ.get(key, None)
    if base_dir is None:
        raise ValueError(f"Value of {key} is not set. Aborting...")
    return base_dir


def copy_filtered_files(src_dir, dst_base_dir, filter_fxn):
    updated_dir_path = None
    dcm_parent_dir = get_immediate_dicom_parent_dir(src_dir)
    for p in glob.glob(DCM_REGEX, recursive=True, root_dir=src_dir):
        if filter_fxn(os.path.join(src_dir, p)):
            updated_dir_path = os.path.join(
                dst_base_dir, os.path.basename(dcm_parent_dir)
            )
            os.makedirs(updated_dir_path, exist_ok=True)
            shutil.copy(os.path.join(src_dir, p), os.path.join(dst_base_dir, p))
    return updated_dir_path


def get_dicom_attribute_from_dir(dir_path: str, attrib_key: tuple) -> str: #| None
    f = dcmread(str(get_one_dcm_path(dir_path)))
    if attrib_key in f:
        return str(f[attrib_key].value)
    return None


def copy_input_dcm_to_output(input_dir, output_dir):
    # This triggers WatchDog maybe?
    LOG.debug(f"Copying {input_dir} -> {output_dir}")
    shutil.copytree(src=input_dir, dst=output_dir, dirs_exist_ok=True)


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return int(memory_free_values[0])
