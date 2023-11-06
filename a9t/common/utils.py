import json
import os.path
import shutil
from glob import glob

from pydicom import dcmread

from a9t.constants import BASE_DIR


def clear_out_old_files(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


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
    return f.Modality == "RTSTRUCT"


def get_rt_file_path(dicom_dir) -> str:
    # We expect only 1 RS per dicom series
    return [
        file_name
        for file_name in glob(f"{dicom_dir}/**/**.dcm", recursive=True)
        if is_rt_file(file_name)
    ][0]


def get_files_not_rt(dicom_dir) -> list[str]:
    return [
        file_name
        for file_name in glob(f"{dicom_dir}/**/**.dcm", recursive=True)
        if not is_rt_file(file_name)
    ]


def remove_stuff(path):
    if os.path.exists(path):
        shutil.rmtree(path)
