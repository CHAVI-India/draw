import os

from django.conf import settings
from slices import *

def compute_slices(input_folder,output_folder,bpart):
     for file_name in os.listdir(output_folder):
                    if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                       parts = file_name.split(".")
                       fol_name = parts[0]
                       sfol_name=os.path.join(output_folder,fol_name)
                       os.makedirs(sfol_name, exist_ok=True)
                       infol_name=fol_name+"_0000.nii.gz"
                       input_file_path = os.path.join(input_folder, infol_name)
                       oflp=os.path.join(output_folder, file_name)
                       print(fol_name)
                       print(input_file_path)
                       print(oflp)
                       overlay_slices(input_file_path,oflp,os.path.join(output_folder, fol_name))

user_folder = os.path.join("/media/neeraj/E/Documents/BTP/autoseg_web/media", "users_data", "batman")
bp="brain"
body_part_folder = os.path.join(user_folder, bp)
timestamp_folder = os.path.join(body_part_folder, "2024-03-12_11-23-01")
output_folder = os.path.join(timestamp_folder, "output")
input_folder = os.path.join(timestamp_folder, "input")
compute_slices(input_folder,output_folder,"brain")

#overlay_slices('/media/neeraj/E/Documents/BTP/autoseg_web/media/users_data/batman/brain/2024-03-12_11-23-01/input/Brats_001_0000.nii.gz', '/media/neeraj/E/Documents/BTP/autoseg_web/media/users_data/batman/brain/2024-03-12_11-23-01/output/Brats_001.nii.gz', 'slices1')