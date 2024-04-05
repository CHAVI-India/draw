import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import nibabel as nib
import SimpleITK as sitk
import matplotlib.colors as colors

# Define custom colors for each label
custom_colors = {
    0: 'black',          # Background
    1: 'blue',           # Bladder
    2: 'green',          # Anorectum
    3: 'red',            # Bag_Bowel
    4: 'yellow',         # Femur_Head_L
    5: 'orange',         # Femur_Head_R
    6: 'purple',         # Penilebulb
}
custom_cmap = colors.ListedColormap([custom_colors[i] for i in range(7)])

def save_axial_slice_images_with_overlay(original_img_path, seg_mask_path, output_dir):
    # Load the original image and segmentation mask
    original_img = nib.load(original_img_path)
    seg_mask = nib.load(seg_mask_path)

    # Get the image data as numpy arrays
    original_data = original_img.get_fdata()
    mask_data = seg_mask.get_fdata()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through axial slices and save images with overlay
    num_slices = original_data.shape[2]
    for slice_idx in range(num_slices):
        # Plot the original image
        plt.imshow(original_data[:, :, slice_idx], cmap="gray")

        # Overlay the segmentation mask
        masked_mask_data = np.ma.masked_where(mask_data[:, :, slice_idx] == 0, mask_data[:, :, slice_idx])

# Overlay the segmentation mask using a different colormap
        plt.imshow(masked_mask_data, alpha=0.3, cmap="jet")  # Using "jet" colormap for non-zero mask values

        # Remove axis
        plt.axis('off')

        # Set the title
        plt.title(f"Axial Slice {slice_idx+1}")

        # Save the plot as an image in the output directory
        plt.savefig(os.path.join(output_dir, f"{slice_idx+1}.png"), bbox_inches='tight', pad_inches=0)

        # Clear the plot for the next iteration
        plt.clf()


def get_array(fn):
    "opens .nii file and return the array"
    img = sitk.ReadImage(str(fn))
    imgd = sitk.GetArrayFromImage(img)
    return imgd

def overlay_segmentation_mask(imgd, maskd, sli, output_dir):
    "Overlay segmentation mask on top of the MRI slice, annotate slice number and save the overlay image"
    alpha = 0.5  # Transparency level for the segmentation mask
    overlay = np.zeros_like(maskd, dtype=np.float32)
    overlay[maskd != 0] = alpha  # Set alpha to 0.5 where mask is non-zero
    plt.imshow(imgd[sli], cmap='gray')
    plt.imshow(maskd[sli], cmap=custom_cmap, alpha=overlay[sli])
    plt.axis('off')
    plt.text(0.5, 0.95, f"Slice {sli+1}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='white')
    plt.savefig(os.path.join(output_dir, f'{sli+1}.png'))
    plt.close()

def overlay_slices(fn, mask_fn, output_dir):
    imgd = get_array(fn)
    maskd = get_array(mask_fn)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_slices = imgd.shape[0]
    for sli in range(num_slices):
        overlay_segmentation_mask(imgd, maskd, sli, output_dir)

# overlay_slices('BraTS20_Training_002_t1ce.nii', 'BraTS20_Training_002_seg.nii', 'slices')


import os
import numpy as np
import nibabel as nib
from rt_utils import RTStructBuilder

def convert_nifti_to_rtstruct(
    nifti_file_path: str,
    dicom_dir: str,
    save_dir: str,
    rt_file_name: str,
) -> str:
    """Convert a NIfTI segmentation mask to RTStruct DICOM"""
    seg_map = {
    0: "background",
    1: "Bladder",
    2: "Anorectum",
    3: "Bag_Bowel",
    4: "Femur_Head_L",
    5: "Femur_Head_R",
    6: "Penilebulb"
}
    os.makedirs(save_dir, exist_ok=True)
    rt_path = os.path.join(save_dir, rt_file_name)

    rtstruct = build_rt_struct(dicom_dir, rt_path)
    np_mask = make_mask_from_rt(nifti_file_path)

    for idx, name in seg_map.items():
        curr_mask = np_mask == idx
        rtstruct.add_roi(mask=curr_mask, name=name)

    rtstruct.save(rt_path)
    print(f"RTStruct saved at: {rt_path}")
    return save_dir

def make_mask_from_rt(nifti_file_path):
    nifti_mask = nib.load(nifti_file_path)
    np_mask = np.asanyarray(nifti_mask.dataobj)
    np_mask = np.transpose(np_mask, [1, 0, 2])
    return np_mask

def build_rt_struct(dicom_dir, rt_path):
    if os.path.exists(rt_path):
        rtstruct = RTStructBuilder.create_from(dicom_dir, rt_path)
    else:
        rtstruct = RTStructBuilder.create_new(dicom_dir)
    return rtstruct

