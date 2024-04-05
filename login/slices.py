import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk

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
    plt.imshow(maskd[sli], cmap='jet', alpha=overlay[sli])
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
