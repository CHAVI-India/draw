import nibabel as nib
import numpy as np
import torch
from einops import repeat
from nnunetv2.training.loss.dice import SoftDiceLoss

l = SoftDiceLoss(smooth=0, batch_dice=True, ddp=False)


def get_tensor(f_path):
    nb = nib.load(f_path)
    new_var = torch.from_numpy(np.asanyarray(nb.dataobj))
    return repeat(torch.tensor(new_var, dtype=torch.bfloat16), "x y z -> 2 1 x y z")


pred = get_tensor(
    "data/nnUNet_results/Dataset720_TSPrime/imagesTr_predhighres/seg_000.nii.gz"
)
gt = get_tensor("data/nnUNet_raw/Dataset720_TSPrime/labelsTr/seg_000.nii.gz")
mask = gt == 7

print(pred.shape, gt.shape, mask.shape)

print(pred.isnan().any())

print(l(gt, pred, loss_mask=mask))
