import os

from draw.accessor.nnunetv2 import NNUNetV2Adapter, default_nnunet_adapter


def postprocess_folder(
    input_folder,
    output_folder,
    pkl_file,
    adapter: NNUNetV2Adapter = default_nnunet_adapter,
):
    os.makedirs(output_folder, exist_ok=True)
    adapter.apply_postprocessing(input_folder, output_folder, pkl_file)
