import os

from a9t.accessor.nnunetv2 import NNUNetV2Adapter, default_nnunet_adapter


def postprocess_folder(
    input_folder,
    output_folder,
    pkl_file,
    adapter: NNUNetV2Adapter = default_nnunet_adapter,
):
    os.makedirs(output_folder, exist_ok=True)
    adapter.postprocess(input_folder, output_folder, pkl_file)