from a9t.adapters.nnunetv2 import NNUNetV2Adapter, default_nnunet_adapter
import os


def postprocess_folder(
    input_folder,
    output_folder,
    pkl_file,
    adapter: NNUNetV2Adapter = default_nnunet_adapter,
):
    os.makedirs(output_folder, exist_ok=True)
    adapter.postprocess(input_folder, output_folder, pkl_file)
