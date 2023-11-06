import os
from a9t.class_mapping import ALL_SEG_MAP
from a9t.common.nifti2rt import (
    add_to_output_csv,
    convert_multilabel_nifti_to_rtstruct,
    get_dcm_root_from_csv,
)


NNUNET_RESULTS_KEY = "nnUNet_results"


if __name__ == "__main__":
    RESULTS_DIR = os.environ.get(NNUNET_RESULTS_KEY, None)

    if RESULTS_DIR is None:
        raise ValueError(f"Value of {NNUNET_RESULTS_KEY} is not set. Aborting...")

    # TODO: get details from CLI. Also number of samples
    dataset_id = 800
    dataset_name = "TSGyne"
    dataset_tag = "seg"

    number_of_samples = 16
    for sample in range(number_of_samples):
        sample_no = str(sample).zfill(3)
        print("Processing Sample: ", sample_no)
        dcm_root_dir, dcm_parent_folder = get_dcm_root_from_csv(dataset_id, sample_no)
        convert_multilabel_nifti_to_rtstruct(
            nifti_file_path=f"{RESULTS_DIR}/Dataset{dataset_id}_{dataset_name}/imagesTr_predhighres/{dataset_tag}_{sample_no}.nii.gz",
            dicom_dir=dcm_root_dir,
            save_dir=f"{RESULTS_DIR}/Dataset{dataset_id}_{dataset_name}/nnUNetTrainer__nnUNetPlans__3d_fullres/preds/{dcm_parent_folder}",
            label_to_name_map=ALL_SEG_MAP["TSGyne"],
        )

    with open(
        f"{RESULTS_DIR}/Dataset{dataset_id}_{dataset_name}/imagesTr_predhighres/summary.json",
        "r",
    ) as p:
        import json

        summaries = json.load(p)
        summaries = summaries["metric_per_case"]

    with open(
        "data/nnUNet_preprocessed/Dataset800_TSGyne/splits_final.json", "r"
    ) as fp:
        import json

        fold_no = 0
        splits = json.load(fp)
        splits = splits[fold_no]

    add_to_output_csv(
        dataset_id,
        summaries,
        splits,
        f"{RESULTS_DIR}/Dataset{dataset_id}_{dataset_name}/nnUNetTrainer__nnUNetPlans__3d_fullres/preds",
    )
