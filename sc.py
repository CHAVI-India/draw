from a9t.converters.nifti2rt import convert_nifti_outputs_to_dicom
dataset_name="TSPrimeOrgans"
preds_dir="data/nnUNet_results/Dataset720_TSPrime/imagesTr_output"
dataset_dir="data/nnUNet_raw/Dataset720_TSPrime"
dataset_id=720
output_dir="data/results/rt"

convert_nifti_outputs_to_dicom(preds_dir, dataset_dir, dataset_id, dataset_name, output_dir)