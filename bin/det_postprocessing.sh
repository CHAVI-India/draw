# options:
#   -h, --help            show this help message and exit
#   -i I                  Input folder
#   -ref REF              Folder with gt labels
#   -plans_json PLANS_JSON
#                         plans file to use. If not specified we will look for the plans.json file in the input folder (input_folder/plans.json)
#   -dataset_json DATASET_JSON
#                         dataset.json file to use. If not specified we will look for the dataset.json file in the input folder (input_folder/dataset.json)
#   -np NP                number of processes to use. Default: 8
#   --remove_postprocessed
#                         set this is you don't want to keep the postprocessed files

source ./bin/env.sh

nnUNetv2_determine_postprocessing \
    -i data/nnUNet_results/Dataset720_TSPrime/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/validation \
    -ref data/nnUNet_preprocessed/Dataset720_TSPrime/gt_segmentations \
    -plans_json data/nnUNet_results/Dataset720_TSPrime/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/plans.json \
    -dataset_json data/nnUNet_results/Dataset720_TSPrime/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/dataset.json \
    --remove_postprocessed
