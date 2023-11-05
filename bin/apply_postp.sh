#   -h, --help            show this help message and exit
#   -i I                  Input folder
#   -o O                  Output folder
#   -pp_pkl_file PP_PKL_FILE
#                         postprocessing.pkl file
#   -np NP                number of processes to use. Default: 8
#   -plans_json PLANS_JSON
#                         plans file to use. If not specified we will look for the plans.json file in the input folder (input_folder/plans.json)
#   -dataset_json DATASET_JSON
#                         dataset.json file to use. If not specified we will look for the dataset.json file in the input folder (input_folder/dataset.json)
source ./bin/env.sh

nnUNetv2_apply_postprocessing \
    -i data/nnUNet_results/Dataset720_TSPrime/imagesTr_predhighres_NoMirror \
    -o data/nnUNet_results/Dataset720_TSPrime/imagesTr_output \
    -pp_pkl data/nnUNet_results/Dataset720_TSPrime/imagesTr_predhighres_NoMirror/postprocessing.pkl \
    -np 10
