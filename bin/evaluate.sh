source bin/env.sh

nnUNetv2_evaluate_folder data/nnUNet_raw/Dataset720_TSPrime/labelsTr \
    data/nnUNet_results/Dataset720_TSPrime/imagesTr_predhighres \
    -djfile data/nnUNet_results/Dataset720_TSPrime/imagesTr_predhighres/dataset.json \
    -pfile data/nnUNet_results/Dataset720_TSPrime/imagesTr_predhighres/plans.json \
    --chill
