source bin/env.sh

PRED=data/nnUNet_results/Dataset800_TSGyne/imagesTr_predhighres
LABELS=data/nnUNet_raw/Dataset800_TSGyne/labelsTr

nnUNetv2_evaluate_folder $LABELS\
    $PRED \
    -djfile $PRED/dataset.json \
    -pfile $PRED/plans.json \
    --chill
