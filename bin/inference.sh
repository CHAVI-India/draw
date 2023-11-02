source ./bin/env.sh
nnUNetv2_predict \
    -i $nnUNet_raw/Dataset720_TSPrime/imagesTr \
    -o $nnUNet_results/Dataset720_TSPrime/imagesTr_predhighres_NoMirror \
    -c 3d_fullres \
    -d 720 \
    -t nnUNetTrainerNoMirroring \
    -f 0 \
    --verbose \
    -chk "checkpoint_best.pth" \
    -device 'cuda' \
    -npp 6 \
    -nps 6
