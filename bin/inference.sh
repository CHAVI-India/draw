source ./env.sh
nnUNetv2_predict \
-i $nnUNet_raw/Dataset720_TSPrime/imagesTr \
-o $nnUNet_results/Dataset720_TSPrime/imagesTr_predhighres \
-c 3d_fullres \
-d 720 \
-f 0 \
--verbose \
-chk "checkpoint_best.pth" \
-npp 1 \
-nps 1 \
-num_parts 1 \
-part_id 0 \
-device 'cuda'
