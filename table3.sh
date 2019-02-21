#!/bin/bash

quick=false
gpu=0
dataset=didemo

while getopts ":HQ" opt; do
    case $opt in
        H) echo "Script will run the experiments corresponding to table 4 in the paper"
        
           echo "Flag 'Q' (optional): if you have run evaluation before, intermediate values will be cached.  Adding -Q will speed up computation"
        ;;
        Q) quick=true
        ;;
        G) gpu="$OPTARG"
        esac
done

#Compute/cache intermediate outputs
if [ $quick == false ] ; then
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mcn
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mcn-tall-loss
    ./experiments/eval_released.sh -D $dataset -G $gpu -M tall-noTEF
    ./experiments/eval_released.sh -D $dataset -G $gpu -M tall
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mcn-tall-sim
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mcn-mult-no-norm
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mllc
fi

echo "RESULTS: MCN (row 1)"
python utils/fusion.py --rgb_tag emnlp2018_rgb_mcn_$dataset \
                       --flow_tag emnlp2018_flow_mcn_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet

echo "RESULTS: MCN, tall loss (row 2)"
python utils/fusion.py --rgb_tag emnlp2018_rgb_mcn-tall-loss_$dataset \
                       --flow_tag emnlp2018_flow_mcn-tall-loss_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --tall \
                       --quiet 

echo "RESULTS: Tall (row 2)"
python utils/fusion.py --rgb_tag emnlp2018_rgb_tall-noTEF_$dataset \
                       --flow_tag emnlp2018_flow_tall-noTEF_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --tall \
                       --quiet 

echo "RESULTS: Tall w/TEF (row 3)"
python utils/fusion.py --rgb_tag emnlp2018_rgb_tall_$dataset \
                       --flow_tag emnlp2018_flow_tall_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --tall \
                       --quiet 

echo "RESULTS: MCN w/Tall feature (row 4)"
python utils/fusion.py --rgb_tag emnlp2018_rgb_mcn-tall-feature_$dataset \
                       --flow_tag emnlp2018_flow_mcn-tall-feature_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet 
 
echo "RESULTS: MCN w/Mult feature, no norm (row 6)"
python utils/fusion.py --rgb_tag emnlp2018_rgb_mcn-mult-no-norm_$dataset \
                       --flow_tag emnlp2018_flow_mcn-mult-no-norm_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet 

echo "RESULTS: MLLC (row 6)"
python utils/fusion.py --rgb_tag emnlp2018_rgb_mllc_$dataset \
                       --flow_tag emnlp2018_flow_mllc_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet 
 
