#!/bin/bash

quick=false
gpu=0
dataset=tempoTL

while getopts ":HG:Q" opt; do
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
    ./experiments/eval_released.sh -D $dataset -G $gpu -M tall
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mllc-global
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mllc-ba
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mllc-ws
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mllc-ws-conTEF
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mllc-ss 
    ./experiments/eval_released.sh -D $dataset -G $gpu -M mllc
fi

echo "RESULTS: MCN (row 1)"
python utils/fusion.py --rgb_tag $dataset/emnlp2018_rgb_mcn_$dataset \
                       --flow_tag $dataset/emnlp2018_flow_mcn_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet

echo "RESULTS: Tall (row 2)"
python utils/fusion.py --rgb_tag $dataset/emnlp2018_rgb_tall_$dataset \
                       --flow_tag $dataset/emnlp2018_flow_tall_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --tall \
                       --quiet 

echo "RESULTS: MLLC-global (row 3)"
python utils/fusion.py --rgb_tag $dataset/emnlp2018_rgb_mllc-global_$dataset \
                       --flow_tag $dataset/emnlp2018_flow_mllc-global_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet 

echo "RESULTS: MLLC-before/after (row 4)"
python utils/fusion.py --rgb_tag $dataset/emnlp2018_rgb_mllc-ba_$dataset \
                       --flow_tag $dataset/emnlp2018_flow_mllc-ba_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --tall \
                       --quiet 

echo "RESULTS: MLLC weak supervision (row 5)"
python utils/fusion.py --rgb_tag $dataset/emnlp2018_rgb_mllc-ws-conTEF_$dataset \
                       --flow_tag $dataset/emnlp2018_flow_mllc-ws-conTEF_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet 
 
echo "RESULTS: MLLC weak supervision w/conTEF feature, no norm (row 6)"
python utils/fusion.py --rgb_tag $dataset/emnlp2018_rgb_mllc-ws-conTEF_$dataset \
                       --flow_tag $dataset/emnlp2018_flow_mllc-ws-conTEF_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet 

echo "RESULTS: MLLC strong supervision (row 7)"
python utils/fusion.py --rgb_tag $dataset/emnlp2018_rgb_mllc-ss-no-conTEF_$dataset \
                       --flow_tag $dataset/emnlp2018_flow_mllc-ss-no-conTEF_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet 

echo "RESULTS: MLLC (row 8)"
python utils/fusion.py --rgb_tag $dataset/emnlp2018_rgb_mllc_$dataset \
                       --flow_tag $dataset/emnlp2018_flow_mllc_$dataset \
                       --iter 45000 \
                       --dataset $dataset \
                       --quiet  
