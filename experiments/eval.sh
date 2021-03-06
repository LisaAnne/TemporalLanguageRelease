#!/bin/bash

quick=false

while getopts ":HG:M:D:R:F:S:Q" opt; do
  case $opt in
    H) echo "Bash script to evaluate released models on tempoTL:"
       echo ""
       echo "   Flag 'R' (required): path with RGB model "
       echo ""
       echo "   Flag 'F' (required): path with flow model "
       echo ""
       echo "   Flag 'D' (required): indicate dataset tempoTL, tempoHL, or didemo"
       echo ""
       echo "   Flag 'M' (required): indicate which model to evaluate"
       echo "       Options:"
       echo "            mcn, mcn-tall-loss, tall-noTEF, tall, mllc-global, mllc-ba, mllc-ws, mllc-ws-conTEF, mlls-ss, mllc"
       echo ""
       echo "   Flag 'S' (optional): indicate which folder models saved to; default 'snapshots'.  Released models are in 'released_models'"
       echo ""
       echo "   Flag 'G' (optional): indicate which GPU to run on"
       echo ""
       echo "   Flag 'Q' (optional): if you have run evaluation before, intermediate values will be cached.  Adding -Q will speed up computation"
       exit 1
    ;;
    R) rgb="$OPTARG"
    ;;
    F) flow="$OPTARG"
    ;;
    G) gpu="$OPTARG"
    ;;
    M) model="$OPTARG"
    ;;
    D) dataset="$OPTARG"
    ;;
    S) snapshot_folder="$OPTARG"
    ;;
    Q) quick=true
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [[ -z $gpu ]] ; then
    echo 'Did not indicate GPU: using default GPU (0)'
    gpu=0
fi
if [[ -z $snapshot_folder ]] ; then
    echo 'Did not indicate snapshot folder: using default GPU ("snapshots")'
    snapshot_folder=snapshots
fi

echo "GPU: " $gpu
echo "Evaluating model: " $model
echo "Dataset: " $dataset

#Parameters shared across all models

DROPOUT_VISUAL=0.3
DROPOUT_LANGUAGE=0.3
LANGUAGE_LAYERS=lstm_no_embed
FEATURE_PROCESS_LANGUAGE=recurrent_embedding
MAX_ITER=45000
SNAPSHOT=45000
STEPSIZE=15000
BASE_LR=0.05
RANDOM_SEED=1701
LW_INTER=0.2
BATCH_SIZE=120
loc_flag=loc
test_set='val+test'
loss='ranking'

if [ "$dataset" == tempoTL ]; then
    TRAIN_JSON=data/tempoTL+didemo_train.json
    VAL_JSON=data/tempoTL+didemo_val.json
    TEST_JSON=data/tempoTL+didemo_test.json
elif [ "$dataset" == tempoHL ]; then
    TRAIN_JSON=data/tempoHL+didemo_train.json
    VAL_JSON=data/tempoHL+didemo_val.json
    TEST_JSON=data/tempoHL+didemo_test.json
elif [ "$dataset" == didemo ]; then
    TRAIN_JSON=data/didemo_train.json
    VAL_JSON=data/didemo_val.json
    TEST_JSON=data/didemo_test.json
    test_set='val'
else
    echo "-D (dataset) must be 'tempoTL', 'tempoHL' or 'didemo'"
    exit 1
fi

#Defining parameters for specific models

if [ "$model" == mllc ]; then
    INPUT_VISUAL_DATA="relational"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_norm
    LOSS_TYPE=triplet
    DISTANCE_FUNCTION=early_combine_mult
    LW_INTER=0.2
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=500
elif [ "$model" == mllc-ws ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_before_after
    LOSS_TYPE=triplet
    DISTANCE_FUNCTION=early_combine_mult
    LW_INTER=0.2
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=500
elif [ "$model" == mllc-ws-conTEF ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_before_after
    LOSS_TYPE=triplet
    DISTANCE_FUNCTION=early_combine_mult
    LW_INTER=0.2
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=500
elif [ "$model" == mllc-ss ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_before_after
    LOSS_TYPE=triplet
    DISTANCE_FUNCTION=early_combine_mult
    LW_INTER=0.2
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=500
elif [ "$model" == mllc-ba ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_before_after
    LOSS_TYPE=triplet
    DISTANCE_FUNCTION=early_combine_mult
    LW_INTER=0.2
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=500
elif [ "$model" == mllc-global ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_context
    LOSS_TYPE=triplet
    DISTANCE_FUNCTION=early_combine_mult
    LW_INTER=0.2
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=500
elif [ "$model" == tall ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_before_after
    LOSS_TYPE=intra
    DISTANCE_FUNCTION=tall_distance
    VISUAL_EMBEDDING_DIM_1=100
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=100
    loss=tall
elif [ "$model" == tall-noTEF ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_before_after
    LOSS_TYPE=intra
    DISTANCE_FUNCTION=tall_distance
    VISUAL_EMBEDDING_DIM_1=100
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=100
    MAX_ITER=45000 
    loc_flag="no-loc"
elif [ "$model" == mcn ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=2
    FEATURE_PROCESS_VISUAL=feature_process_context
    DISTANCE_FUNCTION=euclidean_distance
    LW_INTER=0.2
    LOSS_TYPE=triplet
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=100
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=100
elif [ "$model" == mcn-tall-loss ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=2
    FEATURE_PROCESS_VISUAL=feature_process_context
    DISTANCE_FUNCTION=euclidean_distance
    LW_INTER=0.2
    LOSS_TYPE=triplet
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=100
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=100
    loss=tall
elif [ "$model" == mcn-tall-sim ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_context
    DISTANCE_FUNCTION=early_combine_mult_tall
    LW_INTER=0.2
    LOSS_TYPE=triplet
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=500
elif [ "$model" == mcn-mult-no-norm ]; then
    INPUT_VISUAL_DATA="clip"
    VISION_LAYERS=1
    FEATURE_PROCESS_VISUAL=feature_process_context
    DISTANCE_FUNCTION=early_combine_mult_no_norm
    LW_INTER=0.2
    LOSS_TYPE=triplet
    VISUAL_EMBEDDING_DIM_1=500
    VISUAL_EMBEDDING_DIM_2=''
    LANGUAGE_EMBEDDING_DIM_1=1000
    LANGUAGE_EMBEDDING_DIM_2=500
else
    echo "-M (model) must be 'mcn', 'mcn-tall-loss', 'tall-noTEF', 'tall', 'mllc-global', 'mllc-ba', 'mllc-ws', 'mllc-ws-conTEF', 'mllc-ss', 'mllc'"
    exit 1
fi


if [ $quick == false ] ; then

    ##rgb
    rgb_data=data/average_rgb_feats.h5
    TRAIN_H5=$rgb_data
    TEST_H5=$rgb_data

    #test on val
    python utils/build_net.py --feature_process_visual $FEATURE_PROCESS_VISUAL  \
                        --$loc_flag \
                        --vision_layers $VISION_LAYERS \
                        --dropout_visual $DROPOUT_VISUAL \
                        --dropout_language $DROPOUT_LANGUAGE \
                        --language_layers $LANGUAGE_LAYERS \
                        --feature_process_language $FEATURE_PROCESS_LANGUAGE \
                        --visual_embedding_dim $VISUAL_EMBEDDING_DIM_1 $VISUAL_EMBEDDING_DIM_2 \
                        --language_embedding_dim $LANGUAGE_EMBEDDING_DIM_1 $LANGUAGE_EMBEDDING_DIM_2 \
                        --gpu $gpu \
                        --max_iter $MAX_ITER \
                        --snapshot $SNAPSHOT \
                        --stepsize $MAX_ITER \
                        --base_lr $BASE_LR \
                        --train_json $TRAIN_JSON \
                        --test_json $VAL_JSON \
                        --train_h5 $TRAIN_H5 \
                        --test_h5 $TEST_H5 \
                        --random_seed $RANDOM_SEED \
                        --loss_type $LOSS_TYPE \
                        --lw_inter $LW_INTER \
                        --batch_size $BATCH_SIZE \
                        --distance_function $DISTANCE_FUNCTION \
                        --pool_type max \
                        --strong_supervise \
                        --input_visual_data $INPUT_VISUAL_DATA \
                        --snapshot_folder $snapshot_folder \
                        --tag $rgb 
    
    #test on test 
    python utils/build_net.py --feature_process_visual $FEATURE_PROCESS_VISUAL  \
                        --$loc_flag \
                        --vision_layers $VISION_LAYERS \
                        --dropout_visual $DROPOUT_VISUAL \
                        --dropout_language $DROPOUT_LANGUAGE \
                        --language_layers $LANGUAGE_LAYERS \
                        --feature_process_language $FEATURE_PROCESS_LANGUAGE \
                        --visual_embedding_dim $VISUAL_EMBEDDING_DIM_1 $VISUAL_EMBEDDING_DIM_2 \
                        --language_embedding_dim $LANGUAGE_EMBEDDING_DIM_1 $LANGUAGE_EMBEDDING_DIM_2 \
                        --gpu $gpu \
                        --max_iter $MAX_ITER \
                        --snapshot $SNAPSHOT \
                        --stepsize $MAX_ITER \
                        --base_lr $BASE_LR \
                        --train_json $TRAIN_JSON \
                        --test_json $TEST_JSON \
                        --train_h5 $TRAIN_H5 \
                        --test_h5 $TEST_H5 \
                        --random_seed $RANDOM_SEED \
                        --loss_type $LOSS_TYPE \
                        --lw_inter $LW_INTER \
                        --batch_size $BATCH_SIZE \
                        --distance_function $DISTANCE_FUNCTION \
                        --pool_type max \
                        --strong_supervise \
                        --snapshot_folder $snapshot_folder \
                        --input_visual_data $INPUT_VISUAL_DATA \
                        --tag $rgb 
    

    #flow on test
    flow_data=data/average_flow_feats.h5
    TRAIN_H5=$flow_data
    TEST_H5=$flow_data
    
    #test on val
    python utils/build_net.py --feature_process_visual $FEATURE_PROCESS_VISUAL  \
                        --$loc_flag \
                        --vision_layers $VISION_LAYERS \
                        --dropout_visual $DROPOUT_VISUAL \
                        --dropout_language $DROPOUT_LANGUAGE \
                        --language_layers $LANGUAGE_LAYERS \
                        --feature_process_language $FEATURE_PROCESS_LANGUAGE \
                        --visual_embedding_dim $VISUAL_EMBEDDING_DIM_1 $VISUAL_EMBEDDING_DIM_2 \
                        --language_embedding_dim $LANGUAGE_EMBEDDING_DIM_1 $LANGUAGE_EMBEDDING_DIM_2 \
                        --gpu $gpu \
                        --max_iter $MAX_ITER \
                        --snapshot $SNAPSHOT \
                        --stepsize $MAX_ITER \
                        --base_lr $BASE_LR \
                        --train_json $TRAIN_JSON \
                        --test_json $VAL_JSON \
                        --train_h5 $TRAIN_H5 \
                        --test_h5 $TEST_H5 \
                        --random_seed $RANDOM_SEED \
                        --loss_type $LOSS_TYPE \
                        --lw_inter $LW_INTER \
                        --batch_size $BATCH_SIZE \
                        --distance_function $DISTANCE_FUNCTION \
                        --pool_type max \
                        --strong_supervise \
                        --snapshot_folder $snapshot_folder \
                        --input_visual_data $INPUT_VISUAL_DATA \
                        --tag $flow 

    #test on test 
    python utils/build_net.py --feature_process_visual $FEATURE_PROCESS_VISUAL  \
                        --$loc_flag \
                        --vision_layers $VISION_LAYERS \
                        --dropout_visual $DROPOUT_VISUAL \
                        --dropout_language $DROPOUT_LANGUAGE \
                        --language_layers $LANGUAGE_LAYERS \
                        --feature_process_language $FEATURE_PROCESS_LANGUAGE \
                        --visual_embedding_dim $VISUAL_EMBEDDING_DIM_1 $VISUAL_EMBEDDING_DIM_2 \
                        --language_embedding_dim $LANGUAGE_EMBEDDING_DIM_1 $LANGUAGE_EMBEDDING_DIM_2 \
                        --gpu $gpu \
                        --max_iter $MAX_ITER \
                        --snapshot $SNAPSHOT \
                        --stepsize $MAX_ITER \
                        --base_lr $BASE_LR \
                        --train_json $TRAIN_JSON \
                        --test_json $TEST_JSON \
                        --train_h5 $TRAIN_H5 \
                        --test_h5 $TEST_H5 \
                        --random_seed $RANDOM_SEED \
                        --loss_type $LOSS_TYPE \
                        --lw_inter $LW_INTER \
                        --batch_size $BATCH_SIZE \
                        --distance_function $DISTANCE_FUNCTION \
                        --pool_type max \
                        --snapshot_folder $snapshot_folder \
                        --strong_supervise \
                        --input_visual_data $INPUT_VISUAL_DATA \
                        --snapshot_folder=$snapshot_folder \
                        --tag $flow 
    
fi

if [ "$loss" == tall ] ; then
    python utils/fusion.py --rgb_tag $rgb \
                           --flow_tag $flow \
                           --iter $MAX_ITER \
                           --dataset $dataset \
                           --set $test_set \
                           --tall 
else
    python utils/fusion.py --rgb_tag $rgb \
                           --flow_tag $flow \
                           --iter $MAX_ITER \
                           --set $test_set \
                           --dataset $dataset
fi

echo "Evaluated model: " $model
echo "On dataset: " $dataset
