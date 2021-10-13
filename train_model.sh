

SOURCE=cxr
TASK=original

DATA_DIR=/home/ksaab/Documents/observational/data
if [ $SOURCE == "mets" ]; then
    LR=0.0001 
    L2=0.00001
    Vscale=0.2
    gaze_mtl_task=diffusivity
    tw="0.5"

elif [ $SOURCE == "cxr" ]; then
    LR=0.0001 
    L2=0.0001 
    Vscale=0.2
    gaze_mtl_task=loc
    tw="0.5"

elif [ $SOURCE == "cxr2" ]; then
    LR=0.0001 
    L2=0.01 
    Vscale=0.2
    gaze_mtl_task=diffusivity 
    tw="1.0" 
fi


if [ $TASK == "gaze_mtl" ]; then
    LOG_PTH=/home/ksaab/Documents/observational/results/$TASK/$SOURCE/$gaze_mtl_task
else
    LOG_PTH=/home/ksaab/Documents/observational/results/$TASK/$SOURCE
fi

for seed in 0 #101 102 103 104 105 106 107 108 109 110
do
    python train_emmental.py \
        --source $SOURCE  \
        --train_scale 1 \
        --val_scale $Vscale \
        --data_dir $DATA_DIR \
        --task $TASK \
        --pretrained \
        --task_weights "$tw" \
        --gaze_mtl_task $gaze_mtl_task \
        --log_path $LOG_PTH/seed_$seed \
        --use_exact_log_path True \
        --seed $seed \
        --n_epochs 15 \
        --train_split train \
        --valid_split val \
        --optimizer adam \
        --lr $LR \
        --l2 $L2 \
        --checkpointing 1 \
        --checkpoint_metric target/$SOURCE/val/accuracy:max \
        --batch_size 32 \
        --lr_scheduler cosine_annealing \
        --lr_scheduler_step_unit epoch \
        
done