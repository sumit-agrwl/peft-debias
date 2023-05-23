PEFT=$1
DATASET=$2
PATCH_PATH=$3
LABEL_COLUMN=$4
DEBIAS_METHOD=$5
SAVE_PREFIX=$6

if [[ $DATASET == "gab" ]]; then
  N_EPOCHS=10
  BS=32
  LR=2e-5
  METRIC="eval_f1"
  CLASS_WEIGHTS="{\"0\":1,\"1\":10}"
  EVAL_STEPS=1000
elif [[ $DATASET == "fdcl" ]]; then
  N_EPOCHS=10
  BS=32
  LR=2e-5
  METRIC="eval_f1"
  # CLASS_WEIGHTS="{\"0\":1,\"1\":10}"
  CLASS_WEIGHTS="fail"
  EVAL_STEPS=1000
elif [[ $DATASET == "ws" ]]; then
  N_EPOCHS=10
  BS=32
  LR=2e-5
  METRIC="eval_f1"
  CLASS_WEIGHTS="{\"0\":1,\"1\":6.7}"
  EVAL_STEPS=500
else
  N_EPOCHS=5
  BS=32
  LR=2e-5
  METRIC="eval_accuracy"
  EVAL_STEPS=5000
  CLASS_WEIGHTS="fail"
fi

if [[ $SAVE_PREFIX == "" ]]; then
  OUTPUT_DIR=models/$DATASET/$PEFT/$DEBIAS_METHOD
else
  OUTPUT_DIR=models/$DATASET/$PEFT/$SAVE_PREFIX-$DEBIAS_METHOD
fi

mkdir -p $OUTPUT_DIR

echo $OUTPUT_DIR

python run_extrinsic.py \
  --peft $PEFT \
  --debias_configuration before \
  --patch_path $PATCH_PATH \
  --model_name_or_path bert-base-uncased \
  --train_file data/$DATASET/train.jsonl \
  --validation_file data/$DATASET/validation.jsonl \
  --test_file data/$DATASET/test.jsonl \
  --label_column $LABEL_COLUMN \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size 128 \
  --overwrite_output_dir \
  --overwrite_cache \
  --full_ft_max_epochs_per_iteration $N_EPOCHS \
  --sparse_ft_max_epochs_per_iteration $N_EPOCHS \
  --num_train_epochs $N_EPOCHS \
  --eval_steps $EVAL_STEPS \
  --save_steps $EVAL_STEPS \
  --evaluation_strategy steps \
  --freeze_layer_norm \
  --learning_rate $LR \
  --metric_for_best_model $METRIC \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 \
  --log_level debug \
  --cls_weights $CLASS_WEIGHTS > $OUTPUT_DIR/training.log
  