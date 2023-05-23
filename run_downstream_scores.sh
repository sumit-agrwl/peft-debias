#!/bin/bash

PEFT=$1
DATASET=$2
DEBIAS=$3
AXIS=$4
DEBIAS_CONFIGURATION=$5

if [[ $DEBIAS_CONFIGURATION == "" ]]; then
    DEBIAS_CONFIGURATION="before"
fi

MODEL_PATH=models/$DATASET/$PEFT/$DEBIAS-$DEBIAS_CONFIGURATION/

if [[ $PEFT == "sft" ]]; then
    PEFT_PATH=models/$DATASET/$PEFT/$DEBIAS-$DEBIAS_CONFIGURATION/
else
    if [[ $DEBIAS == "adv" ]]; then
        PEFT_PATH=models/$PEFT/$DEBIAS/$AXIS/$DATASET/class/
    elif [[ $DEBIAS == "cda" ]]; then
        PEFT_PATH=models/$PEFT/$DEBIAS/$AXIS/$DATASET/mlm/
    fi
fi

if [[ $DATASET == "bias-bios" ]]; then
    METRIC="eval_accuracy"
    IN_DOMAIN_DATA_PATH="data/bias-bios/test.jsonl"
    IN_DOMAIN_TEXT_COLUMN=text
    IN_DOMAIN_LABEL_COLUMN=p

    OOD_DATA_PATH=""
    OOD_TEXT_COLUMN=""
    OOD_LABEL_COLUMN=""

elif [[ $DATASET == "gab" ]]; then
    METRIC="eval_f1"
    IN_DOMAIN_DATA_PATH="data/gab/test.jsonl"
    IN_DOMAIN_TEXT_COLUMN=text
    IN_DOMAIN_LABEL_COLUMN=label

    OOD_DATA_PATH="data/iptts77k/test.csv"
    OOD_TEXT_COLUMN="Text"
    OOD_LABEL_COLUMN="Label"
    
elif [[ $DATASET == "fdcl" ]]; then
    METRIC="eval_f1"
    IN_DOMAIN_DATA_PATH="data/fdcl/test.jsonl"
    IN_DOMAIN_TEXT_COLUMN=text
    IN_DOMAIN_LABEL_COLUMN=ND_label

    OOD_DATA_PATH="data/iptts77k/test.csv"
    OOD_TEXT_COLUMN="Text"
    OOD_LABEL_COLUMN="Label"

elif [[ $DATASET == "ws" ]]; then
    METRIC="eval_f1"
    IN_DOMAIN_DATA_PATH="data/ws/test.jsonl"
    IN_DOMAIN_TEXT_COLUMN=text
    IN_DOMAIN_LABEL_COLUMN=label

    OOD_DATA_PATH="data/iptts77k/test.csv"
    OOD_TEXT_COLUMN="Text"
    OOD_LABEL_COLUMN="Label"

fi

bash run_downstream_inference.sh $PEFT $MODEL_PATH $IN_DOMAIN_DATA_PATH $IN_DOMAIN_LABEL_COLUMN $IN_DOMAIN_TEXT_COLUMN $PEFT_PATH $DEBIAS_CONFIGURATION "in_domain" $METRIC

# if [[ $OOD_DATA_PATH != "" ]]; then
#     bash run_inference.sh $PEFT $MODEL_PATH $OOD_DATA_PATH $OOD_LABEL_COLUMN $OOD_TEXT_COLUMN $PEFT_PATH $DEBIAS_CONFIGURATION "out_of_domain" $METRIC
# fi

echo $MODEL_PATH
python extrinsic_metric.py --dataset $DATASET --output_dir $MODEL_PATH --axis $AXIS