# The name of this experiment.
name=$2

# Save logs and models under snap/gqa; make backup.
output=snap/okvqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/okvqa.py \
    --train train --valid testdev \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --optim bert --epochs 20 \
    --batchSize 8 --lr 5e-5 \
    --output $output ${@:3} \
    --tqdm
