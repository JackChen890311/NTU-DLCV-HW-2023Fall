#!/bin/bash

# TODO - run your inference Python3 code
python3 hw3-2/inference.py \
    --folder $1 \
    --output_json $2 \
    --decoder_path $3 \
    --mode beam \
    --model_type cider \
    --model_path model3-2/ \