#!bin/bash
python3 inference.py \
    --mode greedy \
    --model_type cider \
    --folder ../hw3_data/p2_data/images/val \
    --output_json output/lora32/infer_greedy_cider.json \
    --decoder_path ../hw3_data/p2_data/decoder_model.bin \
    --model_path output/lora32/ \

python3 p2_evaluate.py \
    --pred_file output/lora32/infer_greedy_cider.json \
    --images_root ../hw3_data/p2_data/images/val/ \
    --annotation_file ../hw3_data/p2_data/val.json \
