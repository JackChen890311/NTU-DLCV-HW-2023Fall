import os
import json
from PIL import Image

from p2_evaluate import CLIPScore


if __name__ == '__main__':
    with open('output/lora32/infer_greedy_cider.json', 'r') as f:
        result = json.load(f)
    clip_score = CLIPScore()
    
    min_score = 1e9
    max_score = -1e9
    min_name = ''
    max_name = ''
    img_root = '../hw3_data/p2_data/images/val'

    for k, v in result.items():
        img = Image.open(os.path.join(img_root, k + '.jpg')).convert('RGB')
        score = clip_score.getCLIPScore(img, v)
        if score < min_score:
            min_score = score
            min_name = k
        if score > max_score:
            max_score = score
            max_name = k

    print(min_score, min_name)
    print(max_score, max_name)