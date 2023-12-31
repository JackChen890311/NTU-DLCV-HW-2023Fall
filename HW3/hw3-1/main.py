import os
import clip
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt


class DataHandler:
    def __init__(self, data_path, json_path):
        self.data_path = data_path
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.classes = json.load(f)
        self.classes = list(self.classes.values())
        self.filename = sorted(os.listdir(data_path))
        self.filecnt = len(self.filename)

    def getimage(self, idx):
        image = Image.open(os.path.join(self.data_path, self.filename[idx]))
        try:
            label = int(self.filename[idx].split('_')[0])
        except:
            label = -1
        return image, label


def plot(dh, data, top, filename):
    (image, class_id) = data
    (top_probs, top_labels) = top
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs)
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [dh.classes[index] for index in top_labels])
    if class_id in top_labels:
        true_index = top_labels.tolist().index(class_id)
        plt.barh(y[true_index], top_probs[true_index], color='g')
    plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../hw3_data/p1_data/val')
    parser.add_argument('--json_path', type=str, default='../hw3_data/p1_data/id2label.json')
    parser.add_argument('--csv_path', type=str, default='result.csv')
    parser.add_argument('--showCnt', type=int, default=0)
    args = parser.parse_args()

    showCnt = args.showCnt
    if showCnt and not os.path.exists('result'):
        os.mkdir('result')
    data_path = args.data_path
    json_path = args.json_path
    csv_path = args.csv_path

    # Load the dataset
    dh = DataHandler(data_path, json_path)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14', device)

    trueCnt = 0
    vis = [random.randint(0, dh.filecnt) for _ in range(showCnt)]
    result = []
    for idx in tqdm(range(dh.filecnt)):
        # Prepare the inputs
        image, class_id = dh.getimage(idx)
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dh.classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
        values, indices = values.cpu().numpy(), indices.cpu().numpy()

        # # Print the result
        # print("\nTop predictions:\n")
        # for value, index in zip(values, indices):
        #     print(f"{dh.classes[index]:>16s}: {100 * value.item():.2f}%")
        # print('True label:', dh.classes[class_id])

        result.append(indices[0])
        if indices[0] == class_id:
            trueCnt += 1

        if showCnt and idx in vis:
            plot(dh, (image, class_id), (values, indices), f'result/{idx}.png')

    df = pd.DataFrame({'filename': dh.filename, 'label': result})
    df.to_csv(csv_path, index=False)
    print('Accuracy:', trueCnt / dh.filecnt)