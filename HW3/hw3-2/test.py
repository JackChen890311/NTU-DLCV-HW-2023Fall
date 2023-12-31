# import language_evaluation
# from pprint import PrettyPrinter
# pprint = PrettyPrinter().pprint

# predicts = ['i am a boy', 'she is a girl']
# answers = ['am i a boy ?', 'is she a girl ?']

# evaluator = language_evaluation.CocoEvaluator()
# results = evaluator.run_evaluation(predicts, answers)
# pprint(results)


# {'Bleu_1': 0.9999999997500004,
#  'Bleu_2': 0.5773502690332603,
#  'Bleu_3': 4.3679023223468616e-06,
#  'Bleu_4': 1.4287202142987477e-08,
#  'CIDEr': 3.333333333333333,
#  'METEOR': 0.43354749322305886,
#  'ROUGE_L': 0.75,
#  'SPICE': 0.6666666666666666}

# evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=5)
# results = evaluator.run_evaluation(predicts, answers)
# pprint(results)
# {'rouge1': 1.0,
#  'rouge2': 0.3333333333333333,
#  'rougeL': 0.75}

# evaluator = language_evaluation.Rouge155Evaluator(num_parallel_calls=5)
# results = evaluator.run_evaluation(predicts, answers)
# pprint(results)
# {'rouge1': 1.0,
#  'rouge2': 0.3333333333333333,
#  'rougeL': 0.75}

# ==========================================================================
# from tokenizer import BPETokenizer

# encoder_file = 'encoder.json'
# vocab_file = 'vocab.bpe'
# encoder = BPETokenizer(encoder_file, vocab_file)
# prompt = 'a kitchen with a sink and many cooking machines and a pot of food'
# context = encoder.encode(prompt)
# context = [50256] + context + [50256, -100]
# print(context)
# print(encoder.decode(context))


# ==========================================================================
# import timm
# import torch
# import torch.nn as nn

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print(timm.list_models('*vit_*'))
# model = timm.create_model('vit_base_patch16_clip_224', pretrained=True).to(device)


# t = torch.randn(1, 3, 224, 224).to(device)
# model.norm = torch.nn.Identity()
# model.head_drop = torch.nn.Identity()
# model.head = torch.nn.Identity()
# # print(model)
# # print(model(t).shape)
# # y = t
# # for layer in model.children():
# #     print(layer)
# #     y = layer(y)
# #     print(y.shape)

# newmodel = nn.Sequential(*list(model.children())[:-4])
# print(newmodel)
# y = newmodel(t)
# print(y.shape)

import os
import torch

d = torch.load(os.path.join('output/2023-11-22~13:57:37/','lora.pt'))
print(d.keys())