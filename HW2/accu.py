import pandas as pd

gt = pd.read_csv('hw2_data/digits/usps/val.csv')
pred = pd.read_csv('usps.csv')

merged = gt.merge(pred, left_on='image_name', right_on='image_name', how='inner',
          suffixes=('_left', '_right'))

print(len(gt), len(pred), len(merged))
print(merged.head())
cnt = sum(merged.label_left == merged.label_right)

# for i in range(len(gt)):
#     img_name = gt.loc[i].image_name
#     truth = gt.loc[i].label
#     predict = pred.loc[pred['image_name'] == img_name].label.item()
#     if truth == predict:
#         cnt += 1

print(cnt / len(merged))