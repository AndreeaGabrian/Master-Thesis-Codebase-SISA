import os
import shutil
import pandas as pd

RAW_DIR = "../raw/ham10000"
DATA_DIR = "../data/HAM10000"

meta = pd.read_csv("raw/HAM10000_metadata.csv")
classes = sorted(meta['dx'].unique())

for cls in classes:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

for _, row in meta.iterrows():
    img_id = row['image_id']
    label = row['dx']
    src = os.path.join(RAW_DIR, img_id + ".jpg")
    dst = os.path.join(DATA_DIR, label, img_id + ".jpg")
    shutil.copy(src, dst)

print("Images reorganized under", DATA_DIR)
