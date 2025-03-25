import shutil
import os
import tarfile

import requests

ORIGINAL_NAME = 'dataset_texas'
NEW_NAME = 'texas100'
NUM_SAMPLE = 67330
NUM_FEATURES = 6169
NUM_CLASSES = 100

url = f'https://github.com/privacytrustlab/datasets/raw/master/{ORIGINAL_NAME}.tgz'
r = requests.get(url, allow_redirects=True)

with open(f'data/{NEW_NAME}.tgz', 'wb') as f:
    f.write(r.content)

with tarfile.open(f'data/{NEW_NAME}.tgz') as f:
    f.extractall(f'data/')

os.remove(f'data/{NEW_NAME}.tgz')

with open('data/texas/100/feats', 'r') as f:
    features = f.readlines()

with open('data/texas/100/labels', 'r') as f:
    labels = f.readlines()

shutil.rmtree('data/texas')

assert len(features) == NUM_SAMPLE
assert len(labels) == NUM_SAMPLE

lines = []

for i in range(NUM_SAMPLE):
    assert len(features[i]) == 2 * NUM_FEATURES
    y = int(labels[i]) - 1
    assert y >= 0 and y < NUM_CLASSES
    lines.append(f"{y},{features[i]}")

with open(f'data/{NEW_NAME}', 'w') as f:
    f.writelines(lines)
