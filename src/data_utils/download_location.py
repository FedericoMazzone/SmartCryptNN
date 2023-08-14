import os
import tarfile

import requests

ORIGINAL_NAME = 'dataset_location'
NEW_NAME = 'location'
NUM_SAMPLE = 5010
NUM_FEATURES = 446
NUM_CLASSES = 30

url = f'https://github.com/privacytrustlab/datasets/raw/master/{ORIGINAL_NAME}.tgz'
r = requests.get(url, allow_redirects=True)

with open(f'data/{NEW_NAME}.tgz', 'wb') as f:
    f.write(r.content)

with tarfile.open(f'data/{NEW_NAME}.tgz') as f:
    f.extractall(f'data/')

os.remove(f'data/{NEW_NAME}.tgz')

with open('data/bangkok', 'r') as f:
    lines = f.readlines()

os.remove(f'data/bangkok')

assert len(lines) == NUM_SAMPLE

for i in range(NUM_SAMPLE):
    l = lines[i].split(',')
    assert len(l) == NUM_FEATURES + 1
    l[0] = int(l[0][1:-1]) - 1
    assert l[0] >= 0 and l[0] < NUM_CLASSES
    l[0] = str(l[0])
    lines[i] = ','.join(l)

with open(f'data/{NEW_NAME}', 'w') as f:
    f.writelines(lines)
