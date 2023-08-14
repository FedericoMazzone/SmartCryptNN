import requests

NUM_SAMPLE_ORIGINAL = 699
NUM_SAMPLE = 683
NUM_FEATURES = 9
NUM_CLASSES = 2

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
r = requests.get(url, allow_redirects=True)

lines = r.content.decode().splitlines()

assert len(lines) == NUM_SAMPLE_ORIGINAL

#       Attribute                     Domain
#    -- -----------------------------------------
#    1. Sample code number            id number
#    2. Clump Thickness               1 - 10
#    3. Uniformity of Cell Size       1 - 10
#    4. Uniformity of Cell Shape      1 - 10
#    5. Marginal Adhesion             1 - 10
#    6. Single Epithelial Cell Size   1 - 10
#    7. Bare Nuclei                   1 - 10
#    8. Bland Chromatin               1 - 10
#    9. Normal Nucleoli               1 - 10
#   10. Mitoses                       1 - 10
#   11. Class:                        (2 for benign, 4 for malignant)

new_lines = []

for l in lines:
    try:
        tokens = [int(x) for x in l.split(',')]
    except:
        continue
    assert len(tokens) == NUM_FEATURES + 2
    y = (tokens[-1] - 2) // 2
    assert y >= 0 and y < NUM_CLASSES
    new_line = [y] + [x - 1 for x in tokens[1:-1]]
    new_line = ",".join(str(x) for x in new_line) + '\n'
    new_lines.append(new_line)

assert len(new_lines) == NUM_SAMPLE

with open(f'data/bcw', 'w') as f:
    f.writelines(new_lines)
