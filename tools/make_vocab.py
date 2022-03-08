"""
args:
1 corpus file (tokenized)
2 K
prints K most frequent vocab items
"""
import sys
from collections import Counter

print('<unk>')
print('<pad>')
print('<s>')
print('</s>')

file_path = sys.argv[1]
most_common_num = int(sys.argv[2])
static_feature_num = int(sys.argv[3])
c = Counter()
for l in open(file_path):
    for tok in l.strip().split()[:-static_feature_num]:
        c[tok] += 1

for tok, _ in c.most_common(most_common_num):
    print(tok)



