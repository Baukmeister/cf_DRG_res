"""
args:
1 corpus file (tokenized)
2 K
prints K most frequent vocab items
"""
import sys
from collections import Counter

import pandas as pd

print('<unk>')
print('<pad>')
print('<s>')
print('</s>')

file_path = sys.argv[1]
most_common_num = int(sys.argv[2])
c = Counter()
file = pd.read_csv(file_path)
events = file['events']

for event_sequence in events:
    for tok in event_sequence.split():
        c[tok] += 1

for tok, _ in c.most_common(most_common_num):
    print(tok)



