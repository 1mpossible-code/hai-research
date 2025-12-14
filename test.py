# import json, collections
# path="data/lewidi/offensiveness.jsonl"
# n=0
# langs=collections.Counter()
# label_counts=collections.Counter()
# min_k=10**9
# max_k=0
# with open(path) as f:
#     for line in f:
#         obj=json.loads(line); n+=1
#         langs[obj.get("language","?")]+=1
#         hl=obj["human_labels"]
#         min_k=min(min_k,len(hl)); max_k=max(max_k,len(hl))
#         for x in hl: label_counts[x]+=1
# print("N:",n)
# print("Languages:",langs.most_common(10))
# print("Label dist:",label_counts)
# print("Annotators per item min/max:",min_k,max_k)


import json
from collections import Counter
c=Counter(); n=0
with open("runs/2025-12-14_155608_exp_semeval_baseline/judgments.jsonl") as f:
    for line in f:
        n+=1
        c[json.loads(line).get("status","?")]+=1
print("rows:",n)
print(c)