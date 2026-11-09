# Load two files and compute the correlation between the two sets of perplexities
from scipy.stats import pearsonr
import numpy as np

import pandas as pd
import sys
file1=sys.argv[1]
file2=sys.argv[2]
df1=pd.read_csv(file1,names=["perplexity"])
df2=pd.read_csv(file2,names=["perplexity"])
#read file containing comet scores
scores=[]
if len(sys.argv)==5:
    score_file=sys.argv[3]
    print(score_file)

    with open(score_file) as f:
        for l in f:
            scores.append(float(l))
    threshold=float(sys.argv[4])
    print(threshold)
print(len(df1))
print(len(df2))
print(len(scores))
if scores!=[]:
    assert len(scores)==len(df1)==len(df2)
    # only compute correlation for sentences with a score above the threshold
    df1["score"]=scores
    df2["score"]=scores
    df1=df1[df1["score"]>threshold]
    df2=df2[df2["score"]>threshold]
print(len(df1))
print(pearsonr(df1["perplexity"], df2["perplexity"])[0])
# print mean values to stderr
print(np.mean(df1["perplexity"]), np.mean(df2["perplexity"]), file=sys.stderr)