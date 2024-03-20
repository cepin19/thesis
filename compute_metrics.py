import numpy as np
import sys
# load the  and compute the metric for each line
import pandas as pd


def kl_divergence(surp,*args):
    n=len(surp)
    surp_normalized = surp / np.sum(surp)
    uniform_distribution = np.full(n, 1 / n) * n
    return np.sum(surp_normalized * np.log2(surp_normalized / uniform_distribution))

def gini_coefficient(x,*args):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def cv(surp,*args):
    return np.std(surp) / np.mean(surp)

def sl(surp,k):
    #return np.sum(np.asarray(surp)**k)/len(surp)
    if SKIP_FIRST:
        surp = surp.copy()
        surp[0] = 0
    return np.nanmean(np.asarray(surp)**k)
def slor(surp,sentence, k=1):
    print(sentence)
    return sl(surp,k)-get_unigram_log_prob(sentence,unigram_mapping)/len(sentence.split(" "))
def normlp(surp,sentence, k):
    return sl(surp,k)*len(sentence.split(" "))/get_unigram_log_prob(sentence,unigram_mapping)
fnames = ["const", "L"]+["SL_"+str(k) for k in range(25,350,25)] + ["slor_"+str(k) for k in range(25,350,25)]+["normlp_"+str(k) for k in range(25,350,25)] +["gini", "kl", "cv"]
flist = [lambda surp,_: 1, lambda surp,_: sl(surp, 1)] + [lambda surp,_,k=k: sl(surp, k/100.0) for k in range(25,350,25)]+\
         [lambda surp,sentence,k=k: slor(surp,sentence,k/100.0) for k in range(25,350,25)]+\
        [lambda surp,sentence,k=k: normlp(surp,sentence,k/100) for k in range(25,350,25)]+\
         [gini_coefficient, kl_divergence, cv]



def compute_metrics(surps, fn, sentences=None, unigram_log_probs=None):
    if sentences is None:
        sentences = [None]*len(surps)
    if unigram_log_probs is None:
        unigram_log_probs = [None]*len(surps)
    assert len(surps) == len(unigram_log_probs) == len(sentences)
    out = []
    lens = []
    f=flist[fnames.index(fn)]
    print(surps, file=sys.stderr)
    for sent_surp,unigram_log_prob, sent in zip(surps, unigram_log_probs, sentences):
        if "unigram" in fn.lower():
            fval = f(np.asarray(unigram_log_prob),sent)
        else:
            fval = f(np.asarray(sent_surp),sent)
        out.append(fval)
        lens.append(float(len(sent_surp)))
        #print(fn)
        #print(len(out))
        #print(len(df))
       # df.loc[:, fn] = out
       # df.loc[:, fn+'_times_len'] = np.asarray(out)*np.asarray(lens)
        #df=add_lau_accept_measures(df)
    return out



for line in sys.stdin:
    surps= np.asarray([float(x) for x in line.strip().split(",")])
    print(surps)