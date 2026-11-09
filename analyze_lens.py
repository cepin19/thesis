import pandas as pd
import matplotlib.pyplot as plt
pd=pd.read_csv("cola_public/raw/in_domain_train.tsv", sep='\t',header=None, names=['sentence_source', 'label', 'label_author', 'sentence'])
lens=pd["sentence"].str.len()
print(lens)
lens.hist(density=True, bins=20)
plt.show()
