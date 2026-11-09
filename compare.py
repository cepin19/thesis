import numpy as np
import pandas as pd
import sys
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



srcdf = pd.read_csv(sys.argv[1], header=None, names=["sentence","kl","cv","gini","l","sl1","sl2"], skip_blank_lines=False, sep="\t")
tgtdf = pd.read_csv(sys.argv[2], header=None, names=["sentence","kl","cv","gini","l","sl1","sl2"], skip_blank_lines=False, sep="\t")
transdf=pd.read_csv(sys.argv[3], header=None, names=["sentence","kl","cv","gini","l","sl1","sl2"], skip_blank_lines=False, sep="\t")
print(len(srcdf))
print(len(tgtdf))
print(len(transdf))

srcdf=pd.DataFrame(columns=["sentence","kl","cv","gini","l","sl1","sl2","surp"])
tgtdf=pd.DataFrame(columns=["sentence","kl","cv","gini","l","sl1","sl2","surp"])
transdf=pd.DataFrame(columns=["sentence","kl","cv","gini","l","sl1","sl2","surp"])
with open(sys.argv[1],"r") as f:
    for i,line in enumerate(f):
   #     print(line.strip().split('\t'))
        linef=[float(f) for f in line.strip().split('\t')[1:-1]]
        linel = [eval(line.strip().split('\t')[-1])]

        line=[line.split('\t')[0]]
        line.extend(linef)
        line.extend(linel)

        srcdf.loc[len(srcdf.index)]=line
 #       print(line)
#        print(srcdf)
  #      print(srcdf['kl'].mean())

    print(i)
with open(sys.argv[2],"r") as f:
    for i,line in enumerate(f):
        linef=[float(f) for f in line.strip().split('\t')[1:-1]]
        linel=[eval(line.strip().split('\t')[-1])]
        line=[line.split('\t')[0]]
        line.extend(linef)
        line.extend(linel)

        tgtdf.loc[len(tgtdf.index)]=line
    print(i)
with open(sys.argv[3], "r") as f:
    for i, line in enumerate(f):
        linef = [float(f) for f in line.strip().split('\t')[1:-1]]
        linel=[eval(line.strip().split('\t')[-1])]

        line = [line.split('\t')[0]]
        line.extend(linef)
        line.extend(linel)

        transdf.loc[len(transdf.index)] = line
    print(i)
concat=srcdf.copy()

ranges={"kl":[1.4,3],"cv":[4,1.6],"gini":[0.8,0.8],"l":[1200,2000],"sl1":[1400,2000],"sl2":[1400,2000]}
print(len(srcdf))
import seaborn as sns
for c in ["sentence","kl","cv","gini","l","sl1","sl2","surp"]:
    concat[c+"_src"]=srcdf[c]
    concat[c+"_tgt"]=tgtdf[c]
    concat[c+"_trans"]=transdf[c]
    print(c)
    if c=="surp" or c=="sentence":
        continue
    print("SRC",srcdf[c].mean())
    print("TGT",tgtdf[c].mean())
    print("TRANS",transdf[c].mean())
    model = LinearRegression()

    print("SRC-TGT rand",pearsonr(srcdf[c],np.random.permutation(tgtdf[c].to_numpy())))


    print("SRC-TGT",pearsonr(srcdf[c],tgtdf[c]))
    sns_plot = sns.regplot(x=c+"_src", y=c+"_tgt", data=concat, fit_reg=True,scatter_kws={"color": "blue"}, line_kws={"color": "red"})

    sns_plot.set(ylim=(0, ranges[c][0]))
    sns_plot.set(ylim=(0, ranges[c][1]))

    fig = sns_plot.get_figure()
    fig.savefig("{}_src_tgt_corr.png".format(c),bbox_inches='tight',dpi=600)
    fig.clf()
    model.fit(srcdf[c][:9000].to_numpy().reshape(-1, 1),tgtdf[c][:9000].to_numpy())
    y_pred = model.predict(srcdf[c][9000:].to_numpy().reshape(-1, 1))
    mse = mean_squared_error(tgtdf[c][9000:].to_numpy(), y_pred)
    print( mse)

    print("TGT-TRANS",pearsonr(tgtdf[c],transdf[c]))
    sns_plot = sns.regplot(x=c+"_tgt", y=c+"_trans", data=concat, fit_reg=True,scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    sns_plot.set(ylim=(0, ranges[c][0]))
    sns_plot.set(ylim=(0, ranges[c][1]))
    fig = sns_plot.get_figure()
    fig.savefig("{}_tgt_trans_corr.png".format(c),bbox_inches='tight',dpi=600)
    fig.clf()

    model.fit(tgtdf[c][:9000].to_numpy().reshape(-1, 1),transdf[c][:9000].to_numpy())
    y_pred = model.predict(tgtdf[c][9000:].to_numpy().reshape(-1, 1))
    mse = mean_squared_error(transdf[c][9000:].to_numpy(), y_pred)
    print(mse)

    print("SRC-TRANS",pearsonr(srcdf[c],transdf[c]))

    sns_plot = sns.regplot(x=c+"_src", y=c+"_trans", data=concat, fit_reg=True,scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    sns_plot.set(ylim=(0, ranges[c][0]))
    sns_plot.set(ylim=(0, ranges[c][1]))

    fig = sns_plot.get_figure()
    fig.savefig("{}_src_trans_corr.png".format(c),bbox_inches='tight',dpi=600)
    fig.clf()

    model.fit(srcdf[c][:9000].to_numpy().reshape(-1, 1),transdf[c][:9000].to_numpy())
    y_pred = model.predict(srcdf[c][9000:].to_numpy().reshape(-1, 1))
    mse = mean_squared_error(transdf[c][9000:].to_numpy(), y_pred)
    print( mse)


print(concat.head())
print("sorted by TGT")
print(concat.sort_values(by=["sl2_tgt"],ascending=False).head(100)['sl2_src'].mean())
print(concat.sort_values(by=["sl2_tgt"],ascending=False).head(100)['sl2_tgt'].mean())
print(concat.sort_values(by=["sl2_tgt"],ascending=False).head(100)['sl2_trans'].mean())

print(concat.sort_values(by=["sl2_tgt"],ascending=True).head(100)['sl2_src'].mean())
print(concat.sort_values(by=["sl2_tgt"],ascending=True).head(100)['sl2_tgt'].mean())
print(concat.sort_values(by=["sl2_tgt"],ascending=True).head(100)['sl2_trans'].mean())
print("sorted by SRC")
print(concat.sort_values(by=["sl2_src"],ascending=False).head(100)['sl2_src'].mean())
print(concat.sort_values(by=["sl2_src"],ascending=False).head(100)['sl2_tgt'].mean())
print(concat.sort_values(by=["sl2_src"],ascending=False).head(100)['sl2_trans'].mean())

print(concat.sort_values(by=["sl2_src"],ascending=True).head(100)['sl2_src'].mean())
print(concat.sort_values(by=["sl2_src"],ascending=True).head(100)['sl2_tgt'].mean())
print(concat.sort_values(by=["sl2_src"],ascending=True).head(100)['sl2_trans'].mean())


print()
print("sorted by TGT")
print(concat.sort_values(by=["cv_tgt"],ascending=False).head(100)['cv_src'].mean())
print(concat.sort_values(by=["cv_tgt"],ascending=False).head(100)['cv_tgt'].mean())
print(concat.sort_values(by=["cv_tgt"],ascending=False).head(100)['cv_trans'].mean())

print(concat.sort_values(by=["cv_tgt"],ascending=True).head(100)['cv_src'].mean())
print(concat.sort_values(by=["cv_tgt"],ascending=True).head(100)['cv_tgt'].mean())
print(concat.sort_values(by=["cv_tgt"],ascending=True).head(100)['cv_trans'].mean())

print("sorted by SRC")
print(concat.sort_values(by=["cv_src"],ascending=False).head(100)['cv_src'].mean())
print(concat.sort_values(by=["cv_src"],ascending=False).head(100)['cv_tgt'].mean())
print(concat.sort_values(by=["cv_src"],ascending=False).head(100)['cv_trans'].mean())

print(concat.sort_values(by=["cv_src"],ascending=True).head(100)['cv_src'].mean())
print(concat.sort_values(by=["cv_src"],ascending=True).head(100)['cv_tgt'].mean())
print(concat.sort_values(by=["cv_src"],ascending=True).head(100)['cv_trans'].mean())
maxd=0
max_surp_tgt=[]
max_surp_trans=[]
diffs=[]
tgt_sent=""
trans_sent=""
for i in range(len(concat)):
    diff=concat.iloc[i]['kl_tgt']-concat.iloc[i]['kl_trans']
    diffs.append(diff)
    if diff>maxd:
        print(diff)
        if len(concat.iloc[i]['surp_trans'])>30 or len(concat.iloc[i]['surp_trans'])<5 or abs(len(concat.iloc[i]['surp_trans'])-len(concat.iloc[i]['surp_tgt']))>9:
            continue
        maxd=diff
        max_surp_tgt=concat.iloc[i]['surp_tgt']
        max_surp_trans=concat.iloc[i]['surp_trans']
        tgt_sent=concat.iloc[i]['sentence_tgt']
        trans_sent=concat.iloc[i]['sentence_trans']


max_surp_frame=pd.DataFrame()
max_surp_frame['tgt']=[max_surp_tgt]
max_surp_frame['trans']=[max_surp_trans]

idx=np.argpartition(np.asarray(diffs), -20)[-20:]
for i in idx:
    if abs(len((concat.iloc[i]['sentence_tgt']))/len(concat.iloc[i]['sentence_trans']))>1.2:
        continue
    print(concat.iloc[i]['kl_tgt'])
    print(concat.iloc[i]['kl_trans'])
    print(concat.iloc[i]['sentence_tgt'])
    print(concat.iloc[i]['sentence_trans'])
import matplotlib.pyplot as plt
x=[i for i in range(1,max(len(max_surp_tgt),len(max_surp_trans))+1)]
plt.plot([i for i in range(1,len(max_surp_tgt)+1)], max_surp_tgt, label='HT')
plt.plot([i for i in range(1,len(max_surp_trans)+1)], max_surp_trans, label='MT')

# Setting the x and y axis labels
plt.xlabel('Word position')
plt.ylabel('Surprisal')
plt.legend()
#sns.lineplot(data=max_surp_frame)
#fig = sns_plot.get_figure()
plt.savefig("max_surp.pdf",bbox_inches='tight',dpi=600)
print(tgt_sent)
print(trans_sent)
