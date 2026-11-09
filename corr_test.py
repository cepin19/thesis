import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pearson correlation
import scipy.stats as stats

y=1000*[1]#random.randint(0,1)]
df=pd.DataFrame(columns=["x", "power", "y"])
powers=[x/100 for x in range(25,350,25)]
for i in range(1000):
    nums=[random.randint(0,20) for i in range(10)]
    for p in powers:
        df=pd.concat([df, pd.DataFrame({"x": (np.asarray(nums)**p)/len(nums), "power": p, "y": y[i]})])
# compute correlation for each power
corrs=[]
print(df)
for p in powers:
    print(df[df["power"]==p]["x"])
    print(df[df["power"]==p]["y"])
    corrs.append(stats.pearsonr(df[df["power"]==p]["x"], df[df["power"]==p]["y"])[0])
# plot correlation as a function of power
print(corrs)
plt.plot(powers, corrs)
plt.show()