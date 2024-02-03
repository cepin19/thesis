import sys
import pandas as pd
import matplotlib.pyplot as plt


plt.show()
final_df = pd.DataFrame(columns=["measure", "value"])
metric=sys.argv[1]
for f in sys.argv[2:]:
    df = pd.read_csv(f)
    power=int(df["measure"][0].replace("SL_", "").replace("_times_len", "").replace("slor_", "").replace("normlp_", ""))
    print(power)
    result=df[metric]
    baseline=df["baseline_test"]
    if metric=="pearson":
        result=result.abs()
    final_df=pd.concat([final_df, pd.DataFrame({"measure": power, "value": result, "baseline": baseline})])
final_df=final_df.sort_values(by="measure", ascending=True)
print(final_df)
fig, ax = plt.subplots()
ax.plot(final_df["measure"],final_df["value"])
ax.plot(final_df["measure"],final_df["baseline"])
plt.show()