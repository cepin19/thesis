import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.show()
final_df = pd.DataFrame(columns=["score", "value"])
measure=sys.argv[1]
score=sys.argv[2]
filename=sys.argv[3]
#print(sys.argv[4:])
print(measure)
print(score)
print(sys.argv[4:])
for f in sys.argv[4:]:
    try:
        df = pd.read_csv(f)
    except:
        continue
    print(f)
    for i,row in df.iterrows():
        print(row)
        rm=row["measure"]
        if "times_len" in row:
            if row["times_len"]==True:
                rm+="_times_len"
        if measure!=rm:
            continue
        print(row)
        power=row['power']
        result = row[score]
        baseline=row["baseline"]

        if score=="pearson":
            result=abs(result)
        if "lme" in score:
            final_df = pd.concat([final_df, pd.DataFrame([{"score": power, "value": result, "baseline": row["lme_baseline"],
                                                           "baseline2": row["lme_baseline2"], "baseline3": row["lme_baseline3"],
                                                           "baseline4": row["lme_baseline4"]}])])
             #final_df = pd.concat([final_df, pd.DataFrame([{"score": power, "value": result,
#                                                           "baseline2": row["lme_baseline2"]}])])
        else:
            final_df=pd.concat([final_df, pd.DataFrame([{"score": power, "value": result, "baseline": baseline}])])
final_df=final_df.sort_values(by="score", ascending=True)
print(final_df)
fig, ax = plt.subplots()

if len(final_df["score"])==1:
    ax.plot(final_df["score"],final_df["value"],'bo')
    if score!="pearson":
        ax.plot(final_df["score"],final_df["baseline"],'yo')
else:
    label_score=score.replace("_test",'')
#    if score.startswith("lme"):label_score="GLM_SLOR"
    ax.plot(final_df["score"],final_df["value"],label=label_score)
    if score!="pearson" and "lme" not in score:
        ax.plot(final_df["score"],final_df["baseline"],label="baseline")
if "lme" in score:
    ax.plot(final_df["score"],final_df["baseline2"],label="baseline")
#    ax.plot(final_df["score"],final_df["baseline3"],label="baseline3")
 #   ax.plot(final_df["score"],final_df["baseline4"],label="baseline4")
for x_value in final_df["score"]:
    plt.axvline(x=x_value, color='lightgray', linestyle='--', linewidth=1)
plt.legend(loc="lower right")
#set y range based on score
if score=="pearson":
    plt.ylim(0.1,0.35)
elif "lme" in score:
    pass
else:
    plt.ylim(0.69,0.735)

plt.savefig(filename.replace(':',"_"), pad_inches=0.0, bbox_inches='tight')
#plt.show()
