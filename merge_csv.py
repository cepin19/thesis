import pandas as pd
import glob

# Pattern to match the files. Adjust this pattern to match your specific files.
# This pattern matches files that start with 'cola', include 'normlp', and end with '_results.csv'
pattern = 'cola4/cola2_*_drop=True_results.csv'

# Use glob to find files that match the pattern
fl = glob.glob(pattern)
file_list=[]
for f in fl:
    if not "times_len" in f:
        file_list.append(f)
# Initialize an empty list to store data frames
dfs = []

# Loop through the list of files
for file in file_list:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)
merged_df=merged_df[merged_df["power"].isin([0.75,1,1.25,1.5,1.75,2,2.25,2.5]) | (~merged_df["measure"].isin(["slor","normlp","SL"]))]
merged_df["delta_glm"]=merged_df["lme_fn:len_len_slor_normlp"]-merged_df["lme_baseline2"]
print(merged_df.columns)
# Now you have a single DataFrame `merged_df` that you can use for further analysis or save to a new file
# For example, to save to a new CSV file:
#merged_df.to_csv('merged_results.csv', index=False)
cols=[ 'measure', 'power', 'pearson', 'lr_test', 'svm_test', 'mlp_scaled_test', 'delta_glm']

print(merged_df[cols])
merged_df[cols].sort_values(["measure","power"]).to_csv("cola2_paper_longer.csv",index=False)

