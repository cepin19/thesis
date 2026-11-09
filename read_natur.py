import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, GPT2TokenizerFast
import transformers
model_name="gpt2"
#model_name = "bigscience/bloom-1b1"





# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name,add_prefix_space=True)
#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",add_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype="auto",cache_dir="/home/enki/models/marian/perplex/models/",offload_folder="/home/enki/models/marian/perplex/models/")
model.eval()

# read CSV file into a pandas DataFrame
df = pd.read_csv('/home/enki/models/marian/perplex/naturalstories/naturalstories_RTS/processed_RTs.tsv', sep='\t')

# Remove duplicate rows based on 'word' and 'meanItemRT' columns, keeping the original order
unique_data = df.loc[~df[['word', 'meanItemRT']].duplicated(keep='first')]

# Print out the word and its mean reading time for each unique row
text=[]
rts=[]
sents=[""]
sent_rts=[[]]
#sent_rts_mean=[]
for index, row in unique_data.iterrows():
#    print(f"{row['word']}: {row['meanItemRT']}")
    text.append(row['word'])
    rts.append(row['meanItemRT'])
    sent_rts[-1].append(float(row['meanItemRT']))
    if row['word'].endswith('.'):
        sents[-1]+=row['word']
    else:
        sents[-1]+=row['word']+' '


    if row['word'].endswith('.'):
        #sent_rts_mean.append(sent_rts[-1]/(sents[-1].count(' ')+1))
        sents.append("")
        sent_rts.append([])
        
sents.pop()
sent_rts.pop()
chunk_size=512
stories=[' '.join(text)]

def get_surp(stories):
    softmax = torch.nn.Softmax(dim=-1)
    ctx_size = 512 # model.config.max_position_embeddings
    bos_id = model.config.bos_token_id

    batches = []
    words = []
    #print(stories)
    for story in stories:
        words.extend(story.split(" "))
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask
        start_idx = 0
        # sliding windows with 50% overlap
        # start_idx is for correctly indexing the "later 50%" of sliding windows
        while len(ids) > ctx_size:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]),
                                                            "attention_mask": torch.tensor([1] + attn[:ctx_size-1])}),
                                start_idx))

            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)-1

        batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids),
                                                       "attention_mask": torch.tensor([1] + attn)}),
                           start_idx))
    curr_word_ix = 0
    curr_word_surp = []
    curr_toks = ""

    total=[]
    sent_total=[]
    for batch in batches:
        sent_total.append([])
        #print(batch)
        batch_input, start_idx = batch
        output_ids = batch_input.input_ids.squeeze(0)[1:]

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(batch_input.input_ids.squeeze(0))[1:]
        index = torch.arange(0, output_ids.shape[0])
        surp = -1 * torch.log2(softmax(model_output.logits).squeeze(0)[index, output_ids])

        for i in range(start_idx, len(toks)):
            # necessary for diacritics in Dundee
            cleaned_tok = toks[i].replace("Ġ", "", 1).encode("latin-1").decode("utf-8")

            # for token-level surprisal
            # print(cleaned_tok, surp[i].item())

            # for word-level surprisal
            curr_word_surp.append(surp[i].item())
            curr_toks += cleaned_tok
            # summing subword token surprisal ("rolling")
            words[curr_word_ix] = words[curr_word_ix].replace(cleaned_tok, "", 1)
            if words[curr_word_ix] == "":
                total.append(sum(curr_word_surp))
                sent_total[-1].append(sum(curr_word_surp))
                #print(curr_toks, sum(curr_word_surp))
                curr_word_surp = []
                curr_toks = ""
                curr_word_ix += 1
    return total,sent_total

ppls=[]

ppls,ppls_sent=get_surp(sents)
import numpy as np
lengths=[]
# Print the perplexities for each word
for (i,row), (word, perplexities) in zip(unique_data.iterrows(),zip(text,ppls)):
#    print(f"{word}: {perplexities}")
 #   print(f"{row['word']}: {np.log2(row['meanItemRT'])}")
  #  print(f"{row['word']}: {row['meanItemRT']}")
    lengths.append(len(row['word']))

unique_data['ppl']=ppls
unique_data['len']=lengths
print(unique_data)

import numpy as np
import pymer4
from scipy.stats.stats import pearsonr
import numpy as np
from sklearn.linear_model import LinearRegression
print(len(ppls))
print(len(rts))
# import dataviz
import seaborn as sns

sns_plot = sns.regplot(x="ppl", y="meanItemRT", data=unique_data, fit_reg=True)
fig = sns_plot.get_figure()
fig.savefig("output.png")

def kl_divergence(surp):
    n=len(surp)
    surp_normalized = surp / np.sum(surp)
    uniform_distribution = np.full(n, 1 / n) * n
    return np.sum(surp_normalized * np.log2(surp_normalized / uniform_distribution))

def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def cv(surp):
    return np.std(surp) / np.mean(surp)

def sl(surp,k):
    return np.sum(np.asarray(surp)**k)/len(surp)


def eval_sent(rts_sent,ppls_sent,test=10):
    rts_sent_mean=[sum(r)/len(r) for r in rts_sent]
    rts_sent_sum=[sum(r) for r in rts_sent]
    for rts_goal in [rts_sent_mean,rts_sent_sum]:
        print("===============================================================================")
        fnames=["1","L","SL1.1","SL1.2","SL1.3","SL2.0","SL3.0","SL5.0","SL10.0","gini","kl","cv"]
        flist=[lambda surp: 1, lambda surp: sl(surp,1),lambda surp: sl(surp,1.1),lambda surp: sl(surp,1.2),
               lambda surp: sl(surp,1.3),lambda surp: sl(surp,2), lambda surp: sl(surp,3), lambda surp: sl(surp,5), lambda surp: sl(surp,10), gini_coefficient, kl_divergence, cv]
        for fn,f in zip(fnames,flist):
            out=[]
            print("********************************************")
            print(fn)
            print("********************************************")

            out_unnormalized=[]
            ns=[]
            surps=[]
            for surp in ppls_sent:
                n = len(surp)
                surps.append(surp)
                surp=np.asarray(surp)
                out.append(f(surp)*n)
                out_unnormalized.append(f(surp))
                ns.append(float(n))
            out_ns=np.asarray(list(zip(out_unnormalized,ns)))

            print("Pearson's r unnormalized")
            print(pearsonr(rts_goal, out_unnormalized))

            print("Pearson's r normalized")
            print(pearsonr(rts_goal, out))

            print("lin reg unnormalized no len")
            reg = LinearRegression().fit(np.asarray(out_unnormalized[:-test]).reshape(-1, 1), rts_goal[:-test])
            preds = reg.score(np.asarray(out_unnormalized[:-test]).reshape(-1, 1), rts_goal[:-test])
            print(preds)
            preds = reg.score(np.asarray(out_unnormalized[-test:]).reshape(-1, 1), rts_goal[-test:])
            print(preds)


            print("lin reg unnormalized with len")
            reg = LinearRegression().fit(out_ns[:-test], rts_goal[:-test])
            preds = reg.score(np.asarray(out_ns[:-test]), rts_goal[:-test])
            print(preds)
            preds = reg.score(np.asarray(out_ns[-test:]), rts_goal[-test:])
            print(preds)

            print("lin reg normalized no len")
            reg = LinearRegression().fit(np.asarray(out[:-test]).reshape(-1, 1), rts_goal[:-test])
            preds = reg.score(np.asarray(out[:-test]).reshape(-1, 1), rts_goal[:-test])
            print(preds)
            preds = reg.score(np.asarray(out[-test:]).reshape(-1, 1), rts_goal[-test:])
            print(preds)

            dataframe=pd.DataFrame({'measure':out_unnormalized,'rts':rts_goal,'len':ns})
            print("MER unnormalized no len")
            model = pymer4.Lm('rts ~ measure', data=dataframe)
            print(model.fit())

            dataframe=pd.DataFrame({'measure':out_unnormalized,'rts':rts_goal,'len':ns})
            print("MER unnormalized + len")
            model = pymer4.Lm('rts ~ measure + len', data=dataframe)
            print(model.fit())

            dataframe = pd.DataFrame({'measure': out_unnormalized, 'rts': rts_goal, 'len': ns})
            print("MER unnormalized * len")
            model = pymer4.Lm('rts ~ measure * len', data=dataframe)
            print(model.fit())

            dataframe=pd.DataFrame({'measure':out,'rts':rts_goal})
            print("MER normalized no len")
            model = pymer4.Lm('rts ~ measure', data=dataframe)
            print(model.fit())

            dataframe=pd.DataFrame({'measure':out,'rts':rts_goal,'len':ns})
            print("MER normalized + ")
            model = pymer4.Lm('rts ~ measure + len', data=dataframe)
            print(model.fit())

            dataframe = pd.DataFrame({'measure': out, 'rts': rts_goal, 'len': ns})
            print("MER normalized * len")
            model = pymer4.Lm('rts ~ measure * len', data=dataframe)
            print(model.fit())



            print("...............................................................................")

def eval(rts,ppls,test=1000):
    print("Pearson's r")
    print(pearsonr(np.log2(rts[:-test]),ppls[:-test]))
    print(pearsonr(rts[:-test],ppls[:-test]))
    print(pearsonr(np.log2(unique_data['meanItemRT'][:-test]),ppls[:-test]))
    print(pearsonr(unique_data['meanItemRT'][:-test],ppls[:-test]))

    print("linear regression")
    reg = LinearRegression().fit(np.asarray(ppls[:-test]).reshape(-1, 1),np.log2(rts[:-test]))
    preds=reg.score(np.asarray(ppls[:-test]).reshape(-1, 1),np.log2(rts[:-test]))
    print(preds)
    preds=reg.score(np.asarray(ppls[-test:]).reshape(-1, 1),np.log2(rts[-test:]))
    print(preds)

    reg = LinearRegression().fit(np.asarray(ppls[:-test]).reshape(-1, 1),rts[:-test])
    preds=reg.score(np.asarray(ppls[:-test]).reshape(-1, 1),rts[:-test])
    print(preds)
    preds=reg.score(np.asarray(ppls[-test:]).reshape(-1, 1),rts[-test:])
    print(preds)

    print("MER")
    model = pymer4.Lm('meanItemRT ~ ppl + len', data=unique_data)
    model.fit()

# Access model attributes
    print(model.BIC)
    print(model.residuals)

# Get fitted parameters
#    model.coef # population parameters
 #   model.fixef # group/cluster estimates (BLUPs)
  #  model.ranef # group/cluster deviates

import statsmodels.api as sm

import statsmodels.formula.api as smf


import random
#eval_sent(sent_rts_mean,ppls_sent)
eval_sent(sent_rts,ppls_sent)

exit()
eval(rts,ppls)

random.shuffle(ppls)
unique_data['ppl']=np.random.permutation(unique_data['ppl'].values)
unique_data['len']=np.random.permutation(unique_data['len'].values)
eval(rts,ppls)

