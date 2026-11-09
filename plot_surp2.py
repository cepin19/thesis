import sys

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, GPT2TokenizerFast
import transformers
import numpy as np
model_name="gpt2-xl"
#model_name = "bigscience/bloom-3b"
#model_name = "bigscience/bloom-1b1"

#model_name="facebook/opt-6.7b"



def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name,add_prefix_space=True)
#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",add_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype="auto",cache_dir="/home/enki/models/marian/perplex/models/",offload_folder="/home/enki/models/marian/perplex/models/")
model.eval()

def get_ppl_sentence(sentences):

    softmax = torch.nn.Softmax(dim=-1)
    ctx_size = 512 # model.config.max_position_embeddings
    bos_id = model.config.bos_token_id

    batches = []
    words = []
    total = []

    for sent in sentences:
        sent_surp=[]
        words=sent.strip().split(" ")
        tokenizer_output = tokenizer(sent.strip())
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask
        start_idx = 0
        if "gpt" in model_name:
            batch=transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids), "attention_mask": torch.tensor([1] + attn)})
        elif "opt" in model_name:

            batch=transformers.BatchEncoding({"input_ids":  torch.tensor([bos_id] + ids).unsqueeze(0),
                                                        "attention_mask": torch.tensor([1] +attn).unsqueeze(0)})
        else:
            batch=transformers.BatchEncoding({"input_ids":  torch.tensor([bos_id] + ids).unsqueeze(0),
                                                        "attention_mask": torch.tensor([1] +attn).unsqueeze(0)})

        curr_word_ix = 0
        curr_word_surp = []
        curr_toks = ""

        batch_input = batch
        output_ids = batch_input.input_ids.squeeze(0)[1:]

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(batch_input.input_ids.squeeze(0))[1:]
        index = torch.arange(0, output_ids.shape[0])
#        surp = -1 * torch.log2(softmax(model_output.logits).squeeze(0)[index, output_ids])
        print(toks)
        print(list(zip(toks,surp.tolist())))
        total.append(surp.tolist())
        if False:
          #  print(toks)
            for i in range(start_idx, len(toks)):
                # necessary for diacritics in Dundee
                #print(toks[i])
                cleaned_tok = toks[i].replace("Ġ", "", 1).encode("ascii", "ignore").decode("ascii")
                # for token-level surprisal

                # for word-level surprisal
                curr_word_surp.append(surp[i].item())
                curr_toks += cleaned_tok
                # summing subword token surprisal ("rolling")
                print(words)
                print(curr_word_ix)
                #FIX THIS!!!
                words[curr_word_ix] = words[curr_word_ix].encode("ascii", "ignore").decode("ascii").replace(cleaned_tok, "", 1)
                if words[curr_word_ix] == "":
                    sent_surp.append(sum(curr_word_surp))
                    #print(curr_toks, sum(curr_word_surp))
                    curr_word_surp = []
                    curr_toks = ""
                    curr_word_ix += 1
            total.append(sent_surp)
    return total
sentences=sys.stdin.readlines()


def kl_divergence(p, q):
    return np.sum(p * np.log2(p / q))

surp=get_ppl_sentence(sentences)
kls=[]
cvs=[]
sls1=[]
sls2=[]
ginis=[]
ls=[]
surps=[]
for surp,sent in zip(surp,sentences):
    #print(sent.strip())
    n = len(surp)
    surps.append(surp)
    surp=np.asarray(surp)
    if n==0:
        kl=0
        cv=0
        gini=0
        l=0
        sl1=0
        sl2=0
        kls.append(kl)
        cvs.append(cv)
        ginis.append(gini)
        continue
    surp_normalized = surp / np.sum(surp)
    l=np.sum(np.asarray(surp))
    sl1=np.sum(np.asarray(surp)**1.1)
    sl2=np.sum(np.asarray(surp)**1.2)
    uniform_distribution = np.full(n, 1 / n)
    kl = kl_divergence(surp_normalized, uniform_distribution)
    cv=np.std(surp) / np.mean(surp)
    gini=gini_coefficient(surp)
   # print("KL divergence: ", kl)
   # print("KL divergence: ", kl/n)
   # print("CV: ", cv)
  #  print("Gini: ", gini)
    cvs.append(cv)
    kls.append(kl)
    ginis.append(gini)
    sls1.append(sl1)
    sls2.append(sl2)
    ls.append(l)
df=pd.DataFrame(columns=["sentence","kl","cv","gini","l","sl1","sl2"])
df["sentence"]=sentences
df["kl"]=kls
df["cv"]=cvs
df["gini"]=ginis
df["l"]=ls
df["sl1"]=sls1
df["sl2"]=sls2
df["surp"]=surps

for i in range(len(df)):
    print(str(df.iloc[i]["sentence"].strip())+"\t"+str(df.iloc[i]["kl"])+"\t"+str(df.iloc[i]["cv"])+"\t"+str(df.iloc[i]["gini"])+"\t"+str(df.iloc[i]["l"])+"\t"+str(df.iloc[i]["sl1"])+"\t"+str(df.iloc[i]["sl2"]),end='\t')
    print(str(df.iloc[i]["surp"]))

import matplotlib.pyplot as plt
labels=["More uniform","Less uniform"]
for k,(sent,s) in enumerate(zip(sentences,surps)):

    plt.plot([i for i in range(1, len(s) + 1)], s, label=labels[k])
    print(sent)
plt.legend()
plt.xlabel('Word position')
plt.ylabel('Surprisal')
plt.legend()
plt.savefig("surp.pdf",bbox_inches='tight')
#for v in df.sort_values(by=["kl"],ascending=False)['sentence'].head(500):
#    print(v)
#print(df["kl"].mean())
#print(df["cv"].mean())
#print(df["gini"].mean())
#print(get_ppl2(sys.stdin.read()))
