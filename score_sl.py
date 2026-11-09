# Description: This script takes a list of sentences as input and outputs the surprisal of each sentence
import string
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, GPT2TokenizerFast
import transformers
import csv
import numpy as np
import random
from surprisal import AutoHuggingFaceModel

from sacremoses import MosesTokenizer, MosesDetokenizer
from surprisal import AutoHuggingFaceModel

# Processing commandline arguments
import argparse
# Add argument for puctuation removal
argparser = argparse.ArgumentParser(description='')
argparser.add_argument('--ignore_punct', default=False, action='store_true', help='Ignore punctuation')
argparser.add_argument('--model', default="MU-NLPC/CzeGPT-2", type=str, help='Model name')
args=argparser.parse_args()

moses_tokenizer=MosesTokenizer(lang='en')
moses_detokenizer=MosesDetokenizer(lang='en')

model_name="gpt2"
model_name = "bigscience/bloom-1b7"
#model_name="MU-NLPC/CzeGPT-2"
if args.model:
    model_name=args.model

#model_name="BUT-FIT/Czech-GPT-2-XL-133k"
outdir="results/"
SKIP_FIRST=False
if args.ignore_punct:
    ignore_punct=True
else:
    ignore_punct=False

#model = AutoHuggingFaceModel.from_pretrained(model_name, torch_dtype=torch.float16, model_class="gpt")
model = AutoHuggingFaceModel.from_pretrained(model_name,  precision="fp16", model_class="gpt")
model.to('cuda')

# Load GPT-2 tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained(model_name,add_prefix_space=True)
#model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype=torch.float16,cache_dir="/home/enki/models/marian/perplex/models/",offload_folder="/home/enki/models/marian/perplex/models/")
#model.eval()


def get_ppl2(sentences,m):
    all_surps=[]
    for i,sent in enumerate(sentences):
        #sent=sent.replace('é','e')
        words = sent.split(" ")
        orig_words = words.copy()
        [surp] = m.surprise(sent)
        sent_surps=[]
        curr_word_surp = [0.0]
        curr_toks = ""
        curr_word_ix = 0
        print(sent, file=sys.stderr)
        for tok, s in zip(surp.tokens, surp.surprisals):
            print(s)
            cleaned_tok = tok.replace("Ġ", "", 1)#.encode("latin-1")#.decode("utf-8")
            curr_word_surp.append(s)
            curr_toks += cleaned_tok

            words[curr_word_ix] = words[curr_word_ix].replace(cleaned_tok, "", 1)
            # whole word from the input covered by the subwords
            if words[curr_word_ix] == "":
                sent_surps.append(float(sum(curr_word_surp)))
                curr_word_surp = [0.0]
                curr_toks = ""
                curr_word_ix += 1
        if len(sent_surps)==0:
            print(sent)
            print(orig_words)
            print(words)
            print("ERROR!!!!!!!!!!!!!!!")
            print(s)
            print(surp)
            exit()
        all_surps.append(sent_surps)
    return all_surps


def get_ppl3(sentences,m):
    all_surps=[]
    for i,sent in enumerate(sentences):
        #sent=sent.replace('é','e')
        #print(sent.split(" "))
        words = sent.strip().split(" ")
        orig_words = words.copy()
        [surp] = m.surprise(sent)
        sent_surps=[]
        curr_word_surp = []
        #print(surp, file=sys.stderr)
        last_tok= ""
        curr_word=""
#        print(sent)
        if sent.strip()=='':
            all_surps.append([0.0])
            continue
        all_punct=True
        #print(surp)
        for i,(tok, s) in enumerate(zip(surp.tokens, surp.surprisals)):
            if tok.startswith("Ġ") or i==len(surp.tokens)-1:
                if curr_word_surp != []:
                    if ignore_punct and curr_word in string.punctuation:
                        pass
                        #print("ignored", last_tok)
                    else:
                        all_punct=False
                        sent_surps.append(float(sum(curr_word_surp)))
        #            print(curr_word, len(curr_word_surp), float(sum(curr_word_surp)))

                    curr_word_surp = [0.0]

                    curr_word=""
            curr_word_surp.append(s)
            last_tok = tok.replace("Ġ", "", 1)
            curr_word+=last_tok
        if all_punct:
            sent_surps.append(float(sum(curr_word_surp)))
        if len(sent_surps)==0:
            print(sent)
            print(orig_words)
            print(words)
            print("ERROR!!!!!!!!!!!!!!!")
            print(s)
            print(surp)
            exit()
        all_surps.append(sent_surps)
    return all_surps


def get_ppl_sentence(sentences,model):

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
            batch=transformers.BatchEncoding({"input_ids":  torch.tensor(ids).unsqueeze(0),
                                                        "attention_mask": torch.tensor(attn).unsqueeze(0)})

        curr_word_ix = 0
        curr_word_surp = []
        curr_toks = ""

        batch_input = batch
        output_ids = batch_input.input_ids.squeeze(0)#[1:]

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(batch_input.input_ids.squeeze(0))#[1:]
        index = torch.arange(0, output_ids.shape[0])
        surp = -1 * torch.log2(softmax(model_output.logits.float()).squeeze(0)[index, output_ids])
        #print(surp)
        #total.append(surp.tolist())
        curr_word_surp = []
        curr_word=""
        for i in range(start_idx, len(toks)):
            #print(toks[i])
            if toks[i].startswith("Ġ") or i == len(toks) - 1:
                if curr_word_surp != []:
                    #print(curr_word, len(curr_word_surp), float(sum(curr_word_surp)))
                    total.append(float(sum(curr_word_surp)))
                    curr_word_surp = [0.0]
                    curr_word=""
            curr_word += toks[i].replace("Ġ", "", 1)
            curr_word_surp.append(surp[i].item())
        total.append(float(sum(curr_word_surp)))

    return total

# from UID paper
def score_gpt(sentence, model, tokenizer, BOS=True):
    with torch.no_grad():
        all_log_probs = torch.tensor([], device=model.device)
        offset_mapping = []
        start_ind = 0

        while True:
            encodings = tokenizer(sentence[start_ind:], max_length=1022, truncation=True, return_offsets_mapping=True)
            if BOS:
                tensor_input = torch.tensor(
                    [[tokenizer.bos_token_id] + encodings['input_ids'] + [tokenizer.eos_token_id]], device=model.device)
            else:
                tensor_input = torch.tensor([encodings['input_ids'] + [tokenizer.eos_token_id]], device=model.device)
            output = model(tensor_input, labels=tensor_input)
            shift_logits = output['logits'][..., :-1, :].contiguous()
            shift_labels = tensor_input[..., 1:].contiguous()
            log_probs = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                          shift_labels.view(-1), reduction='none')
            assert torch.isclose(torch.exp(sum(log_probs) / len(log_probs)), torch.exp(output['loss']))
            offset = 0 if start_ind == 0 else STRIDE - 1
            all_log_probs = torch.cat([all_log_probs, log_probs[offset:-1]])
            offset_mapping.extend([(i + start_ind, j + start_ind) for i, j in encodings['offset_mapping'][offset:]])
            if encodings['offset_mapping'][-1][1] + start_ind == len(sentence):
                break
            start_ind += encodings['offset_mapping'][-STRIDE][1]
        return np.asarray(all_log_probs.cpu()), offset_mapping
if __name__=="__main__":
    surps=[]
    for s in sys.stdin:
        #if ignore_punct:
        #    s = s.translate(str.maketrans('', '', string.punctuation))
        #surps.append(score_gpt(s, model, tokenizer)[0])
        #surps.append(get_ppl_sentence(s,model)[0])
        surps.append(get_ppl3([s],model)[0])
        #print(s)
        #print(surps)
    for sent in surps:
        print(','.join(str(s) for s in sent))
