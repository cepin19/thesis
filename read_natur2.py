import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

import string
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, GPT2TokenizerFast
import transformers
import csv
model_name="gpt2"
#model_name = "bigscience/bloom-1b1"

from sentence_splitter import SentenceSplitter, split_text_into_sentences
from scipy.stats.stats import pearsonr,gmean
import pymer4

import numpy as np
import random
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm
from sklearn.neural_network import  MLPRegressor,MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from surprisal import AutoHuggingFaceModel
import sacremoses
# import dataviz
import seaborn as sns
from sacremoses import MosesTokenizer, MosesDetokenizer

# Processing commandline arguments
import argparse
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
lme4 = importr('lme4')
utils = importr('utils')
base = importr('base')
stats = importr('stats')


moses_tokenizer=MosesTokenizer(lang='en')
moses_detokenizer=MosesDetokenizer(lang='en')




parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', metavar='N', type=str,
                    help='model name')
# Output directory for results
parser.add_argument('--output_dir', metavar='N', type=str,
                    help='output directory')
parser.add_argument('--sent-seg', type=bool, default=False,
                    help='feed the sentences to LM one by one')
parser.add_argument('--skip-first', type=bool,
                    help='skip first word in each sentence in the measurements')
# If model set by user, use it instead of the desent-levelfault
model_name="gpt2"
outdir="results/"
SKIP_FIRST=False


if parser.parse_args().model:
    model_name = parser.parse_args().model
if parser.parse_args().output_dir:
    outdir = parser.parse_args().output_dir
    #create the output dir if it doesn't exist
    import os
    if not os.path.exists(outdir):
        os.makedirs(outdir)
if parser.parse_args().skip_first:
    SKIP_FIRST=parser.parse_args().skip_first
m = AutoHuggingFaceModel.from_pretrained(model_name)
m.to('cuda')
splitter = SentenceSplitter(language='en')

# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name,add_prefix_space=True)
#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",add_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype="auto",cache_dir="/home/enki/models/marian/perplex/models/",offload_folder="/home/enki/models/marian/perplex/models/")
model.eval()
with open('unigrams.csv', mode='r') as infile:
    reader=csv.reader(infile)
    unigram_mapping={rows[0]:-float(rows[1]) for rows in reader}
def get_unigram_log_prob(text,unigram_mapping):
    tokens = [moses_detokenizer.detokenize([t.lower()]) for t in moses_tokenizer.tokenize(text)]
    tokens=[t for t in tokens if t not in string.punctuation]
    return np.sum([unigram_mapping.get(k, np.nan) for k in tokens])

def get_sentence_freq(sen):
    words = [moses_detokenizer([t]) for t in moses_tokenizer(sen)]
    total = [get_unigram_log_prob(w.strip().strip(string.punctuation).lower()) for w in words]
    return np.nansum(list(filter(lambda x: x != None, total)))
    pass
def add_lau_accept_measures(df):
    df['log_freq'] = df.apply(lambda x: abs(get_unigram_log_prob(x['sentence'],unigram_mapping)), axis=1)
    df['slor'] = df['L'] - df['log_freq']/df['sent_len']
    df['normlp'] = df['L']*df['sent_len']/df['log_freq']
    return df

def read_cola(filename='cola_public/raw/in_domain_train.tsv'):
    df=pd.read_csv(filename, sep='\t',header=None, names=['sentence_source', 'label', 'label_author', 'sentence'])
    sentences=df['sentence'].to_list()
    df['sent_len']=df['sentence'].apply(lambda x: len(x.split(" ")))
    df['sent_chlen']=df['sentence'].apply(lambda x: len(x.replace(" ","")))
    return df

def read_ns(filename='naturalstories/naturalstories_RTS/processed_RTs_bkp.tsv'):
    df = pd.read_csv(filename, sep='\t')

    df=df.sort_values(by=['item', 'zone'])
    df['sent']=0
    unique_data = df.loc[~df[['item', 'zone']].duplicated(keep='first')]
    stories=[""]
    last_item = 1
    for index, row in unique_data.iterrows():
        if row['item'] != last_item:
            stories[-1]=stories[-1].strip()
            stories.append("")
            last_item = row['item']
        stories[-1] += row['word'].replace(' ', '') + ' '
        df.loc[(df['item']==row['item']) & (df['zone']==row['zone']),'chlen']=len(row['word'])
        df.loc[(df['item']==row['item']) & (df['zone']==row['zone']),'unigram_log_prob']=get_unigram_log_prob(row['word'],unigram_mapping)
    #df = pd.merge(df, unique_data[['zone', 'item']], on=['zone', 'item'], how='left')
    stories[-1] = stories[-1].strip()
    item=1
    df['sent']=0

    df['computedmeanItemRT'] = df.groupby(['zone','item'])['RT'].transform('mean')
    df['logmeanItemRT'] = np.log(df['meanItemRT'])
    df['meanUnigramLogProb'] = df.groupby(['zone','item'])['unigram_log_prob'].transform('mean')

    sent_id=0
    for story in stories:
        zone=1
        sents=splitter.split(story)
        for s in sents:
            #print("sentence split at", item, zone)
            df.loc[(df['zone'] >= zone) & (df['item'] == item),'sent']=sent_id
            df.loc[df['sent'] == sent_id, 'sent_len']=len(s.split(" "))
            sent_id+=1
            zone+=len(s.split(" "))
        item+=1
    return df

def read_brown(filename='brown/data/corpora/brown_spr.csv'):
    df=pd.read_csv(filename)
    df.rename({'text_pos':'zone','text_id':'item','time':'RT','subject':'WorkerId'},axis=1,inplace=True)
    df['zone'] = df['zone']+1
    df['item'] = df['item']+1
    df=df.sort_values(by=['item', 'zone'])
    df['sent']=0
    df['chlen']=0

    unique_data = df.loc[~df[['item', 'zone']].duplicated(keep='first')]
    stories=[""]
    last_item = 1
    for index, row in unique_data.iterrows():
        #    print(f"{row['word']}: {row['meanItemRT']}")
        if row['item'] != last_item:
            stories[-1]=stories[-1].strip()
            stories.append("")
            last_item = row['item']
        stories[-1] += row['word'].replace(' ', '') + ' '
        df.loc[(df['item']==row['item']) & (df['zone']==row['zone']),'chlen']=len(row['word'])
        df.loc[(df['item']==row['item']) & (df['zone']==row['zone']),'unigram_log_prob']=get_unigram_log_prob(row['word'],unigram_mapping)

    #df = pd.merge(df, unique_data[['zone', 'item']], on=['zone', 'item'], how='left')
    stories[-1] = stories[-1].strip()


    item=1
    df['sent']=0
    sent_id=0
    df['meanItemRT'] = df.groupby(['zone','item'])['RT'].transform('mean')
    df['gmeanItemRT'] = df.groupby(['zone','item'])['RT'].transform(gmean)
    df['meanUnigramLogProb'] = df.groupby(['zone','item'])['unigram_log_prob'].transform('mean')
    df['logmeanItemRT'] = np.log(df['meanItemRT'])
    for story in stories:
        zone=1
        sents=splitter.split(story)
        for s in sents:
            #print("sentence split at", item, zone)
            df.loc[(df['zone'] >= zone) & (df['item'] == item),'sent']=sent_id
            df.loc[df['sent'] == sent_id, 'sent_len']=len(s.split(" "))
            sent_id+=1
            zone+=len(s.split(" "))
        item+=1
    return df

def get_surp_story(stories,df):
    softmax = torch.nn.Softmax(dim=-1)
    ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id
    df['surp']=np.nan
    batches = []
    words = []
    story_id=0
    for story in stories:
        story_id+=1
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
                                start_idx,story_id))

            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)-1

        batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids),
                                                       "attention_mask": torch.tensor([1] + attn)}),
                           start_idx,story_id))
    curr_word_ix = 0
    curr_word_surp = [0.0]
    curr_toks = ""

    total=[]
    orig_words=words.copy()
    words_all=[]
    zone=1
    item=1
    for batch in batches:
        batch_input, start_idx, new_item = batch
        if new_item!=item:
            zone=1
            item=new_item
            print("NEW STORY")
        output_ids = batch_input.input_ids.squeeze(0)[1:]
        print(len(batch_input.input_ids))

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(batch_input.input_ids.squeeze(0))[1:]
        index = torch.arange(0, output_ids.shape[0])
        #surp = -1 * torch.log2(softmax(model_output.logits).squeeze(0)[index, output_ids])
        surp= -1 * torch.NN.functional.log_softmax(model_output.logits).squeeze(0)[index, output_ids]

        for i in range(start_idx, len(toks)):
            # necessary for diacritics in Dundee
            cleaned_tok = toks[i].replace("Ġ", "", 1).encode("latin-1").decode("utf-8")

            #print(toks[i], cleaned_tok, surp[i].item())

            # for token-level surprisal
            # print(cleaned_tok, surp[i].item())

            # for word-level surprisal
            #if cleaned_tok not in string.punctuation:
            curr_word_surp.append(surp[i].item())
            curr_toks += cleaned_tok
            words[curr_word_ix] = words[curr_word_ix].replace(cleaned_tok, "", 1)
            if words[curr_word_ix] == "":
                if not curr_toks=="": #TODO investigate

                    total.append(sum(curr_word_surp))
                    words_all.append(curr_toks)

                    if curr_toks!=df[(df['zone']==zone) & (df['item']==item)].iloc[0]['word'].replace(' ',''):
                        print("ERROR!!!!!!!!!!!!!!! Wrong word")
                        print(curr_toks)
                        print(df[(df['zone']==zone) & (df['item']==item)].iloc[0]['word'].replace(' ',''))
                        break
                    df.loc[(df['zone'] == zone) & (df['item'] == item),'surp']=float(sum(curr_word_surp))
                    zone+=1


                curr_word_surp = [0.0]
                curr_toks = ""
                curr_word_ix += 1
    for i,w in enumerate(words_all):
        if w!=orig_words[i]:
            print("ERROR!!!!!!!!!!!!!!!")
            break
    return total,df

def get_surp_cola(sentences,df):
    all_surps=[]
    for i,s in enumerate(sentences):
        s=s.replace('é','e')
        words = s.split(" ")
        orig_words = words.copy()
        [surp] = m.surprise(s)
        sent_surps=[]
        curr_word_surp = [0.0]
        curr_toks = ""
        curr_word_ix = 0

        for tok, s in zip(surp.tokens, surp.surprisals):
            # print(s)
            cleaned_tok = tok.replace("Ġ", "", 1)#.encode("latin-1").decode("utf-8")
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
            print(s)
            print(orig_words)
            print(words)
            print("ERROR!!!!!!!!!!!!!!!")
            print(s)
            print(surp)
            exit()
        all_surps.append(sent_surps)
    for fn, f in zip(fnames, flist):
        print(fn)
        print(len(all_surps))
        out=[]
        lens=[]
        for i,(sent_surp,sent) in enumerate(zip(all_surps,sentences)):
            if len(sent_surp)==0:
                print(i)
                print(sent)
                print("ERROR!!!!!!!!!!!!!!!")
                break
            fval = f(np.asarray(sent_surp),sent)
            out.append(fval)
            lens.append(float(len(sent_surp)))
        df.loc[:,fn] = out
        df.loc[:,fn+'_times_len'] = np.asarray(out)*np.asarray(lens)
    df=add_lau_accept_measures(df)
    return df

def get_surp_story2(stories,df):
    df['surp']=np.nan
    item=0
    for story in stories:
        #sentence by sentence
        if type(story)==list:
            pass
        #whole story
        elif type(story)==str:
            story=[story]
        else:
            raise Exception("Something wrong with the story, should be a list of sentences or a string")

        item+=1
        zone=1
        for s in story:
            s=s.replace(" --","--").replace('N. Y.','N.Y.').replace('N. H.','N.H.')
            curr_word_ix = 0
            curr_word_surp = [0.0]
            curr_toks = ""
            words = s.split(" ")
            [surp]=m.surprise(s)
            for tok,s in zip(surp.tokens,surp.surprisals):
                #print(s)
                #print(tok)
                cleaned_tok = tok.replace("Ġ", "", 1).encode("latin-1").decode("utf-8")

                # print(toks[i], cleaned_tok, surp[i].item())

                # for token-level surprisal
                # print(cleaned_tok, surp[i].item())

                # for word-level surprisal
                #if cleaned_tok not in string.punctuation:

                curr_word_surp.append(s)
                curr_toks += cleaned_tok
                words[curr_word_ix] = words[curr_word_ix].replace(cleaned_tok, "", 1)
                # whole word from the input covered by the subwords
                if words[curr_word_ix] == "":
                    if curr_toks != df[(df['zone'] == zone) & (df['item'] == item)].iloc[0]['word'].replace(' ', ''):
                        print("ERROR!!!!!!!!!!!!!!! Wrong word2")
                        print(curr_toks)
                        print(df[(df['zone'] == zone) & (df['item'] == item)].iloc[0]['word'].replace(' ', ''))
                        exit()
                    if sum(curr_word_surp)==np.nan:
                        print("NaN surprisal???")
                        print(curr_toks)
                        print(curr_word_surp)
                        exit()
                    df.loc[(df['zone'] == zone) & (df['item'] == item), 'surp'] = float(sum(curr_word_surp))
                    zone += 1
                    curr_word_surp = [0.0]
                    curr_toks = ""
                    curr_word_ix += 1
    return df


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

fnames = ["const", "L", "SL11", "SL12", "SL13", "SL14", "SL15", "SL175", "SL20", "SL30", "SL50", "slor1", "normlp1", "slor15", "normlp15" "gini", "kl", "cv"]
flist = [lambda surp,_: 1, lambda surp,_: sl(surp, 1), lambda surp,_: sl(surp, 1.1), lambda surp,_: sl(surp, 1.2),
         lambda surp,_: sl(surp, 1.3), lambda surp,_: sl(surp, 1.4), lambda surp,_: sl(surp, 1.5), lambda surp,_: sl(surp, 1.75),
         lambda surp,_: sl(surp, 2), lambda surp,_: sl(surp, 3), lambda surp,_: sl(surp, 5), lambda surp,sentence: slor(surp,sentence,k=1),
            lambda surp,sentence: normlp(surp,sentence,k=1), lambda surp,sentence: slor(surp,sentence,k=1.5), lambda surp,sentence: normlp(surp,sentence,k=1.5),
         gini_coefficient, kl_divergence, cv]

def lme_cross_val(formula, df, d_var, num_folds=10, shuffle=False):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    df['folds'] = pd.cut(range(1, len(df) + 1), bins=num_folds, labels=False)
    estimates = []

    for i in range(num_folds):
        test_data = df[df['folds'] == i]
        train_data = df[df['folds'] != i]
        train_data=train_data.dropna()
        test_data=test_data.dropna()

        pd.set_option('display.max_columns', None)

        print(train_data)
        print(train_data[train_data.isna().any(axis=1)])
        print(len(train_data['WorkerId']))
        with (ro.default_converter + pandas2ri.converter).context():
            train_r = ro.conversion.get_conversion().py2rpy(train_data)
            test_r = ro.conversion.get_conversion().py2rpy(test_data)

        model = lme4.lmer(formula, train_r, REML=False)
        print(dir(lme4))
        res=lme4.residuals_merMod(model)
        res=np.asarray(res)
        sigma=np.mean(res**2)
       # print(result)
       # sigma = np.mean(result.resid**2)

        test_data['predict'] = np.asarray(lme4.predict_merMod(model, newdata=test_r, allow_new_levels=True))
        estimate = np.log(norm.pdf(test_data[d_var], loc=test_data['predict'], scale=np.sqrt(sigma)))
        estimates.extend(estimate)

    return np.array(estimates)

def lme_cross_val3(formula, df, d_var, num_folds=10, shuffle=False):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    df['folds'] = pd.cut(range(1, len(df) + 1), bins=num_folds, labels=False)
    estimates = []

    for i in range(num_folds):
        test_data = df[df['folds'] == i]
        train_data = df[df['folds'] != i]
        train_data=train_data.dropna()
        test_data=test_data.dropna()

        pd.set_option('display.max_columns', None)

        print(train_data)
        print(train_data[train_data.isna().any(axis=1)])
        print(len(train_data['WorkerId']))
        model = MixedLM.from_formula(formula, train_data, groups=train_data['WorkerId'])
        result = model.fit(reml=False)
        print(result)
        sigma = np.mean(result.resid**2)

        test_data['predict'] = result.predict(test_data)
        estimate = np.log(norm.pdf(test_data[d_var], loc=test_data['predict'], scale=np.sqrt(sigma)))
        estimates.extend(estimate)

    return np.array(estimates)

from statsmodels.regression.mixed_linear_model import MixedLM
def lme_cross_val2(formula, df, d_var, num_folds=10, shuffle=False):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    df['fold'] = pd.cut(range(1, len(df) + 1), bins=num_folds, labels=False)
    estimates = []

    for i in range(num_folds):
        test_data = df[df['fold'] == i]
        train_data = df[df['fold'] != i]
        print(train_data)
        print(train_data[train_data.isna().any(axis=1)])

        model = pymer4.Lmer(formula, data=train_data)
        result = model.fit(REML=False)
        print(model.summary())
        sigma = np.std(model.residuals)


        test_data['predict'] = model.predict(test_data)
        estimate = np.log(norm.pdf(test_data[d_var], loc=test_data['predict'], scale=sigma))
        estimates.extend(estimate)

    return np.array(estimates)


def lm_cross_val2(formula, df, d_var, family="gaussian", num_folds=10, shuffle=False):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    df['fold'] = pd.cut(range(1, len(df) + 1), bins=num_folds, labels=False)
    estimates = []

    for i in range(num_folds):
        test_data = df[df['fold'] == i]
        train_data = df[df['fold'] != i]

        model = pymer4.Lm(formula, family=family, data=train_data)
        model.fit()
        print(test_data)
        print(model.data)
        print(model.residuals)
        #sigma = np.std(model.residuals) ???

        predictions = model.predict(test_data,pred_type="response")
        print("predictions")
        print(predictions)
#        estimate = np.log(norm.pdf(test_data[d_var], loc=predictions, scale=sigma))
        estimate = np.log(norm.pdf(test_data[d_var], loc=predictions))
        estimates.extend(estimate)
    return np.array(estimates)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

def lm_cross_val3(formula, df, d_var, family=sm.families.Gaussian(), num_folds=10, shuffle=False):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    df['fold'] = pd.cut(range(1, len(df) + 1), bins=num_folds, labels=False)
    estimates = []
    accs_all=[]
    accs_baseline=[]
    for i in range(num_folds):
        test_data = df[df['fold'] == i]
        train_data = df[df['fold'] != i]

        if family == sm.families.Binomial():
            model = smf.glm(formula, data=train_data, family=family).fit()
            predictions = model.predict(test_data, linear=False)
        else:
            model = smf.glm(formula, data=train_data, family=family).fit()
            predictions = model.predict(test_data)

        pred=np.rint(predictions)

        sigma = model.scale ** 0.5
        estimate = np.log(norm.pdf(test_data[d_var], loc=predictions, scale=sigma))
        estimates.extend(estimate)
        accs_all.append(accuracy_score(test_data[d_var], pred))
        accs_baseline.append(accuracy_score(test_data[d_var], np.ones(len(test_data))))

    return np.array(estimates),accs_all,accs_baseline


def lm_cross_val(formula, df, d_var, family=sm.families.Gaussian(), num_folds=10, shuffle=False):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    df['fold'] = pd.cut(range(1, len(df) + 1), bins=num_folds, labels=False)
    df=df.drop(columns=['label_author'])
    estimates = []
    accs_all=[]
    accs_baseline=[]
    pd.set_option('display.max_columns', None)
    for i in range(num_folds):
        test_data = df[df['fold'] == i]
        train_data = df[df['fold'] != i]
        with (ro.default_converter + pandas2ri.converter).context():
            train_r = ro.conversion.get_conversion().py2rpy(train_data)
            test_r = ro.conversion.get_conversion().py2rpy(test_data)
        model = stats.glm(formula, train_r,family='binomial')
        sigma = stats.sigma(model)
        test_data['predict'] = np.asarray(stats.predict(model, newdata=test_r, type="response"))
        estimate = np.log(norm.pdf(test_data[d_var], loc=test_data['predict'], scale=np.sqrt(sigma)))
        estimates.extend(estimate)
    return np.array(estimates),[],[]


def eval_word(df,dataset_name="", shuffle=False, outdir="results/"):
    """"Evaluate relationship between surprisal and RT on the word level"""
    print(df)
    print("LENS")
    print(len(df))
    df=df.dropna(subset=['unigram_log_prob'])
    print(len(df))
    with open(f'{outdir}/{dataset_name}_word', 'w') as file:
        results = []
        for m in ['meanItemRT', 'gmeanItemRT', 'logmeanItemRT', "RT"]:
            print(m,file=file)
            if "mean" in m:
                df_words=df.loc[~df[['zone', 'item']].duplicated(keep='first')]
            else:
                df_words=df
            surp = np.asarray(df_words['surp'])
            chlen=np.asarray(df_words['chlen'])
            surp_with_chlen=np.asarray(list(zip(df_words['surp'], df_words['chlen'])))
            surp_times_chlen=np.asarray(df_words['surp'])*np.asarray(df_words['chlen'])
            unigram_log_prob=np.asarray(df_words['unigram_log_prob'])
            unigram_with_chlen=np.asarray(list(zip(df_words['unigram_log_prob'], df_words['chlen'])))
            unigram_times_chlen=np.asarray(df_words['unigram_log_prob'])*np.asarray(df_words['chlen'])


            y = np.asarray(df_words[m])

            for xname,x in [("Surprisal",surp),("Character length",chlen),  ("Multiply",surp_times_chlen), ("Both",surp_with_chlen),
                            ("Unigram log prob",unigram_log_prob), ("Unigram log prob times chlen",unigram_times_chlen),
                            ("Unigram log prob and chlen",unigram_with_chlen)]:
                print(xname,file=file)
                train_x = np.asarray(x[:-500])
                test_x = np.asarray(x[-500:])
                train_y = np.asarray(y[:-500])
                test_y = np.asarray(y[-500:])
                print(train_x.ndim)
                if x.ndim==1:
                    train_x=train_x.reshape(-1,1)
                    test_x=test_x.reshape(-1,1)


                try:
                    print(pearsonr(x, y), file=file)
                    #add to results
                    results.append({'measure': xname, 'pearson': pearsonr(x, y)})

                except:
                    print("can't compute pearson", file=file)
                    results.append({'measure': xname, 'pearson': np.nan})
                print("lin reg", file=file)
                reg = LinearRegression().fit(train_x, train_y)
                preds = reg.score(train_x, train_y)
                results.append({'measure': xname, 'linreg_train': preds})
                print(preds, file=file)
                preds = reg.score(test_x, test_y)
                results.append({'measure': xname, 'linreg_test': preds})
                print(preds, file=file)
                print("mlp", file=file)
                reg = MLPRegressor().fit(train_x, train_y)
                preds = reg.score(train_x, train_y)
                results.append({'measure': xname, 'mlp_train': preds})
                print(preds, file=file)
                preds = reg.score(test_x, test_y)
                results.append({'measure': xname, 'mlp_test': preds})
                print(preds, file=file)
                results=pd.DataFrame(results,columns=['measure', 'pearson', 'linreg_train', 'linreg_test', 'mlp_train', 'mlp_test'])
                results.to_csv(f'{outdir}/{dataset_name}_word_results.csv')

                print(".........................................................................................", file=file)




def eval_df(surps_sent,df,test=500, dataset_name="", shuffle=False, outdir="results/"):
    df_sent=df.loc[~df[['sent', 'WorkerId']].duplicated(keep='first')]
    df_sent=df_sent.dropna(subset='TotalRT')
    print(df_sent)

    for fn, f in zip(fnames, flist):
        print(fn)
        out = []
        out_times_len = []
        lens = []
        rts_target_all=[]
        rts_workermean_target=[]
        rts_sentencemean_target=[]
        rts_workersentencemean_target=[]
        out_single=[]
        out_times_len_single=[]
        lens_single=[]
        workers=[]
        wrongs=0
        for i in range(len(df_sent['sent'].unique())):
            sents=df_sent[df_sent['sent'] == i]
            if len(sents) == 0:
                print(i)
                print(df_sent['sent'])
                print("ERROR!!!!! no data for sent")
                break
            #print(df_sent[df_sent['sent']==i])
            #print("************************************************************************************************")
            surp=df[df['sent'] == i]['surp'].drop_duplicates().to_numpy()
            #surp=surps_sent[i]
            #if not np.allclose(np.asarray(ss),np.asarray(surp)):
                #print("ERROR!!!!!!! wrong surp")
                #break
            n = len(surp)
            fval = f(surp)
            for idx,rt in sents.iterrows():
                if int((rt['sent_len']))!=n:
                    print("ERROR!!!!!!! wrong sent len")
                    print(df_sent[df_sent['sent']==i]['word'].unique())
                    print(rt)
                    print(n)
                    print(df_sent[df_sent['sent']==i].to_string())
                    wrongs+=1
                    break
                    exit()
                out_times_len.append(fval * n)
                out.append(fval)
                lens.append(float(n))
                rts_target_all.append(rt['TotalRT'])
                rts_sentencemean_target.append(rt['TotalRT']/n)
                worker_mean=df_sent[df_sent['sent']==i].sum()['TotalRT']/len(df_sent[df_sent['sent']==i])
                rts_workermean_target.append(worker_mean)
                rts_workersentencemean_target.append(worker_mean/n)
                workers.append(rt['WorkerId'])
            out_single.append(fval)
            out_times_len_single.append(fval*n)
            lens_single.append(float(len(surp)))
        df_sent.loc[:,fn]=out
        df_sent.loc[:,fn+'_times_len']=out_times_len
        df_sent.loc[:,'sent_len']=lens
        df_sent.loc[:,'TotalRT']=rts_target_all
        df_sent.loc[:,'MeanOverWorkersRT']=rts_workermean_target
        df_sent.loc[:,'MeanOverWordsRT']=rts_sentencemean_target
        df_sent.loc[:,'MeanOverWordsWorkerRT']=rts_workersentencemean_target
        df_sent.loc[:, 'logTotalRT'] = np.log(df_sent['TotalRT'])
        df_sent.loc[:, 'logMeanOverWorkersRT'] = np.log(df_sent['MeanOverWorkersRT'])
        df_sent.loc[:, 'logMeanOverWordsRT'] = np.log(df_sent['MeanOverWordsRT'])
        df_sent.loc[:, 'logMeanOverWordsWorkerRT'] = np.log(df_sent['MeanOverWordsWorkerRT'])
    print(df_sent)
    for measure in fnames+['meanUnigramLogProb']:
        print("********************************************")
        print(measure)
        print("********************************************")

        if shuffle:
            df_sent_shuf=df_sent.sample(n=len(df_sent), random_state=1)
        else:
            df_sent_shuf = df_sent


        for rts_t in ['TotalRT','MeanOverWordsRT','MeanOverWorkersRT','MeanOverWordsWorkerRT', 'logTotalRT','logMeanOverWordsRT','logMeanOverWorkersRT','logMeanOverWordsWorkerRT']:
            col_list=['sent', 'WorkerId', 'sent_len', rts_t]
            col_list.extend([m+'_times_len' for m in fnames])
            col_list.extend([m for m in fnames])
            df_sent_tgt= df_sent_shuf[col_list]
            test=500
            if rts_t in ['MeanOverWorkersRT','MeanOverWordsWorkerRT', 'logMeanOverWorkersRT','logMeanOverWordsWorkerRT']:
                df_sent_tgt['WorkerId']=0
                test=40

            df_sent_tgt.drop_duplicates(inplace=True)
            y = df_sent_tgt[rts_t]
            out = df_sent_tgt[measure]
            out_times_len = df_sent_tgt[measure + "_times_len"]
            lens = df_sent_tgt['sent_len']
            out_lens = np.asarray(list(zip(out, lens)))

            with open(f'{outdir}/{dataset_name}_{measure}_{rts_t}', 'w') as file:
                for xname,x in [("metric",out), ("metric times len",out_times_len), ("metric and len",out_lens)]:
                    print(xname, file=file)
                    #print(y,file=file)
                    print(measure, file=file)
                    print("len all", len(out), file=file)
                    print(len(out), file=file)

                    train_x = np.asarray(x[:-test])
                    test_x = np.asarray(x[-test:])
                    train_y = np.asarray(y[:-test])
                    test_y = np.asarray(y[-test:])
                    if x.ndim == 1:
                        train_x = train_x.reshape(-1, 1)
                        test_x = test_x.reshape(-1, 1)
                    try:
                        print(pearsonr(x, y), file=file)
                        # add to results
                        results.append({'target': rts_t, 'measure': xname, 'pearson': pearsonr(x, y)})
                    except:
                        print("can't compute pearson", file=file)
                        results.append({'target': rts_t, 'measure': xname, 'pearson': np.nan})

                    print("lin reg", file=file)
                    reg = LinearRegression().fit(train_x, train_y)
                    preds = reg.score(train_x, train_y)
                    results.append({'target': rts_t, 'measure': xname, 'linreg_train': preds})
                    print(preds, file=file)
                    preds = reg.score(test_x,test_y)
                    results.append({'target': rts_t, 'measure': xname, 'linreg_test': preds})
                    print(preds, file=file)

                    print("svr", file=file)
                    reg = svm.SVR().fit(train_x, train_y)
                    preds = reg.score(train_x, train_y)
                    results.append({'target': rts_t, 'measure': xname, 'svr_train': preds})
                    print(preds, file=file)
                    preds = reg.score(test_x, test_y)
                    results.append({'target': rts_t, 'measure': xname, 'svr_test': preds})
                    print(preds, file=file)

                    print("mlp", file=file)
                    reg = MLPRegressor().fit(train_x, train_y)
                    preds = reg.score(train_x, train_y)
                    results.append({'target': rts_t, 'measure': xname, 'mlp_train': preds})
                    print(preds, file=file)
                    preds = reg.score(test_x,test_y)
                    results.append({'target': rts_t, 'measure': xname, 'mlp_test': preds})
                    print(preds, file=file)

                    try:
                        print("mlp (scaled)", file=file)
                        scaler = StandardScaler()
                        scaler.fit(train_x)
                        out_lens_scaled = scaler.transform(out_lens)
                        reg = MLPRegressor().fit(out_lens_scaled[:-test], train_y)
                        preds = reg.score(out_lens_scaled[:-test], train_y)
                        print(preds, file=file)

                        preds = reg.score(out_lens_scaled[-test:], test_y)
                        print(preds, file=file)
                    except:
                        pass
                    print("...................................................", file=file)

                print("Lmer baseline", file=file)
                model = pymer4.Lmer(rts_t+" ~ "  +"sent_len + I(sent_len*meanUnigramLogProb)*sent_chlen + (sent_len+0 | WorkerId) ", data=df_sent_shuf[:-test])
                print(model.fit(REML=False), file=file)
                #print(model.summary(), file=file)
                print("LogLike", model.logLike, file=file)

                print("Lmer baseline2", file=file)
                model = pymer4.Lmer(rts_t + " ~ " + "sent_len + I(sent_len)*sent_chlen + (sent_len+0 | WorkerId) ",
                                    data=df_sent_shuf[:-test])
                print(model.fit(REML=False), file=file)
                # print(model.summary(), file=file)
                print("LogLike", model.logLike, file=file)

                # print("Lmer", file=file)
                # model = pymer4.Lmer(rts_t+' ~ '+measure+':sent_len + ( '+ measure  +' + sent_len + 0 | WorkerId) + ('+'+'
                #                     .join(["const", "L", "SL11", "SL12", "SL13", "SL20", "SL30", "gini", "kl", "cv"])+')', data=df_sent_shuf[:-test])
                # #    baseline <- lme_cross_val("time_sum ~   time_count_nonzero +len + I(len*uni_log_prob_power_1.0)*ch_len + (  len+0 | WorkerId_) ",
                #
                # print(model.fit(REML=False), file=file)
                # print(model.summary(), file=file)
                # print("LogLike", model.logLike, file=file)
                #
                # print("Lmer", file=file)
                # model = pymer4.Lmer(rts_t+' ~ '+measure+':sent_len + ( sent_len + 0 | WorkerId) + ('+'+'.
                #                     join(["const", "L", "SL11", "SL12", "SL13", "SL20", "SL30", "gini", "kl", "cv"])+')', data=df_sent_shuf[:-test])
                # print(model.fit(REML=False), file=file)
                # print(model.summary(), file=file)
                # print("LogLike", model.logLike, file=file)
                #

                print("Lmer main", file=file)

                model = pymer4.Lmer(rts_t + ' ~  '+ measure + ':sent_len + sent_len + (sent_len*meanUnigramLogProb)*sent_chlen+('+measure+'+sent_len + 0 | WorkerId)',
                                    data=df_sent_shuf[:-test])
                print(model.fit(REML=False), file=file)
                #print(model.summary(), file=file)
                print("LogLike", model.logLike, file=file)
                #print(model.predict(df_sent[-test:]), file=file)

                print("Lmer sent", file=file)

                model = pymer4.Lm(rts_t + ' ~  sent_len',
                                    data=df_sent_shuf[:-test])
                print(model.fit(REML=False), file=file)
                #print(model.summary(), file=file)
                print("LogLike", model.logLike, file=file)
                #print(model.predict(df_sent[-test:]), file=file)

                print("Lmer measure:sent_len", file=file)

                model = pymer4.Lm(rts_t + ' ~  '+ measure + ':sent_len',
                                    data=df_sent_shuf[:-test])
                print(model.fit(REML=False), file=file)
                #print(model.summary(), file=file)
                print("LogLike", model.logLike, file=file)
                #print(model.predict(df_sent[-test:]), file=file)

                print("Lmer measure", file=file)

                model = pymer4.Lm(rts_t + ' ~  ' + measure,
                                  data=df_sent_shuf[:-test])
                print(model.fit(REML=False), file=file)
                # print(model.summary(), file=file)
                print("LogLike", model.logLike, file=file)
                # print(model.predict(df_sent[-test:]), file=file)

                print("Lmer masdasdASDASDASDASDASD")
                #formula=rts_t + " ~ " + "sent_len + I(sent_len*meanUnigramLogProb)*sent_chlen"
                #formula=rts_t + " ~ " + "sent_len + I(sent_len*meanUnigramLogProb)*sent_chlen+("+measure+"+sent_len + 0 | WorkerId)"
                formula = rts_t + " ~ " + "sent_len + sent_len*meanUnigramLogProb*sent_chlen+(" + measure + "+sent_len + 0 | WorkerId)"
                baseline=lme_cross_val(formula, df_sent, rts_t, num_folds=10, shuffle=False)
                print(np.mean(baseline), file=file)
                results.append({'target': rts_t, 'measure': xname, 'lme baseline': np.mean(baseline)})
                #formula= rts_t + ' ~  ' + measure + ':sent_len + I(sent_len*meanUnigramLogProb)*sent_chlen+('+measure+'+sent_len + 0 | WorkerId)'
                formula = rts_t + ' ~  ' + measure + ':sent_len + sent_len*meanUnigramLogProb*sent_chlen+(' + measure + '+sent_len + 0 | WorkerId)'
                cv=lme_cross_val(formula, df_sent, rts_t, num_folds=10, shuffle=False)
                print(np.mean(cv), file=file)
                results.append({'target': rts_t, 'measure': xname, 'lme': np.mean(cv)})
                print(np.mean(cv - baseline), np.var(cv - baseline) / len(cv), np.mean(cv), file=file)
                results.append({'target': rts_t, 'measure': xname, 'lme diff': np.mean(cv - baseline)})
                results=pd.DataFrame(results,columns=['target','measure','pearson','linreg_train','linreg_test','svr_train','svr_test','mlp_train','mlp_test'])

                results.to_csv(f'{outdir}/{dataset_name}_{measure}_{rts_t}_results.csv')

                print("...............................................................................")


                print("...............................................................................")

def eval_cola(df,dev=None, dataset_name="", shuffle=False, outdir="results/"):
    for fn in fnames+[f+"_times_len" for f in fnames]:

        with (open(f'{outdir}/{dataset_name}_{fn}', 'w') as file):
            results={}
            df=df.fillna(0.0) #dropna(subset=['slor','normlp'])
            df_majority = df[df.label == 1]
            df_minority = df[df.label == 0]

            # Upsample minority class
            df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True,
                                                       random_state=123)  # Random state for reproducibility

            # Combine majority class with upsampled minority class
            df_upsampled = pd.concat([df_majority, df_minority_upsampled])

            # Display new class counts
            df=df_upsampled
            x=df[fn]
            y=df['label']
            train_x = np.asarray(x[:-500])
            train_y = np.asarray(y[:-500])
            test_x = np.asarray(x[-500:])
            test_y = np.asarray(y[-500:])
            print("baseline", file=file)
            print(accuracy_score(train_y, [1 for i in range(len(train_x))]), file=file)
            print(accuracy_score(test_y, [1 for i in range(len(test_x))]), file=file)
            results['baseline']= accuracy_score(train_y, [1 for i in range(len(train_x))])
            results['baseline_test']= accuracy_score(test_y, [1 for i in range(len(test_x))])
            if dev is not None:
                dev_x=np.asarray(dev[fn])
                dev_y=np.asarray(dev['label'])
                print("baseline dev",file=file)
                print(accuracy_score(dev_y, [1 for i in range(len(dev_x))]), file=file)
            print(fn, file=file)
            print(pearsonr(train_x, train_y), file=file)
            results['pearson']=pearsonr(train_x, train_y)[0]
            print(train_x.reshape(-1,1))
            print(train_y)
            print("svm", file=file)
            clf = svm.SVC()
            reg = clf.fit(train_x.reshape(-1,1), train_y)
            preds = reg.score(train_x.reshape(-1,1), train_y)
            print(preds, file=file)
            results['svm_train']=preds
            preds = reg.score(test_x.reshape(-1,1), test_y)
            print(preds, file=file)
            results['svm_test']=preds
            if dev is not None:
                preds = reg.score(dev_x.reshape(-1,1), dev_y)
                print(preds, file=file)

            print("mlp", file=file)
            reg = MLPClassifier().fit(train_x.reshape(-1,1), train_y)
            preds = reg.score(train_x.reshape(-1,1), train_y)
            print(preds, file=file)
            results['mlp_train']=preds
            preds = reg.score(test_x.reshape(-1,1), test_y)
            print(preds, file=file)
            results['mlp_test']=preds
            if dev is not None:
                preds = reg.score(dev_x.reshape(-1,1), dev_y)
                print(preds, file=file)

            print("mlp (scaled)", file=file)
            scaler = StandardScaler()
            scaler.fit(train_x.reshape(-1, 1))
            train_x_scaled=scaler.transform(train_x.reshape(-1, 1))
            reg = MLPClassifier().fit(train_x_scaled, train_y)
            preds = reg.score(train_x_scaled, train_y)
            print(preds, file=file)
            results['mlp_scaled_train']=preds
            test_x_scaled=scaler.transform(test_x.reshape(-1, 1))
            preds = reg.score(test_x_scaled, test_y)
            print(preds, file=file)
            results['mlp_scaled_test']=preds
            model = pymer4.Lm(
                'label ~ sent_len',
                data=df)
            print(model.fit(REML=False), file=file)

            # print(model.summary(), file=file)
            print("LogLike", model.logLike, file=file)

            model = pymer4.Lm(
                'label ~ ' + fn + ':sent_len',
                data=df)
            print(model.fit(REML=False), file=file)
            # print(model.summary(), file=file)
            print("LogLike", model.logLike, file=file)

            model = pymer4.Lm(
                'label ~ ' + fn ,
                data=df)
            print(model.fit(REML=False), file=file)
            # print(model.summary(), file=file)
            print("LogLike", model.logLike, file=file)


            print("=============================================", file=file)
            family=sm.families.Binomial()

            print("baseline", file=file)
            baseline, accs, accs_base = lm_cross_val(f"label ~ sent_len", df, 'label', family)
            print(baseline,file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)
            results['lme_baseline']=np.mean(baseline)
            print("baseline 2", file=file)
            baseline, accs, accs_base = lm_cross_val(f"label ~ sent_len + slor + normlp", df, 'label', family)
            print(baseline,file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)
            results['lme_baseline2']=np.mean(baseline)

            print("baseline 3", file=file)
            baseline, accs, accs_base = lm_cross_val(f"label ~  slor + normlp", df, 'label', family)
            print(baseline, file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)
            results['lme_baseline3']=np.mean(baseline)


            print("baseline 4", file=file)
            baseline, accs, accs_base = lm_cross_val(f"label ~  1", df, 'label', family)
            print(baseline, file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)
            results['lme_baseline4']=np.mean(baseline)


            print(f"{fn}", file=file)
            formula = f"label ~ {fn}"
            cv,accs, accs_base = lm_cross_val(formula, df, 'label', family)
            print(cv,file=file)
            print(np.mean(cv - baseline), np.var(cv - baseline) / len(cv), np.mean(cv), file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)
            results['lme_fn']=np.mean(baseline)


            print(f"{fn}+sent_len", file=file)
            formula = f"label ~ {fn} + sent_len"
            cv,accs, accs_base = lm_cross_val(formula, df, 'label', family)
            print(cv,file=file)
            print(np.mean(cv - baseline), np.var(cv - baseline) / len(cv), np.mean(cv), file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)

            print(f"{fn}+sent_len + slor + normlp", file=file)
            formula = f"label ~ {fn} + sent_len + slor + normlp"
            cv,accs, accs_base = lm_cross_val(formula, df, 'label', family)
            print(cv,file=file)
            print(np.mean(cv - baseline), np.var(cv - baseline) / len(cv), np.mean(cv), file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)
            results['lme_fn_len_slor_normlp']=np.mean(baseline)

            print(f"{fn}:sent_len + slor + normlp", file=file)
            formula = f"label ~ {fn}:sent_len + slor + normlp"
            cv, accs, accs_base = lm_cross_val(formula, df, 'label', family)
            print(cv, file=file)
            print(np.mean(cv - baseline), np.var(cv - baseline) / len(cv), np.mean(cv), file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)
            results['lme_fn:len_slor_normlp']=np.mean(baseline)


            print(f"{fn}:sent_len", file=file)
            formula = f"label ~ {fn}:sent_len"
            cv,accs, accs_base = lm_cross_val(formula, df, 'label', family)
            print(cv,file=file)
            print(np.mean(cv - baseline), np.var(cv - baseline) / len(cv), np.mean(cv), file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)

            formula = f"label ~ {fn}:sent_len + sent_len"
            cv,accs, accs_base = lm_cross_val(formula, df, 'label', family)
            print(cv,file=file)
            print(np.mean(cv - baseline), np.var(cv - baseline) / len(cv), np.mean(cv), file=file)
            #print(np.mean(accs), file=file)
            #print(np.mean(accs_base), file=file)
            print(results)
            results=pd.DataFrame.from_dict(results,index=[0])
            print(results)
            results.to_csv(f'{outdir}/{dataset_name}_{fn}_results.csv')


            print("...................................................", file=file)


def compute_metrics(surps,df):
    for fn, f in zip(fnames, flist):
        out = []
        lens = []
        for sent_surp in surps:
            fval = f(np.asarray(sent_surp))
            out.append(fval)
            lens.append(float(len(sent_surp)))
        df.loc[:, fn] = out
        df.loc[:, fn+'_times_len'] = np.asarray(out)*np.asarray(lens)
    return df


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


def filter_old(df):
    sent_rt = df.groupby(['sent', 'WorkerId', 'sent_len'])['RT'].sum().reset_index()
    removed = 0
    for sent in sent_rt['sent'].unique():
        sent_pd= sent_rt.loc[sent_rt['sent'] == sent]
        Q1 = sent_pd['RT'].quantile(0.25)
        Q3 = sent_pd['RT'].quantile(0.75)
        IQR = Q3 - Q1
        for i,r in sent_pd.iterrows():
            if r['RT'] < Q1 - 1.5 * IQR or r['RT'] > Q3 + 1.5 * IQR:
                removed+=1
                sent_rt=sent_rt[~((sent_rt['WorkerId'] == r['WorkerId']) & (sent_rt['sent'] == r['sent']) & (sent_rt['RT'] == r['RT']))]
    print('removed', removed)
    sent_rt.rename({'RT':'TotalRT'},axis=1,inplace=True)
    return sent_rt

def filter_z(df):
    sent_rt = df.groupby(['sent', 'WorkerId', 'sent_len'])['RT'].sum().reset_index()
    removed = 0
    from scipy.stats import zscore
    z_scores = zscore(np.log(sent_rt['RT']))
    abs_z_scores = np.abs(z_scores)
    sent_rt.loc[:, 'outlier'] = abs_z_scores > 3
    print("Percentage of outliers:", sum(sent_rt['outlier']) / len(sent_rt))
    for sent in sent_rt['sent'].unique():
        sent_pd= sent_rt.loc[sent_rt['sent'] == sent]
        for i,r in sent_pd.iterrows():
            if r['outlier'] == True:
                removed+=1
                sent_rt=sent_rt[~((sent_rt['WorkerId'] == r['WorkerId']) & (sent_rt['sent'] == r['sent']) & (sent_rt['RT'] == r['RT']))]
    print('removed', removed)
    sent_rt.rename({'RT':'TotalRT'},axis=1,inplace=True)
    return sent_rt
def filter_zz(df):
    sent_rt = df.groupby(['sent', 'WorkerId', 'sent_len'])['RT'].sum().reset_index()
    removed = 0
    from scipy.stats import zscore
    z_scores = zscore(np.log(df['RT']))
    abs_z_scores = np.abs(z_scores)
    sent_rt.loc[:, 'outlier'] = abs_z_scores > 3
    print("Percentage of outliers:", sum(sent_rt['outlier']) / len(sent_rt))
    for sent in sent_rt['sent'].unique():
        sent_pd= sent_rt.loc[sent_rt['sent'] == sent]
        for i,r in sent_pd.iterrows():
            if r['outlier'] == True:
                removed+=1
                sent_rt=sent_rt[~((sent_rt['WorkerId'] == r['WorkerId']) & (sent_rt['sent'] == r['sent']) & (sent_rt['RT'] == r['RT']))]
    print('removed', removed)
    sent_rt.rename({'RT':'TotalRT'},axis=1,inplace=True)
    return sent_rt
def filter_IQR(df):
    def detect_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return ~series.between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Detect outliers in the ReadingTime column
    df['Outlier_iqr'] = detect_outliers(df['RT'])

    # Group by Sentence and WorkerId to find sentences with outliers
    sentence_outliers = df.groupby(['sent', 'WorkerId'])['Outlier_iqr'].any().reset_index()

    # Merge this information back into the original DataFrame
    df = df.merge(sentence_outliers, on=['sent', 'WorkerId'], suffixes=('', '_sentence'))
    print(df)
    # Filter out sentences with outliers
    df_filtered = df[~df['Outlier_iqr_sentence']]

    # Aggregate reading times for sentences without outliers
    sent_rt= df_filtered.groupby(['sent', 'WorkerId'])['RT'].sum().reset_index()
    sent_rt.rename({'RT':'TotalRT'},axis=1,inplace=True)

    return sent_rt
def filter_z(df):
    from scipy.stats import zscore
    z_scores = zscore(np.log(df['RT']))
    abs_z_scores = np.abs(z_scores)
    df.loc[:, 'Outlier'] = False#abs_z_scores > 3
    print("Percentage of outliers:", sum(df['Outlier']) / len(df))
    # Check if all workers have the same number of word in the sentence
    # (otherwise the z-score would be biased)
    wc=df.groupby(['sent', 'WorkerId'])['word'].count().reset_index()
    # Rename the column for clarity
    wc.rename(columns={'word': 'wordCount'}, inplace=True)
    merged_df = df.merge(wc, on=['sent', 'WorkerId'])

    # Filter rows where word count matches the sentence length
    df = merged_df[merged_df['wordCount'] == merged_df['sent_len']]

   # for sentence, group in wc.groupby('sent'):

    #    for i, w in group.iterrows():
     #       if w['wordCount'] != df[df['sent'] == sentence].iloc[0]['sent_len']:
              #  print(f"Error: Sentence {sentence} has varying word counts across workers: {group}")
      #          df = df[~((df['sent'] == sentence) & (df['WorkerId'] == w['WorkerId']))]
    # Group by sent and WorkerId to find sentences with outliers
    sentence_outliers = df.groupby(['sent', 'WorkerId'])['Outlier'].any().reset_index()
    # Merge this information back into the original DataFrame
    df = df.merge(sentence_outliers, on=['sent', 'WorkerId'], suffixes=('', '_sentence'))
    # Filter out sentences with outliers
    df_filtered = df[~df['Outlier_sentence']]
    # Aggregate reading times for sentences without outliers
    sent_rt= df_filtered.groupby(['sent', 'WorkerId'])['RT'].sum().reset_index()
    #sent_rt= df_filtered
    #sent_rt['TotalRT']=total_rt
    sent_rt['sent_chlen']=df_filtered.groupby(['sent', 'WorkerId'])['chlen'].sum().reset_index()['chlen']
    sent_rt.rename({'RT':'TotalRT'},axis=1,inplace=True)
    #sent_rt['meanItemRT'] = sent_rt.groupby(['zone','item'])['RT'].transform('mean')
    #sent_rt['gmeanItemRT'] = sent_rt.groupby(['zone','item'])['RT'].transform(gmean)
    #sent_rt['meanUnigramLogProb'] = sent_rt.groupby(['zone','item'])['unigram_log_prob'].transform('mean')
    #sent_rt['logmeanItemRT'] = np.log(sent_rt['meanItemRT'])
    sent_rt['logTotalRT']=np.log(sent_rt['TotalRT'])
    return sent_rt



df_cola=read_cola()
print(df_cola)
sents_cola=df_cola['sentence'].tolist()
df_cola=get_surp_cola(sents_cola,df_cola)
print(df_cola)

df_cola_dev=read_cola(filename='cola_public/raw/in_domain_dev.tsv')
sents_cola_dev=df_cola_dev['sentence'].tolist()
df_cola_dev=get_surp_cola(sents_cola_dev,df_cola_dev)

eval_cola(df_cola,dev=df_cola_dev,dataset_name="cola", outdir=outdir)

surps=[]
surps_dev=[]
for s in sents_cola:
    surps.append(score_gpt(s, model, tokenizer)[0])
df_cola=compute_metrics(surps,df_cola)
for s in sents_cola_dev:
    surps_dev.append(score_gpt(s, model, tokenizer)[0])
df_cola_dev=compute_metrics(surps_dev,df_cola_dev)
eval_cola(df_cola,dev=df_cola_dev,dataset_name="cola2", outdir=outdir)



df_brown=read_brown()
df=read_ns()
df_sent = df.drop_duplicates(subset=['zone','item'])
#sents_natural = [df_sent.groupby('sent')['word'].apply(list).tolist() for _, group in df.groupby('item')]
sents_natural = [group.groupby('sent')['word'].apply(lambda words: ' '.join(words)).tolist() for _, group in df_sent.groupby('item')]
stories_natural=list(df_sent.groupby(['item']).agg({'word':lambda x: ' '.join(list(x))})['word'])

df_sent = df_brown.drop_duplicates(subset=['zone','item'])

sents_brown = [group.groupby('sent')['word'].apply(lambda words: ' '.join(words)).tolist() for _, group in df_sent.groupby('item')]
print(sents_brown)
stories_brown=list(df_sent.groupby(['item']).agg({'word':lambda x: ' '.join(list(x))})['word'])
#print(df[~df[['sent', 'WorkerId']].duplicated(keep='first').groupby(['sent'])['word'])
#if sentence level processing
print("Processing sentences?")
print(parser.parse_args().sent_seg)
if parser.parse_args().sent_seg:
    df_brown=get_surp_story2(sents_brown,df_brown)
    df=get_surp_story2(sents_natural,df)
else:
    df_brown=get_surp_story2(stories_brown,df_brown)
    surps_natural,df=get_surp_story(stories_natural,df)

sent_rt_brown = filter_z(df_brown)
sent_rt=filter_z(df)

df = df.merge(sent_rt, on=['sent', 'WorkerId'],
                     suffixes=('', '_updated'), how='left')#.mask(df == '')

df_brown = df_brown.merge(sent_rt_brown, on=['sent', 'WorkerId'],
                      suffixes=('', '_updated'), how='left')#.mask(df == '')


df.to_csv(f'{outdir}/out_natural.tsv', sep='\t')
df_brown.to_csv(f'{outdir}/out_brown.tsv', sep='\t')
eval_word(df,dataset_name="naturalstories", outdir=outdir)
eval_word(df_brown,dataset_name="brown", outdir=outdir)
eval_df(None,df_brown,dataset_name="brown",outdir=outdir)
eval_df(None,df,dataset_name="naturalstories", outdir=outdir)

exit()




print("Processing sentences?")
print(parser.parse_args().sent_seg)

df_cola=read_cola()
print(df_cola)
sents_cola=df_cola['sentence'].tolist()
df_cola=get_surp_cola(sents_cola,df_cola)
print(df_cola)

df_cola_dev=read_cola(filename='cola_public/raw/in_domain_dev.tsv')
sents_cola_dev=df_cola_dev['sentence'].tolist()
df_cola_dev=get_surp_cola(sents_cola_dev,df_cola_dev)
eval_cola(df_cola,dev=df_cola_dev,dataset_name="cola", outdir=outdir)

surps=[]
surps_dev=[]
for s in sents_cola:
    surps.append(score_gpt(s, model, tokenizer)[0])
df_cola=compute_metrics(surps,df_cola)
for s in sents_cola_dev:
    surps_dev.append(score_gpt(s, model, tokenizer)[0])
df_cola_dev=compute_metrics(surps_dev,df_cola_dev)
eval_cola(df_cola,dev=df_cola_dev,dataset_name="cola2", outdir=outdir)









df_brown=read_brown()
df_brown.to_csv('/home/enki/models/marian/perplex/out_brown1.tsv', sep='\t')
df=read_ns()
df_sent = df.drop_duplicates(subset=['zone','item'])
#sents_natural = [df_sent.groupby('sent')['word'].apply(list).tolist() for _, group in df.groupby('item')]
sents_natural = [group.groupby('sent')['word'].apply(lambda words: ' '.join(words)).tolist() for _, group in df_sent.groupby('item')]
stories_natural=list(df_sent.groupby(['item']).agg({'word':lambda x: ' '.join(list(x))})['word'])

df_sent = df_brown.drop_duplicates(subset=['zone','item'])

sents_brown = [group.groupby('sent')['word'].apply(lambda words: ' '.join(words)).tolist() for _, group in df_sent.groupby('item')]
print(sents_brown)
stories_brown=list(df_sent.groupby(['item']).agg({'word':lambda x: ' '.join(list(x))})['word'])
#print(df[~df[['sent', 'WorkerId']].duplicated(keep='first').groupby(['sent'])['word'])
#if sentence level processing
print("Processing sentences?")
print(parser.parse_args().sent_seg)
if parser.parse_args().sent_seg:
    df_brown=get_surp_story2(sents_brown,df_brown)
    df=get_surp_story2(sents_natural,df)
else:
    df_brown=get_surp_story2(stories_brown,df_brown)
    surps_natural,df=get_surp_story(stories_natural,df)

sent_rt_brown = filter_z(df_brown)
sent_rt=filter_z(df)

df = df.merge(sent_rt, on=['sent', 'WorkerId'],
                     suffixes=('', '_updated'), how='left')#.mask(df == '')

df_brown = df_brown.merge(sent_rt_brown, on=['sent', 'WorkerId'],
                      suffixes=('', '_updated'), how='left')#.mask(df == '')


df.to_csv(f'{outdir}/out_natural.tsv', sep='\t')
df_brown.to_csv(f'{outdir}/out_brown.tsv', sep='\t')
eval_word(df,dataset_name="naturalstories", outdir=outdir)
eval_word(df_brown,dataset_name="brown", outdir=outdir)
eval_df(None,df_brown,dataset_name="brown",outdir=outdir)
eval_df(None,df,dataset_name="naturalstories", outdir=outdir)

#eval_sent(sent_rts,surps_sent)





sns_plot = sns.regplot(x="surp", y="meanItemRT", data=unique_data, fit_reg=True)
fig = sns_plot.get_figure()
fig.savefig("output.png")
eval(rts,surps)

unique_data['surp']=np.random.permutation(unique_data['surp'].values)
unique_data['len']=np.random.permutation(unique_data['len'].values)
eval(rts,surps)

