'''
Created on Jun. 1, 2023

@author: Yihao Fang
'''
import os
root_path = '.'
os.environ["DSP_CACHEDIR"] = os.path.join(root_path, 'cache')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import numpy as np
import dsp
seed = 42
np.random.seed(seed)
random.seed(seed)
dsp.settings.branch_idx=seed

import sys
sys.setrecursionlimit(10000000)

from dotenv import load_dotenv
load_dotenv()

from dsp.utils.metrics import EM, F1
#from dsp.evaluation.utils import evaluate
import gzip
import time
import csv
import ijson
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from metrics import OpenSQuADEM, OpenSQuADF1, HotPotEM, HotPotF1, QReCCF1, QReCCnF1, FELMF1, FELMBalAcc, FELMMetric, WysdomEM, WysdomF1, ElapsedTime
from pipelines import Vanilla_LM_QA, Retrieve_then_Read_SC_QA, Multihop_QA, DSP_QA, GoT_QA, ReAct
from judges import nli_electoral_college
import matplotlib.ticker as ticker
from functools import partial
from utils import df_to_dsp, df_to_dsp_augmented

verbose = False
if verbose:
    from nli import _t5_nli_logged as _t5_nli
    from nli import _gpt_nli_logged as _gpt_nli
else:
    from nli import _t5_nli, _gpt_nli


def load_jsonl(filename, q_attr_name, seg_attr_name, lbl_attr_name, dm_attr_name):
    df_list = []
    with open(filename, 'r') as f:
        for jl in f:
            r = json.loads(jl)
            df_list.append((r[q_attr_name], r[seg_attr_name], r[lbl_attr_name], r[dm_attr_name]))
    df = pd.DataFrame(df_list, columns =['Question', 'Segments', 'Labels', 'Domain'])
    return df

def load_json_gzip(filename, q_attr_name, a_attr_name):
    with gzip.open(filename, 'rb') as f:
        rows = ijson.items(f, 'item')
        df = pd.DataFrame(list(((r[q_attr_name], r[a_attr_name]) for r in rows)), columns =['Question', 'Answer'])
    return df

def load_json(filename, q_attr_name, a_attr_name, l_attr_name = None, cn_attr_name = None, tn_attr_name = None, cx_attr_name = None, dm_attr_name = None):
    with open(filename, 'r') as f:
        rows = ijson.items(f, 'item')
        if l_attr_name:
            df = pd.DataFrame(list(((r[q_attr_name], r[a_attr_name], r[l_attr_name]) for r in rows)), columns =['Question', 'Answer', 'Level'])
        elif cn_attr_name and tn_attr_name and cx_attr_name:
            df = pd.DataFrame(list(((r[q_attr_name], r[a_attr_name], r[cn_attr_name], r[tn_attr_name], r[cx_attr_name]) for r in rows)), columns =['Question', 'Answer', 'Conversation#', 'Turn#', 'Context'])
        #elif dm_attr_name:
        #    df = pd.DataFrame(list(((r[q_attr_name], r[a_attr_name], r[dm_attr_name]) for r in rows)), columns =['Question', 'Answer', 'Domain'])
        else:
            df = pd.DataFrame(list(((r[q_attr_name], r[a_attr_name]) for r in rows)), columns =['Question', 'Answer'])
    return df

def load_tsv(filename):
    with open(filename) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        df = pd.DataFrame(list(reader), columns =['Question', 'Answer'])
    return df

def sent_len(sentences):
    return [len(word_tokenize(sent)) for sent in sentences]

def select_long_questions(df, proportion):
    df["Question Length"] = sent_len(df["Question"])
    len_threshold = df["Question Length"].quantile([1-proportion]).values[0]
    long_df = df[df["Question Length"] > len_threshold]
    
    if len(long_df)>round(len(df)*proportion):
        long_df = long_df.sample(n=round(len(df)*proportion), random_state=seed)
    return long_df

def select_short_questions(df, proportion):
    df["Question Length"] = sent_len(df["Question"])
    len_threshold = df["Question Length"].quantile([proportion]).values[0]
    short_df = df[df["Question Length"] <= len_threshold]
    
    if len(short_df)>round(len(df)*proportion):
        short_df = short_df.sample(n=round(len(df)*proportion), random_state=seed)
    #short_df = short_df.sample(n=round(len(df)*0.02*0.2), random_state=seed)
    return short_df

def select_medium_questions(df, proportion):
    df["Question Length"] = sent_len(df["Question"])
    len_lower_threshold, len_upper_threshold = df["Question Length"].quantile([proportion, 1-proportion]).values
    medium_df = df[(df["Question Length"] > len_lower_threshold) & (df["Question Length"] <= len_upper_threshold)]
    #medium_df = medium_df.sample(frac=(proportion/(1-2*proportion)), random_state=seed)
    if len(medium_df)>round(len(df)*proportion):
        medium_df = medium_df.sample(n=round(len(df)*proportion), random_state=seed)
    
    #medium_df = medium_df.sample(n=round(len(df)*0.02*0.2), random_state=seed)
    return medium_df

def sample_n_save_data_by_sent_len(dataset, sent_len, train_df, dev_df, test_df, proportion = 0.02, transform={"train":lambda x: x, "dev":lambda x: x, "test":lambda x: x}):
    dataset_dict = {"open-squad": "Open-SQuAD", "hotpotqa": "HotPotQA", "qrecc": "QReCC", "felm": "FELM", "wysdomqa": "WysdomQA"}
    
    if sent_len == "long":
        train_df, dev_df, test_df = select_long_questions(train_df, proportion = proportion), select_long_questions(dev_df, proportion = proportion), select_long_questions(test_df, proportion = proportion)
    elif sent_len == "medium":
        train_df, dev_df, test_df = select_medium_questions(train_df, proportion = proportion), select_medium_questions(dev_df, proportion = proportion), select_medium_questions(test_df, proportion = proportion)
    elif sent_len == "short":
        train_df, dev_df, test_df = select_short_questions(train_df, proportion = proportion), select_short_questions(dev_df, proportion = proportion), select_short_questions(test_df, proportion = proportion)
    else:
        raise NotImplementedError()
    
    train_df, dev_df, test_df = transform["train"](train_df), transform["dev"](dev_df), transform["test"](test_df)
    
    train_df.to_csv("data/%s/train_%s.csv"%(dataset_dict[dataset], sent_len), index=False)
    dev_df.to_csv("data/%s/dev_%s.csv"%(dataset_dict[dataset], sent_len), index=False)
    test_df.to_csv("data/%s/test_%s.csv"%(dataset_dict[dataset], sent_len), index=False)
    
    return train_df, dev_df, test_df

def preprocess_data(dataset):
    if dataset == "open-squad":
        train_df = load_json_gzip("data/Open-SQuAD/biencoder-squad1-train.json.gz", q_attr_name = 'question', a_attr_name = 'answers')
        dev_df = load_json_gzip("data/Open-SQuAD/biencoder-squad1-dev.json.gz", q_attr_name = 'question', a_attr_name = 'answers')
        test_df = load_tsv("data/Open-SQuAD/squad1-test.qa.csv")
        
        train_df.to_csv("data/Open-SQuAD/train.csv", index=False)
        dev_df.to_csv("data/Open-SQuAD/dev.csv", index=False)
        test_df.to_csv("data/Open-SQuAD/test.csv", index=False)
        
        for sent_len in ["long", "medium", "short"]:
            sample_n_save_data_by_sent_len(dataset, sent_len, train_df, dev_df, test_df, proportion=0.015)

    elif dataset == "hotpotqa":
        train_dev_df = load_json("data/HotPotQA/hotpot_train_v1.1.json", q_attr_name = 'question', a_attr_name = 'answer', l_attr_name = 'level')
        train_df = train_dev_df.sample(frac = 0.9, random_state=seed)
        dev_df = train_dev_df.drop(train_df.index)
        test_df = load_json("data/HotPotQA/hotpot_dev_fullwiki_v1.json", q_attr_name = 'question', a_attr_name = 'answer', l_attr_name = 'level')
        
        train_df.to_csv("data/HotPotQA/train.csv", index=False)
        dev_df.to_csv("data/HotPotQA/dev.csv", index=False)
        test_df.to_csv("data/HotPotQA/test.csv", index=False)
        
        for sent_len in ["long", "medium", "short"]:
            sample_n_save_data_by_sent_len(dataset, sent_len, train_df, dev_df, test_df)
        
    elif dataset == "qrecc":
        def clean(df):
            #print("df:", len(df))
            # remove conversations that have only one or two questions
            sf = df.groupby(['Conversation#']).size()
            gy_df = pd.DataFrame({'Conversation#':sf.index, 'Count':sf.values})
            gy_df = gy_df[gy_df['Count'] >= 3]
            df = df[df['Conversation#'].isin(gy_df['Conversation#'].values)]
            #print("df:", len(df))
            
            # remove conversations that have one or more empty ground truth answers
            df["Answer Length"] = df.apply(lambda x: len(x.Answer.strip()), axis=1)
            lq_df = df[df["Answer Length"]==0]
            df = df[~df['Conversation#'].isin(lq_df['Conversation#'].values)]
            df = df.drop('Answer Length', axis=1)
            #print("df:", len(df))
            
            # remove any conversation that includes the keywords "other interesting" or "else"
            df['Low Quality'] = df.apply(lambda x: "other interesting" in x.Question or "else" in x.Question, axis=1)
            lq_df = df[df['Low Quality']==True]
            df = df[~df['Conversation#'].isin(lq_df['Conversation#'].values)]
            df = df.drop('Low Quality', axis=1)
            #print("df:", len(df))
            return df
        
        train_dev_df = load_json("data/QReCC/qrecc_train.json", q_attr_name = 'Rewrite', a_attr_name = 'Answer', cn_attr_name = 'Conversation_no', tn_attr_name = 'Turn_no', cx_attr_name = 'Context')
        train_dev_df = clean(train_dev_df)
        
        train_df = train_dev_df.sample(frac = 0.9, random_state=seed)
        dev_df = train_dev_df.drop(train_df.index)
        test_df = load_json("data/QReCC/qrecc_test.json", q_attr_name = 'Rewrite', a_attr_name = 'Answer', cn_attr_name = 'Conversation_no', tn_attr_name = 'Turn_no', cx_attr_name = 'Context')
        test_df = clean(test_df)
        
        train_df.to_csv("data/QReCC/train.csv", index=False)
        dev_df.to_csv("data/QReCC/dev.csv", index=False)
        test_df.to_csv("data/QReCC/test.csv", index=False)
        
        for sent_len in ["long", "medium", "short"]:
            sample_n_save_data_by_sent_len(dataset, sent_len, train_df, dev_df, test_df)
    
    elif dataset == "felm":
        train_dev_test_df = load_jsonl("data/FELM/all.jsonl", q_attr_name = 'prompt', seg_attr_name = 'segmented_response', lbl_attr_name = 'labels', dm_attr_name = 'domain')
        train_dev_test_df['Question'] = train_dev_test_df.apply(
            lambda x: 'Is the query "' + x['Question'].rstrip('?') 
            + '" accurately addressed by the response (which has been split into a list of text segments) "' 
            + ' '.join(['%d. %s'%(i+1, segment) for i, segment in enumerate(x['Segments'])])
            + '" ? List the IDs of the segments with errors (separated by commas). If all the segments are correct, output "ALL_CORRECT".', axis=1)
            #+ '" ? List the IDs of the segments with errors if any (separated by commas).', axis=1)
            
        train_dev_test_df['Answer'] = train_dev_test_df.apply(lambda x: "ALL_CORRECT" if all(x['Labels']) else ",".join([str(i+1) for i, label in enumerate(x['Labels']) if not label]), axis=1)
        #train_dev_test_df['Answer'] = train_dev_test_df['Labels']
        
        # transform to a balanced dataset
        def transform(df, granularity="segment"):
            if granularity=="response":
                cor_len = len(df[df['Answer'] == "ALL_CORRECT"].index)
                incor_len = len(df[df['Answer'] != "ALL_CORRECT"].index)
                if cor_len > incor_len:
                    df = df.drop(df[df['Answer'] == "ALL_CORRECT"].sample(n=cor_len-incor_len).index)
                else:
                    df = df.drop(df[df['Answer'] != "ALL_CORRECT"].sample(n=incor_len-cor_len).index)
                return df
            elif granularity=="segment":
                cor_len = len(df[df['Answer'] == "ALL_CORRECT"].index)
                incor_len = len(df[df['Answer'] != "ALL_CORRECT"].index)
                sum_cor_t_count = np.sum([np.sum(np.array(labels).astype(int)) for labels in df[df['Answer'] == "ALL_CORRECT"]['Labels'].values])
                sum_cor_n_count = np.sum([np.sum(np.logical_not(labels).astype(int)) for labels in df[df['Answer'] == "ALL_CORRECT"]['Labels'].values])
                sum_incor_t_count = np.sum([np.sum(np.array(labels).astype(int)) for labels in df[df['Answer'] != "ALL_CORRECT"]['Labels'].values])
                sum_incor_n_count = np.sum([np.sum(np.logical_not(labels).astype(int)) for labels in df[df['Answer'] != "ALL_CORRECT"]['Labels'].values])
                
                if (sum_cor_t_count + sum_incor_t_count) > (sum_cor_n_count + sum_incor_n_count):
                    mean_cor_t_count = sum_cor_t_count / cor_len
                    cor_drop_count = int(((sum_cor_t_count + sum_incor_t_count) - (sum_cor_n_count + sum_incor_n_count))/mean_cor_t_count)
                    cor_drop_count = min(cor_drop_count, cor_len-1)
                    df = df.drop(df[df['Answer'] == "ALL_CORRECT"].sample(n=cor_drop_count).index)
                else:
                    mean_incor_n_count = sum_incor_n_count / incor_len
                    mean_incor_t_count = sum_incor_t_count / incor_len
                    incor_drop_count = int(((sum_cor_n_count + sum_incor_n_count)-(sum_cor_t_count + sum_incor_t_count))/(mean_incor_n_count - mean_incor_t_count))
                    incor_drop_count = min(incor_drop_count, incor_len-1)
                    df = df.drop(df[df['Answer'] != "ALL_CORRECT"].sample(n=incor_drop_count).index)
                return df
            else:
                raise NotImplementedError()
        
        train_df = train_dev_test_df.sample(frac = 0.4, random_state=seed)
        dev_test_df = train_dev_test_df.drop(train_df.index)
        dev_df = dev_test_df.sample(frac = 0.1/0.6, random_state=seed)
        test_df = dev_test_df.drop(dev_df.index)
        
        train_df.to_csv("data/FELM/train.csv", index=False)
        dev_df.to_csv("data/FELM/dev.csv", index=False)
        test_df.to_csv("data/FELM/test.csv", index=False)
        
        for sent_len in ["long", "medium", "short"]:
            sample_n_save_data_by_sent_len(dataset, sent_len, train_df, dev_df, test_df, proportion=0.25, transform={"train": transform, "dev": transform, "test": transform})
        
    elif dataset == "wysdomqa":
        train_dev_test_df = pd.read_csv("data/WysdomQA/WysdomQA.csv")
        train_dev_test_df['Question'] = train_dev_test_df.apply(lambda x: 'Is the query "' + x['Question'].rstrip('?') + '" accurately addressed by the response "' + x['Generative Answer'] + '" ? Kindly answer with either "Yes" or "No".', axis=1)
        train_dev_test_df['Answer'] = train_dev_test_df['Correct Answer']
        train_dev_test_df.drop('Generative Answer', axis=1, inplace=True)
        train_dev_test_df.drop('Correct Answer', axis=1, inplace=True)
        
        train_df = train_dev_test_df.sample(frac = 0.18, random_state=seed)
        dev_test_df = train_dev_test_df.drop(train_df.index)
        
        dev_df = dev_test_df.sample(frac = 0.0/0.82, random_state=seed)
        test_df = dev_test_df.drop(dev_df.index)
        
        train_df.to_csv("data/WysdomQA/train.csv", index=False)
        dev_df.to_csv("data/WysdomQA/dev.csv", index=False)
        test_df.to_csv("data/WysdomQA/test.csv", index=False)
        
        train_df.to_csv("data/WysdomQA/train_medium.csv", index=False)
        dev_df.to_csv("data/WysdomQA/dev_medium.csv", index=False)
        test_df.to_csv("data/WysdomQA/test_medium.csv", index=False)
    else:
        raise NotImplementedError()

def load_data(dataset):
    if dataset == "open-squad":
        train_df = pd.read_csv("data/Open-SQuAD/train.csv")
        dev_df = pd.read_csv("data/Open-SQuAD/dev.csv")
        test_df = pd.read_csv("data/Open-SQuAD/test.csv")
    elif dataset == "open-squad-long":
        train_df = pd.read_csv("data/Open-SQuAD/train_long.csv")
        dev_df = pd.read_csv("data/Open-SQuAD/dev_long.csv")
        test_df = pd.read_csv("data/Open-SQuAD/test_long.csv")
    elif dataset == "open-squad-medium":
        train_df = pd.read_csv("data/Open-SQuAD/train_medium.csv")
        dev_df = pd.read_csv("data/Open-SQuAD/dev_medium.csv")
        test_df = pd.read_csv("data/Open-SQuAD/test_medium.csv")
    elif dataset == "open-squad-short":
        train_df = pd.read_csv("data/Open-SQuAD/train_short.csv")
        dev_df = pd.read_csv("data/Open-SQuAD/dev_short.csv")
        test_df = pd.read_csv("data/Open-SQuAD/test_short.csv")
    elif dataset == "hotpotqa":
        train_df = pd.read_csv("data/HotPotQA/train.csv")
        dev_df = pd.read_csv("data/HotPotQA/dev.csv")
        test_df = pd.read_csv("data/HotPotQA/test.csv")
    elif dataset == "hotpotqa-long":
        train_df = pd.read_csv("data/HotPotQA/train_long.csv")
        dev_df = pd.read_csv("data/HotPotQA/dev_long.csv")
        test_df = pd.read_csv("data/HotPotQA/test_long.csv")
    elif dataset == "hotpotqa-medium":
        train_df = pd.read_csv("data/HotPotQA/train_medium.csv")
        dev_df = pd.read_csv("data/HotPotQA/dev_medium.csv")
        test_df = pd.read_csv("data/HotPotQA/test_medium.csv")
    elif dataset == "hotpotqa-short":
        train_df = pd.read_csv("data/HotPotQA/train_short.csv")
        dev_df = pd.read_csv("data/HotPotQA/dev_short.csv")
        test_df = pd.read_csv("data/HotPotQA/test_short.csv")
    elif dataset == "qrecc":    
        train_df = pd.read_csv("data/QReCC/train.csv")
        dev_df = pd.read_csv("data/QReCC/dev.csv")
        test_df = pd.read_csv("data/QReCC/test.csv")
    elif dataset == "qrecc-long":    
        train_df = pd.read_csv("data/QReCC/train_long.csv")
        dev_df = pd.read_csv("data/QReCC/dev_long.csv")
        test_df = pd.read_csv("data/QReCC/test_long.csv")
    elif dataset == "qrecc-medium":    
        train_df = pd.read_csv("data/QReCC/train_medium.csv")
        dev_df = pd.read_csv("data/QReCC/dev_medium.csv")
        test_df = pd.read_csv("data/QReCC/test_medium.csv")
    elif dataset == "qrecc-short":    
        train_df = pd.read_csv("data/QReCC/train_short.csv")
        dev_df = pd.read_csv("data/QReCC/dev_short.csv")
        test_df = pd.read_csv("data/QReCC/test_short.csv")
    elif dataset == "felm-long":    
        train_df = pd.read_csv("data/FELM/train_long.csv")
        dev_df = pd.read_csv("data/FELM/dev_long.csv")
        test_df = pd.read_csv("data/FELM/test_long.csv")
    elif dataset == "felm-medium":    
        train_df = pd.read_csv("data/FELM/train_medium.csv")
        dev_df = pd.read_csv("data/FELM/dev_medium.csv")
        test_df = pd.read_csv("data/FELM/test_medium.csv")
    elif dataset == "felm-short":    
        train_df = pd.read_csv("data/FELM/train_short.csv")
        dev_df = pd.read_csv("data/FELM/dev_short.csv")
        test_df = pd.read_csv("data/FELM/test_short.csv")
    elif dataset == "wysdomqa":    
        train_df = pd.read_csv("data/WysdomQA/train.csv")
        dev_df = pd.read_csv("data/WysdomQA/dev.csv")
        test_df = pd.read_csv("data/WysdomQA/test.csv")
    elif dataset == "wysdomqa-medium":    
        train_df = pd.read_csv("data/WysdomQA/train_medium.csv")
        dev_df = pd.read_csv("data/WysdomQA/dev_medium.csv")
        test_df = pd.read_csv("data/WysdomQA/test_medium.csv")
    else:
        raise NotImplementedError()
    return [train_df, dev_df, test_df]


def analyze_data(df_dict):
    dataset_dict = {"open-squad":"Open-SQuAD", "hotpotqa":"HotPotQA", "qrecc":"QReCC", "felm":"FELM", "wysdomqa":"WysdomQA"}
    
    for dataset in dataset_dict:
        train_len_df = pd.DataFrame(sent_len(df_dict[dataset][0]["Question"].values), columns=["Train"])
        dev_len_df = pd.DataFrame(sent_len(df_dict[dataset][1]["Question"].values), columns=["Dev"])
        test_len_df = pd.DataFrame(sent_len(df_dict[dataset][2]["Question"].values), columns=["Test"])
        df = pd.concat([train_len_df, dev_len_df, test_len_df], join = 'outer', axis = 1)
        
        fig, ax = plt.subplots(figsize=(3,3))
        sns.histplot(df, element='step', fill=True, kde=False, alpha=0.5, shrink=0.8, multiple='dodge', bins=20)
        
        ax.legend_.set_title('') # remove the legend title (the name of the hue column)
        ax.margins(x=0.01) # less spacing
        #ax.set_title(dataset_dict[dataset])
        ax.set_xlabel("Sentence Length")
        plt.tight_layout()
        plt.draw()
        #plt.show()
        plt.savefig("log/%s_question_length.png"%dataset)
        plt.clf()
        plt.close() 
    
    
    #fig = plt.figure(figsize=(6,2))
    fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(5,2))
    
    for i, dataset in enumerate(dataset_dict):
        train_len_df = pd.DataFrame(sent_len(df_dict[dataset][0]["Question"].values), columns=["Train"])
        dev_len_df = pd.DataFrame(sent_len(df_dict[dataset][1]["Question"].values), columns=["Dev"])
        test_len_df = pd.DataFrame(sent_len(df_dict[dataset][2]["Question"].values), columns=["Test"])
        df = pd.concat([train_len_df, dev_len_df, test_len_df], join = 'outer', axis = 1)
        
        #plt.subplot(1, 3, i+1)
        ax = axes[i]
        #ax = plt.gca()
        sns.histplot(df, element='step', fill=True, kde=False, alpha=0.5, shrink=0.8, multiple='dodge', bins=20, ax=ax, log_scale=False)
        
        
        ax.margins(x=0.01) # less spacing
        ax.set_title("SQuAD" if dataset == "open-squad" else dataset_dict[dataset])
        ax.set_xlabel("Sentence Length")
        #plt.yscale('log')
        ax.set_yscale('log')
        ax.set_yticks([10,1000])
        #formatter = ticker.FuncFormatter(lambda x, pos: "%.0fK"%(x/1000) if x > 0 else "0")
        #ax.yaxis.set_major_formatter(formatter)
        
        if i < len(dataset_dict) - 1:
            ax.get_legend().remove()
        else:
            handles = ax.get_legend().legendHandles
            ax.get_legend().remove()
    
    fig.tight_layout() 
    fig.subplots_adjust(top=0.7)   ##  Need to play with this number.
    fig.legend(handles, ['Train', 'Dev', 'Test'], loc='upper center', ncol=3)
    #plt.tight_layout()
    
    plt.draw()
    plt.savefig("log/question_length.png")
    plt.clf()
    plt.close() 



def retrieve_demos(dataset, segments=["plan", "rewrite", "rationale"]):
    demos = {}
    if dataset == "open-squad-long":
        if "plan" in segments:
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_long_augmented.csv",keep_default_na=False), segment="plan")[:2]
        if "rewrite" in segments:
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_long_augmented.csv",keep_default_na=False), segment="rewrite")[1:2]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_long_augmented.csv",keep_default_na=False), segment="rationale")[:2]
    elif dataset == "open-squad-medium":
        if "plan" in segments:
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_medium_augmented.csv",keep_default_na=False), segment="plan")[:2]
        if "rewrite" in segments:
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_medium_augmented.csv",keep_default_na=False), segment="rewrite")[0:1]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_medium_augmented.csv",keep_default_na=False), segment="rationale")[:2]
    elif dataset == "open-squad-short":
        if "plan" in segments:
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_short_augmented.csv",keep_default_na=False), segment="plan")[:2]
        if "rewrite" in segments:
            # demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_short_augmented.csv",keep_default_na=False))[0:1]
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="rewrite")[4:5]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_short_augmented.csv",keep_default_na=False), segment="rationale")[:2]
    elif dataset == "hotpotqa-long":
        if "plan" in segments:
            #demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_long_augmented.csv",keep_default_na=False), segment="plan")[:2]
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="plan")[:2]
        if "rewrite" in segments:
            # demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_long_augmented.csv",keep_default_na=False))[0:1]
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="rewrite")[4:5]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="rationale")[5:7]
    elif dataset == "hotpotqa-medium":
        if "plan" in segments:
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_medium_augmented.csv",keep_default_na=False), segment="plan")[:2]
        if "rewrite" in segments:
            # demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_medium_augmented.csv",keep_default_na=False))[5:6] 
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="rewrite")[4:5]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_medium_augmented.csv",keep_default_na=False), segment="rationale")[:2]
    elif dataset == "hotpotqa-short":
        if "plan" in segments:
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_short_augmented.csv",keep_default_na=False), segment="plan")[3:5]
        if "rewrite" in segments:
            # demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_short_augmented.csv",keep_default_na=False))[5:6] 
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="rewrite")[4:5]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="rationale")[5:7]
    elif dataset == "qrecc-long":
        if "plan" in segments:
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_long_augmented.csv",keep_default_na=False), segment="plan")[2:4]
        if "rewrite" in segments:
            # demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_long_augmented.csv",keep_default_na=False))[5:6]
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="rewrite")[4:5]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_long_augmented.csv",keep_default_na=False), segment="rationale")[2:4]
    elif dataset == "qrecc-medium":
        if "plan" in segments:
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_medium_augmented.csv",keep_default_na=False), segment="plan")[3:5]
        if "rewrite" in segments:
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_medium_augmented.csv",keep_default_na=False), segment="rewrite")[4:5]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_medium_augmented.csv",keep_default_na=False), segment="rationale")[3:5]
    elif dataset == "qrecc-short":    
        if "plan" in segments:
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_short_augmented.csv",keep_default_na=False), segment="plan")[:2]
        if "rewrite" in segments:
            # demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_short_augmented.csv",keep_default_na=False))[5:6]
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="rewrite")[4:5]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_short_augmented.csv",keep_default_na=False), segment="rationale")[:2]
    elif dataset == "wysdomqa-medium":
        if "plan" in segments:
            demos["plan"] = df_to_dsp_augmented(pd.read_csv("data/WysdomQA/train_medium_augmented.csv",keep_default_na=False), segment="plan")[:2]
        if "rewrite" in segments:
            demos["rewrite"] = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False), segment="rewrite")[4:5]
        if "rationale" in segments:
            demos["rationale"] = df_to_dsp_augmented(pd.read_csv("data/WysdomQA/train_medium_augmented.csv",keep_default_na=False), segment="rationale")[:2]
    else:
        raise NotImplementedError()
    return demos


def init_langauge_model(language_model='gpt-3.5-turbo-1106'):
    
    #language_model='gpt-3.5-turbo'
    #language_model='gpt-3.5-turbo-1106'
    #language_model='gpt-4'
    #language_model='gpt-4-1106-preview'
    
    openai_key = os.getenv('OPENAI_API_KEY')  # or replace with your API key (optional)
    
    if language_model=='text-davinci-002':
        lm = dsp.GPT(model=language_model, api_key=openai_key)
    else:
        lm = dsp.GPT(model=language_model, api_key=openai_key, model_type="chat")
    
    dsp.settings.configure(lm=lm)
    dsp.settings.configure(vectorizer=dsp.SentenceTransformersVectorizer())
    dsp.settings.lm.kwargs["max_tokens"] = 300
    

def init_retrieval_model(retrieval_model='google'):
    
    serpapi_key = os.getenv('SERPAPI_API_KEY')  # or replace with your API key (optional)
    
    if retrieval_model=='google':
        rm = dsp.Google(serpapi_key)
    else:
        #colbert_server = 'http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search'
        colbert_server = 'http://192.168.3.200:8893/api/search'
        rm = dsp.ColBERTv2(url=colbert_server)
    
    dsp.settings.configure(rm=rm)


def evaluate(method, dataset):
    
    if method == "react":
        language_model='text-davinci-002'
    else:
        language_model='gpt-3.5-turbo-1106'
    retrieval_model='google'

    init_langauge_model(language_model=language_model)
    init_retrieval_model(retrieval_model=retrieval_model)
    
    if method == "react":
        dsp.settings.lm.kwargs["max_tokens"] = 100
        dsp.settings.lm.kwargs["stop"] = ("\n",)
    
    default_stdout = sys.stdout
    log_file = open("log/%s_%s_%s_%s.log"%(dataset, method, language_model, retrieval_model),"w")
    sys.stdout = log_file
    annot_dump = "log/%s_%s_%s_%s_annot.csv"%(dataset, method, language_model, retrieval_model)
    annot_log = "log/%s_%s_%s_%s_annot.log"%(dataset, method, language_model, retrieval_model)
    
    train, dev, test = load_data(dataset)

    train, dev, test = df_to_dsp(train), df_to_dsp(dev), df_to_dsp(test)
    
    
    def _annot_selector():
        if "open-squad" in dataset:
            return EM
        elif "hotpotqa" in dataset:
            return EM
        elif "qrecc" in dataset:
            return lambda prediction, answer: F1(prediction,answer) > 0.7
        elif "felm" in dataset:
            def felm_annot_selector(prediction, answer):
                threshold = 0.5
                felm_metric = FELMMetric()
                result = felm_metric.evaluate(prediction,answer)
                metric = felm_metric.consolidate(result)
                if metric['true num'] != 0 and metric['false num'] != 0:
                    return metric['balanced'] > threshold
                elif metric['true num'] != 0 and metric['false num'] == 0:
                    return metric['class 1'] > threshold
                elif metric['true num'] == 0 and metric['false num'] != 0:
                    return metric['class 0'] > threshold
                else:
                    return False
            return felm_annot_selector
        elif "wysdomqa" in dataset:
            return EM
        else:
            raise NotImplementedError()
    
    if method == "vanilla":
        dsp.settings.configure(electoral_college=None)
        method_func = Vanilla_LM_QA()
    elif method == "retrieve_then_read_sc":
        dsp.settings.configure(electoral_college=None)
        method_func = Retrieve_then_Read_SC_QA()
    elif method == "multihop":
        dsp.settings.configure(electoral_college=None)
        method_func = Multihop_QA()
    elif method == "react":
        dsp.settings.configure(electoral_college=None)
        method_func = ReAct()
    elif method == "dsp+sample":
        dsp.settings.configure(electoral_college=None)
        method_func = DSP_QA(dsp.sample)
    elif method == "dsp+sample+t5-nli-ec":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = DSP_QA(dsp.sample)
    elif method == "dsp+sample+gpt-nli-ec":
        dsp.settings.configure(nli=_gpt_nli)
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = DSP_QA(dsp.sample)
    elif method == "dsp+knn":
        dsp.settings.configure(electoral_college=None)
        method_func = DSP_QA(dsp.knn(train))
    elif method == "dsp+knn+t5-nli-ec":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = DSP_QA(dsp.knn(train))
    elif method == "dsp+knn+gpt-nli-ec":
        dsp.settings.configure(nli=_gpt_nli)
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = DSP_QA(dsp.knn(train))
    elif method == "got-3":
        dsp.settings.configure(electoral_college=None)
        method_func = GoT_QA(demo_flags=None, demos=None, p_context=False)
    elif method == "got-3+demos":
        dsp.settings.configure(electoral_college=None)
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset, segments=["plan", "rewrite"]), p_context=False)
    elif method == "got-3+demos+t5-nli-ec+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=False, W=[0.3,0.6,0.1])
    elif method == "got-3+demos+t5-nli-ec+ci+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=False, W=[0.3,0.6,0.1])
    elif method == "got-3+demos+gpt-nli-ec+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_gpt_nli)
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=False, W=[0.3,0.6,0.1])
    elif method == "got-3+demos+gpt-nli-ec+ci+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_gpt_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=False, W=[0.3,0.6,0.1])
    elif method == "got-3+demos+cx":
        dsp.settings.configure(electoral_college=None)
        method_func = GoT_QA(demo_flags="plan+rewrite", demos=retrieve_demos(dataset, segments=["plan", "rewrite"]), p_context=True)
    elif method == "got-3+demos+cx+t5-nli-ec+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=True, W=[0.3,0.6,0.1])
    elif method.startswith("got-3+demos+cx+t5-nli-ec+ci"):
        W = eval(method.split('+')[-1])
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=True, W=W)
    elif method.startswith("got-3+demos-sa+cx+t5-nli-ec+ci"):
        W = eval(method.split('+')[-1])
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", p_context=True, W=W, annot_selector=_annot_selector(), annot_dump=annot_dump, annot_log=annot_log)
    elif method.startswith("got-3+demos-sa-knn+cx+t5-nli-ec+ci"):
        W = eval(method.split('+')[-1])
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", p_context=True, W=W, annot_selector=_annot_selector(), annot_dump=annot_dump, annot_log=annot_log, demo_sel_func=dsp.knn)
    elif method == "got-2+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=True, depth=2, W=[0.3,0.6,0.1])
    elif method == "got-4+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=True, depth=4, W=[0.3,0.6,0.1])
    elif method == "got-3+demos+cx+gpt-nli-ec+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_gpt_nli)
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=True, W=[0.3,0.6,0.1])
    elif method == "got-3+demos+cx+gpt-nli-ec+ci+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_gpt_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", demos=retrieve_demos(dataset), p_context=True, W=[0.3,0.6,0.1])
    else:
        raise NotImplementedError()
    
    '''
    elif method == "got-3+demos+cx+t5-nli-ec+ci+[0.35,0.6,0.05]":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demos=retrieve_demos(dataset), p_context=True, W=[0.35,0.6,0.05])
    elif method == "got-3+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demos=retrieve_demos(dataset), p_context=True, W=[0.3,0.6,0.1])
    elif method == "got-3+demos+cx+t5-nli-ec+ci+[0.2,0.6,0.2]":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demos=retrieve_demos(dataset), p_context=True, W=[0.2,0.6,0.2])
    elif method == "got-3+demos+cx+t5-nli-ec+ci+[0.1,0.6,0.3]":
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
        method_func = GoT_QA(demos=retrieve_demos(dataset), p_context=True, W=[0.1,0.6,0.3])
    '''
    #dsp.settings.configure(reranker=partial(nli_reranker, retrieval_weight=0.5, nli_weight=0.5))

    if "open-squad" in dataset:
        metrics = [OpenSQuADEM(), OpenSQuADF1(), ElapsedTime()]
    elif "hotpotqa" in dataset:
        metrics = [HotPotEM(), HotPotF1(), ElapsedTime()]
    elif "qrecc" in dataset:
        metrics = [QReCCF1(), QReCCnF1(), ElapsedTime()]
    elif "felm" in dataset:
        metrics = [FELMF1(), FELMBalAcc(), ElapsedTime()]
    elif "wysdomqa" in dataset:
        metrics = [WysdomEM(), WysdomF1(), ElapsedTime()]
    else:
        raise NotImplementedError()

    inst_by_inst_log = "log/%s_inst_by_inst.csv"%dataset
    if os.path.exists(inst_by_inst_log):
        inst_by_inst_df = pd.read_csv(inst_by_inst_log)
    else:
        inst_by_inst_df = pd.DataFrame([(example.question, example.answer) for example in test]+[(str(metric),"") for metric in metrics], columns=["Question", "GT Answer"])
        
    pred_answers = []
    for example in test: 
        print("#"*10 + example.question + "#"*10)
        start = time.time()
        prediction = method_func(train, example.question)
        end = time.time()
        print("="*35 + " ANSWER " + "="*35)
        print("."*35 + " prediction " + "."*35)
        print(prediction["answer"])
        print("."*35 + " ground truth " + "."*35)
        print(example.answer)
        if "confidence" in prediction:
            print("."*35 + " confidence " + "."*35)
            print(prediction["confidence"])

        for metric in metrics:
            if isinstance(metric, ElapsedTime):
                metric.evaluate(end-start)
            #elif isinstance(metric, CitationRecall) or isinstance(metric, CitationPrecision):
            #    metric.evaluate(prediction["retrieval_history"])
            elif isinstance(metric, QReCCnF1):
                metric.evaluate(prediction["answer"], example.answer, example.history) 
            elif isinstance(metric, FELMF1) or isinstance(metric, FELMBalAcc):
                metric.evaluate(prediction["answer"], example.labels)
            else:
                metric.evaluate(prediction["answer"], example.answer)
        
        pred_answers.append(prediction["answer"])
        
    print("#"*35 + " RESULT " + "#"*35)
    results = [metric.average() for metric in metrics]
    [metric.clear() for metric in metrics]
        
    inst_by_inst_df["%s_%s_%s_%s"%(dataset, method, language_model, retrieval_model)]=pd.Series(pred_answers + results)
    inst_by_inst_df.to_csv(inst_by_inst_log, index=False)
    
    sys.stdout = default_stdout
    log_file.close()
    
def preprocess_n_analyze():
    [preprocess_data(dataset) for dataset in ["open-squad", "hotpotqa", "qrecc", "felm", "wysdomqa"]]
    
    df_dict = {}
    df_dict["open-squad"] = load_data("open-squad")
    df_dict["hotpotqa"] = load_data("hotpotqa")
    df_dict["qrecc"] = load_data("qrecc")
    df_dict["felm"] = load_data("felm")
    df_dict["wysdomqa"] = load_data("wysdomqa")
    analyze_data(df_dict)
    
def main(preprocess = False):
    if preprocess:
        preprocess_n_analyze()
    
    #for method in ["vanilla", "retrieve_then_read_sc", "multihop", "dsp+sample", "dsp+knn", "got-3", "got-3+demos", "got-3+demos+t5-nli-ec+[0.3,0.6,0.1]", "got-3+demos+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-3+demos+cx", "got-3+demos+cx+t5-nli-ec+[0.3,0.6,0.1]", "got-3+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-3+demos+cx+t5-nli-ec+ci+[0.2,0.6,0.2]", "got-3+demos+cx+t5-nli-ec+ci+[0.1,0.6,0.3]", "got-3+demos+cx+t5-nli-ec+ci+[0.35,0.6,0.05]", "got-2+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-4+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]"]:
    #for method in ["got-3", "got-3+demos", "got-3+demos+t5-nli-ec+[0.3,0.6,0.1]", "got-3+demos+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-3+demos+cx", "got-3+demos+cx+t5-nli-ec+[0.3,0.6,0.1]", "got-3+demos+cx+t5-nli-ec+ci+[0.35,0.6,0.05]", "got-3+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-3+demos+cx+t5-nli-ec+ci+[0.2,0.6,0.2]", "got-3+demos+cx+t5-nli-ec+ci+[0.1,0.6,0.3]",  "got-2+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-4+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]"]:
    #for method in ["got-3+demos+cx+t5-nli-ec+ci+[0.35,0.6,0.05]", "got-3+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-3+demos+cx+t5-nli-ec+ci+[0.2,0.6,0.2]", "got-3+demos+cx+t5-nli-ec+ci+[0.1,0.6,0.3]", "got-3+demos+cx+t5-nli-ec+ci+[0.1,0.5,0.4]", "got-3+demos+cx+t5-nli-ec+ci+[0.1,0.4,0.5]", "got-3+demos+cx+t5-nli-ec+ci+[0.05,0.6,0.35]"]:
    #for method in ["got-3+demos-sa+cx+t5-nli-ec+ci+[0.35,0.6,0.05]", "got-3+demos-sa+cx+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-3+demos-sa+cx+t5-nli-ec+ci+[0.2,0.6,0.2]", "got-3+demos-sa+cx+t5-nli-ec+ci+[0.1,0.6,0.3]", "got-3+demos-sa+cx+t5-nli-ec+ci+[0.1,0.5,0.4]", "got-3+demos-sa+cx+t5-nli-ec+ci+[0.1,0.4,0.5]", "got-3+demos-sa+cx+t5-nli-ec+ci+[0.05,0.6,0.35]", "dsp+sample", "dsp+knn", "got-3+demos+cx+t5-nli-ec+ci+[0.1,0.6,0.3]"]:
    #for method in ["got-3+demos-sa+cx+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-3+demos-sa+cx+t5-nli-ec+ci+[0.1,0.6,0.3]", "got-3+demos+cx+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-3+demos+cx+t5-nli-ec+ci+[0.1,0.6,0.3]"]:
    #for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.6,0.2]", "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.5,0.3]", "got-3+demos-sa+cx+t5-nli-ec+ci+[0.2,0.6,0.2]", "got-3+demos-sa+cx+t5-nli-ec+ci+[0.2,0.5,0.3]", "got-3+demos+cx+t5-nli-ec+ci+[0.2,0.6,0.2]", "got-3+demos+cx+t5-nli-ec+ci+[0.2,0.5,0.3]"]:
    #for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.6,0.1]"]:
    for method in ["react"]:
        #for dataset in ["open-squad-long","open-squad-medium", "open-squad-short", "hotpotqa-long","hotpotqa-medium","hotpotqa-short", "qrecc-long", "qrecc-medium", "qrecc-short"]:
        #for dataset in ["felm-medium"]:
        for dataset in ["hotpotqa-short"]:
            evaluate(method, dataset)
'''    
def annotate():
    default_stdout = sys.stdout
    log_file = open("log/annotate.log","w")
    sys.stdout = log_file
    
    dsp.settings.configure(electoral_college=nli_electoral_college)
    method_func = GoT_QA(p_context=True)
    #method_func = GoT_QA()
    dataset_dict = {"open-squad": "Open-SQuAD", "hotpotqa": "HotPotQA", "qrecc": "QReCC", "felm": "FELM", "wysdomqa": "WysdomQA"}
    #for dataset in ["open-squad-long", "open-squad-medium", "open-squad-short", "hotpotqa-long", "hotpotqa-medium", "hotpotqa-short", "qrecc-long", "qrecc-medium", "qrecc-short", "wysdomqa-medium"]:
    for dataset in ["hotpotqa-medium"]:
        folder_key, category = dataset.rsplit('-', 1)
        
        save_path = "data/%s/train_%s_augmented.csv"%(dataset_dict[folder_key], category)
        train, dev, test = load_data(dataset)
        train, dev, test = df_to_dsp(train), df_to_dsp(dev), df_to_dsp(test)
    
        method_func.annotate(train, save_path)
    
    sys.stdout = default_stdout
    log_file.close()
'''    
def main_test():
    language_model='gpt-3.5-turbo-1106'
    retrieval_model='google'
    
    init_langauge_model(language_model=language_model)
    init_retrieval_model(retrieval_model=retrieval_model)
    
    dataset = "hotpotqa-short"
    
    default_stdout = sys.stdout
    log_file = open("log/bak/%s_%s_%s_%s.log"%(dataset, "test", language_model, retrieval_model),"w")
    sys.stdout = log_file
    
    train, dev, test = load_data(dataset)
    train, dev, test = df_to_dsp(train), df_to_dsp(dev), df_to_dsp(test)

    W = [0.3,0.6,0.1]
    dsp.settings.configure(nli=_t5_nli)
    dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
    method_func = GoT_QA(demo_flags="plan+rewrite+rationale", p_context=True, W=W, annot_selector=EM, annot_dump="log/bak/%s_%s_%s_%s.csv"%(dataset, "test", language_model, retrieval_model))

    #question = 'Who held the record for the longest service in the Australian Parliament for a woman, and was surpassed by  a former Australian politician who was the 29th Speaker of the House of Representatives?'
    #question = 'This American crime film set in South Los Angeles was written and directed by the same director and writer of screenwriter of "Street Kings", "End of Watch", "Sabotage", "Fury" and what other film?'
    #question = "Mookychick is an independent daily online magazine and community with more than 100,000 readers a month and over 5,000 forum members, content includes analysis of current sociopolitical events, social and cultural trends, alternative fashion, movies, books, music and arts and crafts from a feminist perspective, in contrast with feminist publications and communities such as which women's lifestyle magazine that is published six times a year, and is published by Debbie Stoller and Laurie Henzel?"
    #question = "When was Buddha alive?"
    
    #question = """The American attorney, law professor and former member of the Texas House of Representatives who was portrayed by the actress known for "Love Child" (1982), "Places in the Heart" (1984), "Field of Dreams" (1989) and etc., is best known for which U. S. Supreme Court case?"""
    #question = "Hawaii v. Office of Hawaiian Affairs, 556 U.S. 163 (2009), was a United States Supreme Court case about the former crown lands of the Hawaiian monarchy, the Court, in an opinion by which Associate Justice of the Supreme Court of the United States, and has served on the court since January 31, 2006?"
    #question = "Capital Carnage was a UK-only professional wrestling pay-per-view (PPV) event produced by the World Wrestling Federation (WWF) that took place on which date, Jim Ross suffered his second Bells palsy attack on-air during this event, he officially called matches again for the WWE, in the main event of WrestleMania XV?"
    #question = """In the Cherokee Rose episode of "The Walking Dead," the character that continues to search for Sophia Peletier is portrayed by an actor who is also famous for his work in what company's advertisements?"""
    question = test[0].question
    
    print("#"*10 + question + "#"*10)
    prediction = method_func(train, question)
    
    print("="*35 + " ANSWER " + "="*35)
    print("."*35 + " prediction " + "."*35)
    print(prediction)
    print("."*35 + " ground truth " + "."*35)
    #print("Kathryn Jean Martin")
    #print("Suicide Squad")
    #print("Prada")
    print(test[0].answer)
    
    sys.stdout = default_stdout
    log_file.close()

if __name__=='__main__':
    main()
    #main(True)
    #main_test()
    #preprocess_data("felm")
    #annotate()
    '''
    df_dict = {}
    df_dict["open-squad"] = load_data("open-squad")
    df_dict["hotpotqa"] = load_data("hotpotqa")
    df_dict["qrecc"] = load_data("qrecc")
    df_dict["wysdomqa"] = load_data("wysdomqa")
    analyze_data(df_dict)
    '''
    