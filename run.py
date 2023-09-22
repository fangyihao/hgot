'''
Created on Jun. 1, 2023

@author: Yihao Fang
'''
import os
import dsp
#from dsp.evaluation.utils import evaluate
import gzip
import time
import csv
import ijson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import numpy as np
import sys
from metrics import OpenSQuADEM, OpenSQuADF1, HotPotEM, HotPotF1, QReCCF1, QReCCnF1, QueensEM, QueensF1, ElapsedTime
from pipelines import Vanilla_LM_QA, Retrieve_then_Read_SC_QA, Multihop_QA, DSP_QA, GoT_QA
import random
from judges import nli_electoral_college
import matplotlib.ticker as ticker
from functools import partial

from utils import df_to_dsp, df_to_dsp_augmented
from dotenv import load_dotenv

load_dotenv()

seed = 42
#language_model='gpt-3.5-turbo'
language_model='gpt-4'
retrieval_model='google'


np.random.seed(seed)
random.seed(seed)

root_path = '.'
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(root_path, 'cache')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

openai_key = os.getenv('OPENAI_API_KEY')  # or replace with your API key (optional)
serpapi_key = os.getenv('SERPAPI_API_KEY')  # or replace with your API key (optional)


if retrieval_model=='google':
    rm = dsp.Google(serpapi_key)
else:
    #colbert_server = 'http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search'
    colbert_server = 'http://192.168.3.200:8893/api/search'
    rm = dsp.ColBERTv2(url=colbert_server)

if language_model=='text-davinci-002':
    lm = dsp.GPT3(model=language_model, api_key=openai_key)
else:
    lm = dsp.GPT3(model=language_model, api_key=openai_key, model_type="chat")



dsp.settings.configure(lm=lm, rm=rm)
dsp.settings.configure(vectorizer=dsp.SentenceTransformersVectorizer())
dsp.settings.configure(electoral_college=None)
dsp.settings.lm.kwargs["max_tokens"] = 300




def load_json_gzip(filename, q_attr_name, a_attr_name):
    with gzip.open(filename, 'rb') as f:
        rows = ijson.items(f, 'item')
        df = pd.DataFrame(list(((r[q_attr_name], r[a_attr_name]) for r in rows)), columns =['Question', 'Answer'])
    return df

def load_json(filename, q_attr_name, a_attr_name, l_attr_name = None, cn_attr_name = None, tn_attr_name = None, cx_attr_name = None):
    with open(filename, 'r') as f:
        rows = ijson.items(f, 'item')
        if l_attr_name:
            df = pd.DataFrame(list(((r[q_attr_name], r[a_attr_name], r[l_attr_name]) for r in rows)), columns =['Question', 'Answer', 'Level'])
        elif cn_attr_name and tn_attr_name and cx_attr_name:
            df = pd.DataFrame(list(((r[q_attr_name], r[a_attr_name], r[cn_attr_name], r[tn_attr_name], r[cx_attr_name]) for r in rows)), columns =['Question', 'Answer', 'Conversation#', 'Turn#', 'Context'])
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

def select_hard_questions(df):
    df["Question Length"] = sent_len(df["Question"])
    len_threshold = df["Question Length"].quantile([0.98]).values[0]
    hard_df = df[df["Question Length"] > len_threshold]
    return hard_df

def select_easy_questions(df):
    df["Question Length"] = sent_len(df["Question"])
    len_threshold = df["Question Length"].quantile([0.02]).values[0]
    easy_df = df[df["Question Length"] <= len_threshold]
    
    easy_df = easy_df.sample(n=round(len(df)*0.02*0.2), random_state=seed)
    return easy_df

def select_medium_questions(df):
    df["Question Length"] = sent_len(df["Question"])
    len_lower_threshold, len_upper_threshold = df["Question Length"].quantile([0.02, 0.98]).values
    medium_df = df[(df["Question Length"] > len_lower_threshold) & (df["Question Length"] <= len_upper_threshold)]
    medium_df = medium_df.sample(frac=(0.02/0.96), random_state=seed)
    
    medium_df = medium_df.sample(n=round(len(df)*0.02*0.2), random_state=seed)
    return medium_df

def sample_n_save_data_by_difficulty(dataset, difficulty, train_df, dev_df, test_df):
    dataset_dict = {"open-squad": "Open-SQuAD", "hotpotqa": "HotPotQA", "qrecc": "QReCC", "queensqa": "QueensQA"}
    
    if difficulty == "hard":
        train_df, dev_df, test_df = select_hard_questions(train_df), select_hard_questions(dev_df), select_hard_questions(test_df)
    elif difficulty == "medium":
        train_df, dev_df, test_df = select_medium_questions(train_df), select_medium_questions(dev_df), select_medium_questions(test_df)
    elif difficulty == "easy":
        train_df, dev_df, test_df = select_easy_questions(train_df), select_easy_questions(dev_df), select_easy_questions(test_df)
    else:
        raise NotImplementedError()
    
    train_df.to_csv("data/%s/train_%s.csv"%(dataset_dict[dataset], difficulty), index=False)
    dev_df.to_csv("data/%s/dev_%s.csv"%(dataset_dict[dataset], difficulty), index=False)
    test_df.to_csv("data/%s/test_%s.csv"%(dataset_dict[dataset], difficulty), index=False)
    
    return train_df, dev_df, test_df

def preprocess_data(dataset):
    if dataset == "open-squad":
        train_df = load_json_gzip("data/Open-SQuAD/biencoder-squad1-train.json.gz", q_attr_name = 'question', a_attr_name = 'answers')
        dev_df = load_json_gzip("data/Open-SQuAD/biencoder-squad1-dev.json.gz", q_attr_name = 'question', a_attr_name = 'answers')
        test_df = load_tsv("data/Open-SQuAD/squad1-test.qa.csv")
        
        train_df.to_csv("data/Open-SQuAD/train.csv", index=False)
        dev_df.to_csv("data/Open-SQuAD/dev.csv", index=False)
        test_df.to_csv("data/Open-SQuAD/test.csv", index=False)
        
        for difficulty in ["hard", "medium", "easy"]:
            sample_n_save_data_by_difficulty(dataset, difficulty, train_df, dev_df, test_df)

    elif dataset == "hotpotqa":
        train_dev_df = load_json("data/HotPotQA/hotpot_train_v1.1.json", q_attr_name = 'question', a_attr_name = 'answer', l_attr_name = 'level')
        train_df = train_dev_df.sample(frac = 0.9, random_state=seed)
        dev_df = train_dev_df.drop(train_df.index)
        test_df = load_json("data/HotPotQA/hotpot_dev_fullwiki_v1.json", q_attr_name = 'question', a_attr_name = 'answer', l_attr_name = 'level')
        
        train_df.to_csv("data/HotPotQA/train.csv", index=False)
        dev_df.to_csv("data/HotPotQA/dev.csv", index=False)
        test_df.to_csv("data/HotPotQA/test.csv", index=False)
        
        for difficulty in ["hard", "medium", "easy"]:
            sample_n_save_data_by_difficulty(dataset, difficulty, train_df, dev_df, test_df)
        
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
        
        for difficulty in ["hard", "medium", "easy"]:
            sample_n_save_data_by_difficulty(dataset, difficulty, train_df, dev_df, test_df)
            
    elif dataset == "queensqa":
        train_dev_test_df = pd.read_csv("data/QueensQA/QueensQA.csv")
        train_dev_test_df['Question'] = train_dev_test_df.apply(lambda x: 'Is the query "' + x['Question'].rstrip('?') + '" accurately addressed by the response "' + x['Generative Answer'] + '" ? Kindly answer with either "Yes" or "No".', axis=1)
        train_dev_test_df['Answer'] = train_dev_test_df['Correct Answer']
        train_dev_test_df.drop('Generative Answer', axis=1, inplace=True)
        train_dev_test_df.drop('Correct Answer', axis=1, inplace=True)
        
        train_df = train_dev_test_df.sample(frac = 0.18, random_state=seed)
        dev_test_df = train_dev_test_df.drop(train_df.index)
        
        dev_df = dev_test_df.sample(frac = 0.0/0.82, random_state=seed)
        test_df = dev_test_df.drop(dev_df.index)
        
        train_df.to_csv("data/QueensQA/train.csv", index=False)
        dev_df.to_csv("data/QueensQA/dev.csv", index=False)
        test_df.to_csv("data/QueensQA/test.csv", index=False)
        
        train_df.to_csv("data/QueensQA/train_medium.csv", index=False)
        dev_df.to_csv("data/QueensQA/dev_medium.csv", index=False)
        test_df.to_csv("data/QueensQA/test_medium.csv", index=False)
    else:
        raise NotImplementedError()

def load_data(dataset):
    if dataset == "open-squad":
        train_df = pd.read_csv("data/Open-SQuAD/train.csv")
        dev_df = pd.read_csv("data/Open-SQuAD/dev.csv")
        test_df = pd.read_csv("data/Open-SQuAD/test.csv")
    elif dataset == "open-squad-hard":
        train_df = pd.read_csv("data/Open-SQuAD/train_hard.csv")
        dev_df = pd.read_csv("data/Open-SQuAD/dev_hard.csv")
        test_df = pd.read_csv("data/Open-SQuAD/test_hard.csv")
    elif dataset == "open-squad-medium":
        train_df = pd.read_csv("data/Open-SQuAD/train_medium.csv")
        dev_df = pd.read_csv("data/Open-SQuAD/dev_medium.csv")
        test_df = pd.read_csv("data/Open-SQuAD/test_medium.csv")
    elif dataset == "open-squad-easy":
        train_df = pd.read_csv("data/Open-SQuAD/train_easy.csv")
        dev_df = pd.read_csv("data/Open-SQuAD/dev_easy.csv")
        test_df = pd.read_csv("data/Open-SQuAD/test_easy.csv")
    elif dataset == "hotpotqa":
        train_df = pd.read_csv("data/HotPotQA/train.csv")
        dev_df = pd.read_csv("data/HotPotQA/dev.csv")
        test_df = pd.read_csv("data/HotPotQA/test.csv")
    elif dataset == "hotpotqa-hard":
        train_df = pd.read_csv("data/HotPotQA/train_hard.csv")
        dev_df = pd.read_csv("data/HotPotQA/dev_hard.csv")
        test_df = pd.read_csv("data/HotPotQA/test_hard.csv")
    elif dataset == "hotpotqa-medium":
        train_df = pd.read_csv("data/HotPotQA/train_medium.csv")
        dev_df = pd.read_csv("data/HotPotQA/dev_medium.csv")
        test_df = pd.read_csv("data/HotPotQA/test_medium.csv")
    elif dataset == "hotpotqa-easy":
        train_df = pd.read_csv("data/HotPotQA/train_easy.csv")
        dev_df = pd.read_csv("data/HotPotQA/dev_easy.csv")
        test_df = pd.read_csv("data/HotPotQA/test_easy.csv")
    elif dataset == "qrecc":    
        train_df = pd.read_csv("data/QReCC/train.csv")
        dev_df = pd.read_csv("data/QReCC/dev.csv")
        test_df = pd.read_csv("data/QReCC/test.csv")
    elif dataset == "qrecc-hard":    
        train_df = pd.read_csv("data/QReCC/train_hard.csv")
        dev_df = pd.read_csv("data/QReCC/dev_hard.csv")
        test_df = pd.read_csv("data/QReCC/test_hard.csv")
    elif dataset == "qrecc-medium":    
        train_df = pd.read_csv("data/QReCC/train_medium.csv")
        dev_df = pd.read_csv("data/QReCC/dev_medium.csv")
        test_df = pd.read_csv("data/QReCC/test_medium.csv")
    elif dataset == "qrecc-easy":    
        train_df = pd.read_csv("data/QReCC/train_easy.csv")
        dev_df = pd.read_csv("data/QReCC/dev_easy.csv")
        test_df = pd.read_csv("data/QReCC/test_easy.csv")
    elif dataset == "queensqa":    
        train_df = pd.read_csv("data/QueensQA/train.csv")
        dev_df = pd.read_csv("data/QueensQA/dev.csv")
        test_df = pd.read_csv("data/QueensQA/test.csv")
    elif dataset == "queensqa-medium":    
        train_df = pd.read_csv("data/QueensQA/train_medium.csv")
        dev_df = pd.read_csv("data/QueensQA/dev_medium.csv")
        test_df = pd.read_csv("data/QueensQA/test_medium.csv")
    else:
        raise NotImplementedError()
    return [train_df, dev_df, test_df]


def analyze_data(df_dict):
    dataset_dict = {"open-squad":"Open-SQuAD", "hotpotqa":"HotPotQA", "qrecc":"QReCC", "queensqa":"QueensQA"}
    
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



def retrieve_demos(dataset):
    if dataset == "open-squad-hard":
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_hard_augmented.csv",keep_default_na=False))[:2]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_hard_augmented.csv",keep_default_na=False))[1:2]
    elif dataset == "open-squad-medium":
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_medium_augmented.csv",keep_default_na=False))[:2]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_medium_augmented.csv",keep_default_na=False))[0:1]
    elif dataset == "open-squad-easy":
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_easy_augmented.csv",keep_default_na=False))[:2]
        #rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/Open-SQuAD/train_easy_augmented.csv",keep_default_na=False))[0:1]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    elif dataset == "hotpotqa-hard":
        #plan_demos = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_hard_augmented.csv",keep_default_na=False))[:2]
        #rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_hard_augmented.csv",keep_default_na=False))[0:1]
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[:2]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    elif dataset == "hotpotqa-medium":
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_medium_augmented.csv",keep_default_na=False))[:2]
        #rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_medium_augmented.csv",keep_default_na=False))[5:6] 
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    elif dataset == "hotpotqa-easy":
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_easy_augmented.csv",keep_default_na=False))[3:5]
        #rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/HotPotQA/train_easy_augmented.csv",keep_default_na=False))[5:6] 
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    elif dataset == "qrecc-hard":
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_hard_augmented.csv",keep_default_na=False))[2:4]
        #rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_hard_augmented.csv",keep_default_na=False))[5:6]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    elif dataset == "qrecc-medium":
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_medium_augmented.csv",keep_default_na=False))[3:5]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_medium_augmented.csv",keep_default_na=False))[4:5]
    elif dataset == "qrecc-easy":    
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_easy_augmented.csv",keep_default_na=False))[:2]
        #rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/QReCC/train_easy_augmented.csv",keep_default_na=False))[5:6]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    elif dataset == "queensqa-medium":
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/QueensQA/train_medium_augmented.csv",keep_default_na=False))[:2]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    else:
        raise NotImplementedError()
    return plan_demos, rewrite_demos

def evaluate(method, dataset):
    
    old_stdout = sys.stdout
    log_file = open("log/%s_%s_%s_%s.log"%(dataset, method, language_model, retrieval_model),"w")
    sys.stdout = log_file

    
    train, dev, test = load_data(dataset)
    #if "gpt-4" in language_model:
    #    train, dev, test = train.sample(frac=0.1, random_state=seed), dev.sample(frac=0.1, random_state=seed), test.sample(frac=0.1, random_state=seed)
    
    train, dev, test = df_to_dsp(train), df_to_dsp(dev), df_to_dsp(test)
    

    if method == "vanilla":
        dsp.settings.configure(electoral_college=None)
        method_func = Vanilla_LM_QA()
    elif method == "retrieve_then_read_sc":
        dsp.settings.configure(electoral_college=None)
        method_func = Retrieve_then_Read_SC_QA()
    elif method == "multihop":
        dsp.settings.configure(electoral_college=None)
        method_func = Multihop_QA()
    elif method == "dsp+sample":
        dsp.settings.configure(electoral_college=None)
        method_func = DSP_QA(dsp.sample)
    elif method == "dsp+knn":
        dsp.settings.configure(electoral_college=None)
        method_func = DSP_QA(dsp.knn(train))
    elif method == "dsp+knn+nli-ec":
        #dsp.settings.configure(reranker=partial(nli_reranker, retrieval_weight=0.5, nli_weight=0.5))
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = DSP_QA(dsp.knn(train))
    elif method == "got":
        dsp.settings.configure(electoral_college=None)
        method_func = GoT_QA(demos=None, p_context=False)
    elif method == "got+demos":
        dsp.settings.configure(electoral_college=None)
        method_func = GoT_QA(p_context=False)
        plan_demos, rewrite_demos = retrieve_demos(dataset)
        train += (plan_demos + rewrite_demos)
        method_func.PLAN_DEMOS = len(plan_demos)
        method_func.REWRITE_DEMOS = len(rewrite_demos)
    elif method == "got+demos+nli-ec":
        #dsp.settings.configure(reranker=partial(nli_reranker, retrieval_weight=0.5, nli_weight=0.5))
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = GoT_QA(p_context=False)
        plan_demos, rewrite_demos = retrieve_demos(dataset)
        train += (plan_demos + rewrite_demos)
        method_func.PLAN_DEMOS = len(plan_demos)
        method_func.REWRITE_DEMOS = len(rewrite_demos)
    elif method == "got+demos+cx":
        dsp.settings.configure(electoral_college=None)
        method_func = GoT_QA(p_context=True)
        plan_demos, rewrite_demos = retrieve_demos(dataset)
        train += (plan_demos + rewrite_demos)
        method_func.PLAN_DEMOS = len(plan_demos)
        method_func.REWRITE_DEMOS = len(rewrite_demos)
    elif method == "got+demos+cx+nli-ec":
        #dsp.settings.configure(reranker=partial(nli_reranker, retrieval_weight=0.5, nli_weight=0.5))
        dsp.settings.configure(electoral_college=nli_electoral_college)
        method_func = GoT_QA(p_context=True)
        plan_demos, rewrite_demos = retrieve_demos(dataset)
        train += (plan_demos + rewrite_demos)
        method_func.PLAN_DEMOS = len(plan_demos)
        method_func.REWRITE_DEMOS = len(rewrite_demos)
    else:
        raise NotImplementedError()

    if "open-squad" in dataset:
        metrics = [OpenSQuADEM(), OpenSQuADF1(), ElapsedTime()]
    elif "hotpotqa" in dataset:
        metrics = [HotPotEM(), HotPotF1(), ElapsedTime()]
    elif "qrecc" in dataset:
        metrics = [QReCCF1(), QReCCnF1(), ElapsedTime()]
    elif "queensqa" in dataset:
        metrics = [QueensEM(), QueensF1(), ElapsedTime()]
    else:
        raise NotImplementedError()


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

        for metric in metrics:
            if isinstance(metric, ElapsedTime):
                metric.evaluate(end-start)
            #elif isinstance(metric, CitationRecall) or isinstance(metric, CitationPrecision):
            #    metric.evaluate(prediction["retrieval_history"])
            elif isinstance(metric, QReCCnF1):
                metric.evaluate(prediction["answer"], example.answer, example.history) 
            else:
                metric.evaluate(prediction["answer"], example.answer)
    print("#"*35 + " RESULT " + "#"*35)
    [metric.average() for metric in metrics]
    [metric.clear() for metric in metrics]

    
    sys.stdout = old_stdout
    log_file.close()
    
def preprocess_n_analyze():
    [preprocess_data(dataset) for dataset in ["open-squad", "hotpotqa", "qrecc", "queensqa"]]
    
    df_dict = {}
    df_dict["open-squad"] = load_data("open-squad")
    df_dict["hotpotqa"] = load_data("hotpotqa")
    df_dict["qrecc"] = load_data("qrecc")
    df_dict["queensqa"] = load_data("queensqa")
    analyze_data(df_dict)
    
def main(preprocess = False):    
    if preprocess:
        preprocess_n_analyze()
    
    #for method in ["vanilla", "retrieve_then_read_sc", "multihop", "dsp+sample", "dsp+knn", "dsp+knn+nli-ec", "got", "got+demos", "got+demos+cx", "got+demos+cx+nli-ec"]:
    #for method in ["got+demos+cx+nli-ec", "dsp+knn+nli-ec"]:
    #for method in ["vanilla", "retrieve_then_read_sc", "multihop", "dsp+sample", "dsp+knn", "got"]:
    for method in ["got+demos+nli-ec"]:
        #for dataset in ["open-squad-hard","open-squad-medium", "open-squad-easy", "hotpotqa-hard","hotpotqa-medium","hotpotqa-easy", "qrecc-hard", "qrecc-medium", "qrecc-easy"]:
        #for dataset in ["queensqa-medium"]:
        for dataset in ["hotpotqa-medium"]:
            evaluate(method, dataset)
    
def annotate():
    old_stdout = sys.stdout
    log_file = open("log/annotate.log","w")
    sys.stdout = log_file
    
    method_func = GoT_QA()
    dataset_dict = {"open-squad": "Open-SQuAD", "hotpotqa": "HotPotQA", "qrecc": "QReCC", "queensqa": "QueensQA"}
    #for dataset in ["open-squad-hard", "open-squad-medium", "open-squad-easy", "hotpotqa-hard", "hotpotqa-medium", "hotpotqa-easy", "qrecc-hard", "qrecc-medium", "qrecc-easy", "queensqa-medium"]:
    for dataset in ["queensqa-medium"]:
        folder_key, category = dataset.rsplit('-', 1)
        
        save_path = "data/%s/train_%s_augmented.csv"%(dataset_dict[folder_key], category)
        train, dev, test = load_data(dataset)
        train, dev, test = df_to_dsp(train), df_to_dsp(dev), df_to_dsp(test)
    
        method_func.annotate(train, save_path)
    
    sys.stdout = old_stdout
    log_file.close()
    
def main_test():
    
    old_stdout = sys.stdout
    log_file = open("log/bak/test.log","w")
    sys.stdout = log_file
    
    dataset = "hotpotqa-hard"
    train, dev, test = load_data(dataset)
    train, dev, test = df_to_dsp(train), df_to_dsp(dev), df_to_dsp(test)
    '''
    method_func = GoT_QA()

    plan_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[:2]
    rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    #rewrite_demos = [dsp.Example(rewite_questions="Step 1: Who authored the opinion in Hawaii v. Office of Hawaiian Affairs, 556 U.S. 163 (2009)? ANSWER: Justice Samuel Alito. Step 2: When did the author join the Supreme Court?", rewrite="When did Justice Samuel Alito join the Supreme Court?", augmented=True)]
    train += (plan_demos + rewrite_demos)
    
    method_func.PLAN_DEMOS = len(plan_demos)
    method_func.REWRITE_DEMOS = len(rewrite_demos)
    '''
    
    method_func = GoT_QA()
    plan_demos, rewrite_demos = retrieve_demos(dataset)
    train += (plan_demos + rewrite_demos)
    method_func.PLAN_DEMOS = len(plan_demos)
    method_func.REWRITE_DEMOS = len(rewrite_demos)
    #dsp.settings.configure(reranker=partial(nli_reranker, retrieval_weight=0.5, nli_weight=0.5))
    

    #question = 'Who held the record for the longest service in the Australian Parliament for a woman, and was surpassed by  a former Australian politician who was the 29th Speaker of the House of Representatives?'
    #question = 'This American crime film set in South Los Angeles was written and directed by the same director and writer of screenwriter of "Street Kings", "End of Watch", "Sabotage", "Fury" and what other film?'
    #question = "Mookychick is an independent daily online magazine and community with more than 100,000 readers a month and over 5,000 forum members, content includes analysis of current sociopolitical events, social and cultural trends, alternative fashion, movies, books, music and arts and crafts from a feminist perspective, in contrast with feminist publications and communities such as which women's lifestyle magazine that is published six times a year, and is published by Debbie Stoller and Laurie Henzel?"
    #question = "When was Buddha alive?"
    
    #question = """The American attorney, law professor and former member of the Texas House of Representatives who was portrayed by the actress known for "Love Child" (1982), "Places in the Heart" (1984), "Field of Dreams" (1989) and etc., is best known for which U. S. Supreme Court case?"""
    #question = "Hawaii v. Office of Hawaiian Affairs, 556 U.S. 163 (2009), was a United States Supreme Court case about the former crown lands of the Hawaiian monarchy, the Court, in an opinion by which Associate Justice of the Supreme Court of the United States, and has served on the court since January 31, 2006?"
    #question = "Capital Carnage was a UK-only professional wrestling pay-per-view (PPV) event produced by the World Wrestling Federation (WWF) that took place on which date, Jim Ross suffered his second Bells palsy attack on-air during this event, he officially called matches again for the WWE, in the main event of WrestleMania XV?"
    question = """In the Cherokee Rose episode of "The Walking Dead," the character that continues to search for Sophia Peletier is portrayed by an actor who is also famous for his work in what company's advertisements?"""
    
    print("#"*10 + question + "#"*10)
    prediction = method_func(train, question)
    
    print("="*35 + " ANSWER " + "="*35)
    print("."*35 + " prediction " + "."*35)
    print(prediction)
    print("."*35 + " ground truth " + "."*35)
    #print("Kathryn Jean Martin")
    #print("Suicide Squad")
    print("Prada")
    
    sys.stdout = old_stdout
    log_file.close()

if __name__=='__main__':
    main()
    #main(True)
    #main_test()
    #preprocess_data("queensqa")
    #annotate()
    '''
    df_dict = {}
    df_dict["open-squad"] = load_data("open-squad")
    df_dict["hotpotqa"] = load_data("hotpotqa")
    df_dict["qrecc"] = load_data("qrecc")
    df_dict["queensqa"] = load_data("queensqa")
    analyze_data(df_dict)
    '''
    