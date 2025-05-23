'''
Created on Feb 21, 2025

@author: Yihao Fang
'''


import random
import numpy as np
import dsp
seed = 42
np.random.seed(seed)
random.seed(seed)
dsp.settings.branch_idx=seed

import gzip

import csv
import ijson
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize





#def load_jsonl(filename, q_attr_name, seg_attr_name, lbl_attr_name, dm_attr_name):
def load_jsonl(filename, attrs):
    df_list = []
    with open(filename, 'r') as f:
        for jl in f:
            r = json.loads(jl)
            t = []
            for key in attrs:
                t.append(r[key])
            t = tuple(t)
            df_list.append(t)
    columns = []
    for key in attrs:
        columns.append(attrs[key])
    df = pd.DataFrame(df_list, columns = columns)
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
    dataset_dict = {"open-squad": "Open-SQuAD", "hotpotqa": "HotPotQA", "qrecc": "QReCC", "fever": "FEVER", "felm": "FELM", "wysdomqa": "WysdomQA"}
    
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
    
    elif dataset == "fever":
        train_df = load_jsonl("data/FEVER/train.jsonl", attrs={'claim':'Question', 'label':'Answer'})
        dev_df = load_jsonl("data/FEVER/paper_dev.jsonl", attrs={'claim':'Question', 'label':'Answer'})
        test_df = load_jsonl("data/FEVER/paper_test.jsonl", attrs={'claim':'Question', 'label':'Answer'})
        
        train_df['Question'] = train_df.apply(lambda x: x['Question'] + ' Kindly answer with "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO".', axis=1)
        dev_df['Question'] = dev_df.apply(lambda x: x['Question'] + ' Kindly answer with "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO".', axis=1)
        test_df['Question'] = test_df.apply(lambda x: x['Question'] + ' Kindly answer with "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO".', axis=1)
        
        train_df.to_csv("data/FEVER/train.csv", index=False)
        dev_df.to_csv("data/FEVER/dev.csv", index=False)
        test_df.to_csv("data/FEVER/test.csv", index=False)
        
        for sent_len in ["long", "medium", "short"]:
            sample_n_save_data_by_sent_len(dataset, sent_len, train_df, dev_df, test_df, proportion=0.015)
        
    elif dataset == "felm":
        train_dev_test_df = load_jsonl("data/FELM/all.jsonl", attrs = {'prompt':'Question', 'segmented_response':'Segments', 'labels':'Labels', 'domain':'Domain'})
        train_dev_test_df['Question'] = train_dev_test_df.apply(
            lambda x: 'Is the query "' + x['Question'].rstrip('?') 
            + '" accurately addressed by the response (which has been split into a list of text segments) "' 
            + ' '.join(['%d. %s'%(i+1, segment) for i, segment in enumerate(x['Segments'])])
            #+ '" ? List the IDs of the segments with errors (separated by commas). If all the segments are correct, output "ALL_CORRECT".', axis=1)
            + '" ? Generate an answer detailing the accuracy of each individual segment, e.g., 1. True; 2. False; ...', axis=1)
            
            
        train_dev_test_df['_Answer'] = train_dev_test_df.apply(lambda x: "ALL_CORRECT" if all(x['Labels']) else ",".join([str(i+1) for i, label in enumerate(x['Labels']) if not label]), axis=1)
        train_dev_test_df['Answer'] = train_dev_test_df.apply(lambda x: "; ".join(["%d. %s"%(i+1,str(label)) for i, label in enumerate(x['Labels'])]), axis=1)
        
        
        # transform to a balanced dataset
        def transform(df, granularity="segment"):
            if granularity=="response":
                cor_len = len(df[df['_Answer'] == "ALL_CORRECT"].index)
                incor_len = len(df[df['_Answer'] != "ALL_CORRECT"].index)
                if cor_len > incor_len:
                    df = df.drop(df[df['_Answer'] == "ALL_CORRECT"].sample(n=cor_len-incor_len).index)
                else:
                    df = df.drop(df[df['_Answer'] != "ALL_CORRECT"].sample(n=incor_len-cor_len).index)
                return df
            elif granularity=="segment":
                cor_len = len(df[df['_Answer'] == "ALL_CORRECT"].index)
                incor_len = len(df[df['_Answer'] != "ALL_CORRECT"].index)
                sum_cor_t_count = np.sum([np.sum(np.array(labels).astype(int)) for labels in df[df['_Answer'] == "ALL_CORRECT"]['Labels'].values])
                sum_cor_n_count = np.sum([np.sum(np.logical_not(labels).astype(int)) for labels in df[df['_Answer'] == "ALL_CORRECT"]['Labels'].values])
                sum_incor_t_count = np.sum([np.sum(np.array(labels).astype(int)) for labels in df[df['_Answer'] != "ALL_CORRECT"]['Labels'].values])
                sum_incor_n_count = np.sum([np.sum(np.logical_not(labels).astype(int)) for labels in df[df['_Answer'] != "ALL_CORRECT"]['Labels'].values])
                
                if (sum_cor_t_count + sum_incor_t_count) > (sum_cor_n_count + sum_incor_n_count):
                    mean_cor_t_count = sum_cor_t_count / cor_len
                    cor_drop_count = int(((sum_cor_t_count + sum_incor_t_count) - (sum_cor_n_count + sum_incor_n_count))/mean_cor_t_count)
                    cor_drop_count = min(cor_drop_count, cor_len-1)
                    df = df.drop(df[df['_Answer'] == "ALL_CORRECT"].sample(n=cor_drop_count).index)
                else:
                    mean_incor_n_count = sum_incor_n_count / incor_len
                    mean_incor_t_count = sum_incor_t_count / incor_len
                    incor_drop_count = int(((sum_cor_n_count + sum_incor_n_count)-(sum_cor_t_count + sum_incor_t_count))/(mean_incor_n_count - mean_incor_t_count))
                    incor_drop_count = min(incor_drop_count, incor_len-1)
                    df = df.drop(df[df['_Answer'] != "ALL_CORRECT"].sample(n=incor_drop_count).index)
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
    elif dataset == "fever":
        train_df = pd.read_csv("data/FEVER/train.csv")
        dev_df = pd.read_csv("data/FEVER/dev.csv")
        test_df = pd.read_csv("data/FEVER/test.csv")
    elif dataset == "fever-long":
        train_df = pd.read_csv("data/FEVER/train_long.csv")
        dev_df = pd.read_csv("data/FEVER/dev_long.csv")
        test_df = pd.read_csv("data/FEVER/test_long.csv")
    elif dataset == "fever-medium":    
        train_df = pd.read_csv("data/FEVER/train_medium.csv")
        dev_df = pd.read_csv("data/FEVER/dev_medium.csv")
        test_df = pd.read_csv("data/FEVER/test_medium.csv")
    elif dataset == "fever-short":    
        train_df = pd.read_csv("data/FEVER/train_short.csv")
        dev_df = pd.read_csv("data/FEVER/dev_short.csv")
        test_df = pd.read_csv("data/FEVER/test_short.csv")
    elif dataset == "felm":
        train_df = pd.read_csv("data/FELM/train.csv")
        dev_df = pd.read_csv("data/FELM/dev.csv")
        test_df = pd.read_csv("data/FELM/test.csv")
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
    title_dict = {"open-squad":"Open-SQuAD", "open-squad-long":"Open-SQuAD (Long)", "open-squad-medium":"Open-SQuAD (Medium)", "open-squad-short":"Open-SQuAD (Short)",
                  "hotpotqa":"HotPotQA", "hotpotqa-long":"HotPotQA (Long)", "hotpotqa-medium":"HotPotQA (Medium)", "hotpotqa-short":"HotPotQA (Short)",
                  "qrecc":"QReCC", "qrecc-long":"QReCC (Long)", "qrecc-medium":"QReCC (Medium)", "qrecc-short":"QReCC (Short)",
                  "fever":"FEVER", "fever-long":"FEVER (Long)", "fever-medium":"FEVER (Medium)", "fever-short":"FEVER (Short)",
                  "felm":"FELM", "felm-long":"FELM (Long)", "felm-medium":"FELM (Medium)", "felm-short":"FELM (Short)",
                  "wysdomqa":"WysdomQA", "wysdomqa-long":"WysdomQA (Long)", "wysdomqa-medium":"WysdomQA (Medium)", "wysdomqa-short":"WysdomQA (Short)"}
    
    stats_df = pd.DataFrame(columns=['Dataset', 'Subset', '# of Instances', 'Maximum', 'Minimum', 'Median', 'Mean', 'Standard Deviation'])
    for dataset in df_dict:
        train_len_df = pd.DataFrame(sent_len(df_dict[dataset][0]["Question"].values), columns=["Train"])
        dev_len_df = pd.DataFrame(sent_len(df_dict[dataset][1]["Question"].values), columns=["Dev"])
        test_len_df = pd.DataFrame(sent_len(df_dict[dataset][2]["Question"].values), columns=["Test"])
        
        stats_df.loc[len(stats_df)] = [title_dict[dataset],"Train", len(train_len_df["Train"].values), train_len_df["Train"].max(), train_len_df["Train"].min(), train_len_df["Train"].median(), train_len_df["Train"].mean(), train_len_df["Train"].std()]
        stats_df.loc[len(stats_df)] = [title_dict[dataset],"Dev", len(dev_len_df["Dev"].values), dev_len_df["Dev"].max(), dev_len_df["Dev"].min(), dev_len_df["Dev"].median(), dev_len_df["Dev"].mean(), dev_len_df["Dev"].std()]
        stats_df.loc[len(stats_df)] = [title_dict[dataset],"Test", len(test_len_df["Test"].values), test_len_df["Test"].max(), test_len_df["Test"].min(), test_len_df["Test"].median(), test_len_df["Test"].mean(), test_len_df["Test"].std()]
    stats_df.to_csv("log/question_length.csv", index=False)
    
    for dataset in df_dict:
        train_len_df = pd.DataFrame(sent_len(df_dict[dataset][0]["Question"].values), columns=["Train"])
        dev_len_df = pd.DataFrame(sent_len(df_dict[dataset][1]["Question"].values), columns=["Dev"])
        test_len_df = pd.DataFrame(sent_len(df_dict[dataset][2]["Question"].values), columns=["Test"])
        df = pd.concat([train_len_df, dev_len_df, test_len_df], join = 'outer', axis = 1)
        
        fig, ax = plt.subplots(figsize=(3,3))
        sns.histplot(df, element='step', fill=True, kde=False, alpha=0.5, shrink=0.8, multiple='dodge', bins=20)
        
        ax.legend_.set_title('') # remove the legend title (the name of the hue column)
        ax.margins(x=0.01) # less spacing
        #ax.set_title(title_dict[dataset])
        ax.set_xlabel("Sentence Length")
        ax.set_yscale('log')
        plt.tight_layout()
        plt.draw()
        #plt.show()
        plt.savefig("log/%s_question_length.png"%dataset)
        plt.clf()
        plt.close() 
    
    
    fig, axes = plt.subplots(nrows=1,ncols=len(df_dict), figsize=(5/3*len(df_dict),2))
    
    for i, dataset in enumerate(df_dict):
        train_len_df = pd.DataFrame(sent_len(df_dict[dataset][0]["Question"].values), columns=["Train"])
        dev_len_df = pd.DataFrame(sent_len(df_dict[dataset][1]["Question"].values), columns=["Dev"])
        test_len_df = pd.DataFrame(sent_len(df_dict[dataset][2]["Question"].values), columns=["Test"])
        df = pd.concat([train_len_df, dev_len_df, test_len_df], join = 'outer', axis = 1)
        
        #plt.subplot(1, 3, i+1)
        ax = axes[i]
        #ax = plt.gca()
        sns.histplot(df, element='step', fill=True, kde=False, alpha=0.5, shrink=0.8, multiple='dodge', bins=20, ax=ax, log_scale=False)
        
        
        ax.margins(x=0.01) # less spacing
        ax.set_title(title_dict[dataset])
        ax.set_xlabel("Sentence Length")
        #plt.yscale('log')
        ax.set_yscale('log')
        ax.set_yticks([10,1000])
        #formatter = ticker.FuncFormatter(lambda x, pos: "%.0fK"%(x/1000) if x > 0 else "0")
        #ax.yaxis.set_major_formatter(formatter)
        
        if i < len(df_dict) - 1:
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

