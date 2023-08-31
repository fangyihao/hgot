'''
Created on Jun. 1, 2023

@author: Yihao Fang
'''
import os
import dsp
#from dsp.evaluation.utils import evaluate
import gzip
import json
import csv
import ijson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from dsp.utils import deduplicate
import functools
import numpy as np
import sys
from dsp.utils.metrics import EM, F1, HotPotF1, nF1
import re
import networkx as nx
from networkx.exception import NetworkXNoCycle
from collections import OrderedDict
import random
import copy
import matplotlib.ticker as ticker
from functools import partial
from reranker import nli_reranker
import time
import openai
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on

from dotenv import load_dotenv

load_dotenv()

seed = 42
language_model='gpt-3.5-turbo'
#language_model='gpt-4'
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

dsp.settings.lm.kwargs["max_tokens"] = 300




def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIConnectionError, openai.error.APIError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@CacheMemory.cache
def paraphrase(passage, n):
    start = time.time()
            
    if n == 1:
        instruction = "Please paraphrase the sentence below:\n"
    else:
        instruction = "Please generate %d paraphrases of the sentence below:\n"%n
       
    if n == 1:
        paraphrases = []   
        response = completions_with_backoff(
            model=language_model, 
            messages=[{"role": "user", "content": instruction + passage}]
        )
        
        content = response.choices[0].message.content
        paraphrases.append(content)
    
    else:
        #paraphrases = []
        contents = []
        content=""
        while not content.split('\n')[-1].startswith(str(n)):
            messages = []
            messages.append({"role": "user", "content": instruction + passage})
            if len(contents) > 0:
                messages.append({"role": "assistant", "content": contents[-1]})
                messages.append({"role": "user", "content": "continue"})
            
            response = completions_with_backoff(
                model=language_model, 
                messages=messages, 
                max_tokens=1920
            )
        
            content = response.choices[0].message.content
            contents.append(content)
            
        paraphrases = (''.join(contents)).split('\n')
        
    for i in range(len(paraphrases)):
        paraphrase = paraphrases[i]
        if len(paraphrase.strip()) > 0:
            mo = re.match(r"[0-9]+\.\s+(.*)", paraphrase)
            if mo:
                paraphrases[i] = mo.group(1)
            
    end = time.time()
    return paraphrases


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


class Vanilla_LM_QA:
    def __init__(self):
        Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
        Answer = dsp.Type(prefix="Answer:", desc="${a short factoid answer, often between 1 and 5 words}", format=dsp.format_answers)
        
        self.qa_template = dsp.Template(instructions="Answer questions with short factoid answers.", question=Question(), answer=Answer())
    
    def __call__(self, train, question):
        demos = dsp.sample(train, k=7)
        example = dsp.Example(question=question, demos=demos)
    
        example, completions = dsp.generate(self.qa_template)(example, stage='qa')
        return completions.answer
    
class Retrieve_then_Read_QA:
    def __init__(self):
        Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
        Answer = dsp.Type(prefix="Answer:", desc="${a short factoid answer, often between 1 and 5 words}", format=dsp.format_answers)
        
        qa_template = dsp.Template(instructions="Answer questions with short factoid answers.", question=Question(), answer=Answer())
    
        Context = dsp.Type(
            prefix="Context:\n",
            desc="${sources that may contain relevant content}",
            format=dsp.passages2text
        )
        
        self.qa_template_with_passages = dsp.Template(
            instructions=qa_template.instructions,
            context=Context(), question=Question(), answer=Answer()
        )
    def __call__(self, train, question):
        demos = dsp.sample(train, k=7)
        passages = dsp.retrieve(question, k=1)
        
        example = dsp.Example(question=question, context=passages, demos=demos)
        example, completions = dsp.generate(self.qa_template_with_passages)(example, stage='qa')
    
        return completions.answer
    
class Retrieve_then_Read_SC_QA:
    def __init__(self):
        Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
        Answer = dsp.Type(prefix="Answer:", desc="${a short factoid answer, often between 1 and 5 words}", format=dsp.format_answers)
        
        qa_template = dsp.Template(instructions="Answer questions with short factoid answers.", question=Question(), answer=Answer())
    
        Context = dsp.Type(
            prefix="Context:\n",
            desc="${sources that may contain relevant content}",
            format=dsp.passages2text
        )
        
        Rationale = dsp.Type(
            prefix="Rationale: Let's think step by step.",
            desc="${a step-by-step deduction that identifies the correct response, which will be provided below}"
        )
        
        self.qa_template_with_CoT = dsp.Template(
            instructions=qa_template.instructions,
            context=Context(), question=Question(), rationale=Rationale(), answer=Answer()
        )
        
    @dsp.transformation
    def QA_predict(self, example: dsp.Example, sc=True):
        if sc:
            example, completions = dsp.generate(self.qa_template_with_CoT, n=20, temperature=0.7)(example, stage='qa')
            completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(self.qa_template_with_CoT)(example, stage='qa')
        
        return example.copy(answer=completions.answer)
    
    def __call__(self, train, question):
        demos = dsp.sample(train, k=7)
        passages = dsp.retrieve(question, k=5)
        example = dsp.Example(question=question, context=passages, demos=demos)
        
        return self.QA_predict(example).answer
    
    
class Multihop_QA:
    def __init__(self):
        
        Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
        Answer = dsp.Type(prefix="Answer:", desc="${a short factoid answer, often between 1 and 5 words}", format=dsp.format_answers)
        
        qa_template = dsp.Template(instructions="Answer questions with short factoid answers.", question=Question(), answer=Answer())
    
        Context = dsp.Type(
            prefix="Context:\n",
            desc="${sources that may contain relevant content}",
            format=dsp.passages2text
        )
        
        Rationale = dsp.Type(
            prefix="Rationale: Let's think step by step.",
            desc="${a step-by-step deduction that identifies the correct response, which will be provided below}"
        )
        
        self.qa_template_with_CoT = dsp.Template(
            instructions=qa_template.instructions,
            context=Context(), question=Question(), rationale=Rationale(), answer=Answer()
        )
            
        SearchRationale = dsp.Type(
            prefix="Rationale: Let's think step by step. To answer this question, we first need to find out",
            desc="${the missing information}"
        )
        
        SearchQuery = dsp.Type(
            prefix="Search Query:",
            desc="${a simple question for seeking the missing information}"
        )
        
        self.rewrite_template = dsp.Template(
            instructions="Write a search query that will help answer a complex question.",
            question=Question(), rationale=SearchRationale(), query=SearchQuery()
        )
        
        CondenseRationale = dsp.Type(
            prefix="Rationale: Let's think step by step. Based on the context, we have learned the following.",
            desc="${information from the context that provides useful clues}"
        )
        
        self.hop_template = dsp.Template(
            instructions=self.rewrite_template.instructions,
            context=Context(), question=Question(), rationale=CondenseRationale(), query=SearchQuery()
        )
        
        
    @dsp.transformation
    def multihop_search_v1(self, example: dsp.Example, max_hops=2, k=2) -> dsp.Example:
        example.context = []
        
        for hop in range(max_hops):
            # Generate a query based
            template = self.rewrite_template if hop == 0 else self.hop_template
            example, completions = dsp.generate(template)(example, stage=f'h{hop}')
    
            # Retrieve k results based on the query generated
            passages = dsp.retrieve(completions.query, k=k)
    
            # Update the context by concatenating old and new passages
            example.context = deduplicate(example.context + passages)
    
        return example
    
    @dsp.transformation
    def QA_predict(self, example: dsp.Example, sc=True):
        if sc:
            example, completions = dsp.generate(self.qa_template_with_CoT, n=20, temperature=0.7)(example, stage='qa')
            completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(self.qa_template_with_CoT)(example, stage='qa')
        
        return example.copy(answer=completions.answer)
    
    def __call__(self, train, question):
        demos = dsp.sample(train, k=7)
        x = dsp.Example(question=question, demos=demos)
        
        x = self.multihop_search_v1(x)
        x = self.QA_predict(x, sc=False)
    
        return x.answer


class DSP_QA(Multihop_QA):
    
    def __init__(self, train_sel_func):
        super().__init__()
        self.train_sel_func = train_sel_func
    
    def annotate(self, train):
        """Returns an Augment function that applies the provided transformations to the Examples"""
    
        def do_augment(demos, k=None, return_all=False):
            rdemos = []
            ademos = []
    
            for example in demos:  # tqdm.tqdm
                raw_example = dsp.Example(example)
    
                if k and len(ademos) >= k:
                    example = None
    
                
                if example is None:
                    break

                example = self.multihop_attempt(train, example)
    
                if example is not None:
                    example.augmented = True
                    ademos.append(example)
                else:
                    raw_example.augmented = False
                    rdemos.append(raw_example)
    
            if return_all:
                return ademos + rdemos
    
            return ademos
    
        return do_augment
    
    @dsp.transformation
    def multihop_attempt(self, train, d: dsp.Example) -> dsp.Example:
        # Prepare unaugmented demonstrations for the example.
        x = dsp.Example(question=d.question, demos=dsp.all_but(train, d))
        
        # Search. And skip examples where search fails.
        # Annotate demonstrations for multihop_search_v2 with the simpler multihop_search_v1 pipeline.
        x = self.multihop_search_v1(x)
        if not dsp.passage_match(x.context, d.answer): return None
        
        # Predict. And skip examples where predict fails.
        x = self.QA_predict(x, sc=False)
        if not dsp.answer_match(x.answer, d.answer): return None
        
        return d.copy(**x)
    
    
    
    @dsp.transformation
    def multihop_demonstrate(self, train, x: dsp.Example) -> dsp.Example:
        if "sample" in self.train_sel_func.__name__:
            demos = self.train_sel_func(train, k=7)
        elif "knn" in self.train_sel_func.__name__:
            demos = self.train_sel_func(x, k=7)
        else:
            raise NotImplementedError()
        x.demos = self.annotate(demos)(demos, k=3, return_all=True)
        return x
    
    
    @dsp.transformation
    def multihop_search_v2(self, example: dsp.Example, max_hops=2, k=5) -> dsp.Example:
        example.context = []
    
        for hop in range(max_hops):
            # Generate queries
            template = self.rewrite_template if hop == 0 else self.hop_template
            example, completions = dsp.generate(template, n=3, temperature=0.7)(example, stage=f'h{hop}')
            
            # Collect the queries and search with result fusion
            queries = [c.query for c in completions] + [example.question]
            example.context = dsp.retrieveEnsemble(queries, k=k)
    
            # Arrange the passages for the next hop
            if hop > 0:
                example.context = [completions[0].rationale] + example.context
        
        return example
    
    def __call__(self, train, question: str) -> str: 
        x = dsp.Example(question=question)
        print("="*35 + " DEMONSTRATE " + "="*35)
        x = self.multihop_demonstrate(train, x)
        print("="*35 + " SEARCH " + "="*35)
        x = self.multihop_search_v2(x)
        print("="*35 + " PREDICT " + "="*35)
        x = self.QA_predict(x)
        return x.answer


class GoT_QA:

    def __init__(self, has_demos = True, has_context = True, retrieve_ensemble_n = 3):
        self.EDGE_PATTERN = r'\s*([Ss]tep [0-9]+)\s*->\s*([Ss]tep [0-9]+)\s*'
        self.NODE_PATTERN = r'\s*([Ss]tep [0-9]+):\s*(.*)'
        
        self.has_demos = has_demos
        self.has_context = has_context
        self.retrieve_ensemble_n = retrieve_ensemble_n
        self.annotator = None

        Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
        Answer = dsp.Type(prefix="Answer:", desc="${a short factoid answer, often between 1 and 5 words}", format=dsp.format_answers)
        
        qa_template = dsp.Template(instructions="Answer questions with short factoid answers.", question=Question(), answer=Answer())
    
        Context = dsp.Type(
            prefix="Context:\n",
            desc="${sources that may contain relevant content}",
            format=dsp.passages2text
        )
        
        Rationale = dsp.Type(
            prefix="Rationale: Let's think step by step.",
            desc="${a step-by-step deduction that identifies the correct response, which will be provided below}"
        )
        
        self.qa_template_with_CoT = dsp.Template(
            instructions=qa_template.instructions,
            context=Context(), question=Question(), rationale=Rationale(), answer=Answer()
        )
        
        Plan = dsp.Type(
            prefix="Plan:\n",
            desc="Step 1: ${a standalone search question} Step 2: ${a standalone search question} ... Step n: ${a standalone search question}"
        )
        
        Dependencies = dsp.Type(
            prefix="Dependencies: ",
            desc="${interdependencies among multiple steps}"
        )
        
        self.plan_template = dsp.Template(
            instructions="Sketch a plan to answer the following question with the provided context. List only the essential steps which can be answered by search engines. Express each step as a standalone search question. Highlight interdependencies if any. Higher number steps can depend on lower number steps, while the reverse is not possible.",
            question=Question(), context=Context(), plan = Plan(), dependencies = Dependencies()
        )
        
        self.plan_wo_cx_template = dsp.Template(
            instructions="Sketch a plan to answer the following question. List only the essential steps which can be answered by search engines. Express each step as a standalone search question. Highlight interdependencies if any. Higher number steps can depend on lower number steps, while the reverse is not possible.",
            question=Question(), plan = Plan(), dependencies = Dependencies()
        )
        
        Rewrite_Questions = dsp.Type(
            prefix="Questions:\n",
            desc="${previous questions and answers}"
        )
        
        Rewrite = dsp.Type(
            prefix="Rewrite: ",
            desc="${the last question after the rewrite}"
        )
        
        self.rewrite_template = dsp.Template(
            instructions="Rewrite the last question in a standalone manner by giving the answers to previous questions. Do not consider answers that were not specified. Only show the last question after the rewrite.",
            rewrite_questions=Rewrite_Questions(), rewrite = Rewrite()
        )
        
        Descriptions = dsp.Type(
            prefix="Descriptions: ",
            desc="${descriptions of dependencies}"
        )
        
        Dependencies_2 = dsp.Type(
            prefix="Dependencies: ",
            desc="${e.g. If Step 2 depends on Step 1, then write Step 1 -> Step 2; If Step 2 and Step 3 depend on Step 1, then write Step 1 -> (Step 2 and Step 3); If Step 3 depends on Step 1 and Step 2, then write (Step 1 and Step 2) -> Step 3}"
        )
        
        
        self.formalization_template = dsp.Template(
            instructions="Express the dependencies in formal language by giving the descriptions below.",
            descriptions=Descriptions(), dependencies = Dependencies_2()
        )
        
        self.reflection_template = dsp.Template(
            instructions="Highlight interdependencies among the steps below if any. Higher number steps can depend on lower number steps, while the reverse is not possible.",
            plan = Plan(), dependencies = Dependencies()
        )
    
    @dsp.transformation
    def QA_predict(self, example: dsp.Example, sc=True):
        if sc:
            example, completions = dsp.generate(self.qa_template_with_CoT, n=20, temperature=0.7)(example, stage='qa')
            completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(self.qa_template_with_CoT)(example, stage='qa')

        return example.copy(answer=completions.answer, rationale = completions.rationale)
    
    def find_step_question(self, step):
        mo = re.match(self.NODE_PATTERN, step)
        if mo:
            return mo.group(2)
        else:
            return step
        
    def find_steps(self, plan):

        plan = re.sub(r"\s(([Ss]tep [0-9]+):)", "\n\\1", plan)
        return plan.split('\n')
        
    def format_dependencies(self, dependencies):
        dependencies = dependencies.split(';')
        formatted = []
        for dependency in dependencies:
            mo_iter = re.finditer(r'(?=(([Ss]tep [0-9]+)\s*->\s*([Ss]tep [0-9]+)))', dependency)
            for mo in mo_iter:
                formatted.append(mo.group(1))
            mo_iter = re.finditer(r'(?=(([Ss]tep [0-9]+)\s*->\s*\(\s*([Ss]tep [0-9]+)\s*and\s*([Ss]tep [0-9]+)\s*\)))', dependency)
            for mo in mo_iter:
                formatted.append(mo.group(2) + " -> " + mo.group(3))
                formatted.append(mo.group(2) + " -> " + mo.group(4))
            mo_iter = re.finditer(r'(?=(\(\s*([Ss]tep [0-9]+)\s*and\s*([Ss]tep [0-9]+)\s*\)\s*->\s*([Ss]tep [0-9]+)))', dependency)
            for mo in mo_iter:
                formatted.append(mo.group(2) + " -> " + mo.group(4))
                formatted.append(mo.group(3) + " -> " + mo.group(4))
        return formatted
    

    @dsp.transformation
    def multistep_search(self, train, example: dsp.Example, k=2) -> dsp.Example:
        
        if self.retrieve_ensemble_n:
            
            def retrieve_ensemble(query: str, k: int) -> list[str]:
                #psgs = dsp.retrieve(query, k=k*3)
                #return psgs[:k]
                assert self.retrieve_ensemble_n >= 1
                
                if self.retrieve_ensemble_n > 1:
                    paraphrases = paraphrase(query, self.retrieve_ensemble_n-1)
                    return dsp.retrieveEnsemble([query]+paraphrases, k=k)
                elif self.retrieve_ensemble_n == 1:
                    return dsp.retrieveEnsemble([query], k=k)

        
        if self.has_context == False:
            example.context = []
        steps = self.find_steps(example.plan)

        _, completions = dsp.generate(self.formalization_template)(dsp.Example(descriptions=example.dependencies, demos=example.demos), stage='formalization')
        
        example = example.copy(dependencies=completions.dependencies)

        dependencies = self.format_dependencies(example.dependencies)
        G = nx.DiGraph()
        
        questions = {}
        for step in steps:
            mo = re.match(self.NODE_PATTERN, step)
            if mo:
                questions[mo.group(1).lower()] = step
                G.add_node(mo.group(1).lower())
        
        for dependency in dependencies:
            mo = re.match(self.EDGE_PATTERN, dependency)
            if mo:
                u = mo.group(1).lower()
                v = mo.group(2).lower()
                if u in G.nodes and v in G.nodes:
                    G.add_edge(u, v)
    
        no_cycle_found = False
        while not no_cycle_found:
            try:
                cycle = nx.find_cycle(G)
                G.remove_edge(*cycle[-1])
            except NetworkXNoCycle:
                no_cycle_found = True
        
        #num_func = lambda n: re.match(r'step ([0-9]+)',n).group(1)
        rev_G = G.reverse(copy=True)
        answers = OrderedDict()
        passages = OrderedDict()
        #while len(answers) < len(G.nodes):
            #for u in G.nodes:
        for u in nx.topological_sort(G):
            if G.in_degree(u) == 0:
                print("~"*35 + u.capitalize() + "~"*35)

                if self.retrieve_ensemble_n:
                    passages[u] = retrieve_ensemble(self.find_step_question(questions[u]), k=k)
                else:
                    passages[u] = dsp.retrieve(self.find_step_question(questions[u]), k=k)
                completions = self.QA_predict(dsp.Example(question=self.find_step_question(questions[u]), demos=example.demos, context=passages[u]))
                answers[u] = completions.answer
                rationale = completions.rationale
                #example.context.extend(passages[u])
                #example.context.extend([questions[u] + " | " + answers[u]])
            else:
                all_rev_neighbors = True
                rewrite_questions = []
                for v in rev_G.neighbors(u):
                    if v in answers:
                        rewrite_questions.extend([questions[v], "ANSWER: " + answers[v] + "."])
                    else:
                        all_rev_neighbors = False
                if all_rev_neighbors:
                    print("~"*35 + u.capitalize() + "~"*35)
                    rewrite_questions.append(questions[u])
                    
                    if self.annotator is not None:
                        if self.annotator._get_value(len(self.annotator)-1, 'Rewrite Questions') is None:
                            self.annotator._set_value(len(self.annotator)-1, 'Rewrite Questions', ' '.join(rewrite_questions))
                    
                    if self.has_demos == True:
                        _, completions = dsp.generate(self.rewrite_template)(dsp.Example(rewrite_questions=' '.join(rewrite_questions), demos=train[-self.REWRITE_DEMOS:]), stage='rewrite')
                    else:
                        _, completions = dsp.generate(self.rewrite_template)(dsp.Example(rewrite_questions=' '.join(rewrite_questions), demos=example.demos), stage='rewrite')
                    rewrite = completions.rewrite
                    
                    if self.annotator is not None:
                        if self.annotator._get_value(len(self.annotator)-1, 'Rewrite') is None:
                            self.annotator._set_value(len(self.annotator)-1, 'Rewrite', rewrite)
                    
                    if not rewrite.lower().startswith("step"):
                        rewrite = u.capitalize() + ": " + rewrite
                    questions[u] = rewrite

                    if self.retrieve_ensemble_n:
                        passages[u] = retrieve_ensemble(self.find_step_question(rewrite), k=k)
                    else:
                        passages[u] = dsp.retrieve(self.find_step_question(rewrite), k=k)
                    completions = self.QA_predict(dsp.Example(question=self.find_step_question(rewrite), demos=example.demos, context=passages[u]))
                    answers[u] = completions.answer
                    rationale = completions.rationale
                    #example.context.extend(passages[u])
                    #example.context.extend([questions[u] + " | " + answers[u]])
        assert len(answers) == len(G.nodes)     
        
        for u in questions:
            #example.context.extend([questions[u] + " | " + answers[u]])
            example.context.extend(passages[u][:1])
            
        print("-"*35 + " STEPS WITH ANSWERS " + "-"*35)
        for u in questions:
            print(questions[u] + " ANSWER: " + answers[u])
            
        return example
    
    def extract_plan(self, plan):
        return re.sub(r"(.+)Context:.+", r"\1", plan, flags=re.DOTALL)
    
    @dsp.transformation
    def plan(self, example: dsp.Example, self_reflect=True) -> dsp.Example:
        if self.has_context == True:
            example, completions = dsp.generate(self.plan_template)(example, stage='plan')
        else:
            example, completions = dsp.generate(self.plan_wo_cx_template)(example, stage='plan')
        plan = self.extract_plan(completions.plan)
        if self_reflect:
            _, completions = dsp.generate(self.reflection_template)(dsp.Example(plan=plan, demos=example.demos), stage='plan')
            return example.copy(plan = plan, dependencies = completions.dependencies)
        else:
            return example.copy(plan = plan, dependencies = completions.dependencies)
    
    def __call__(self, train, question):
        if self.has_demos == False and self.has_context == False:
            x = dsp.Example(question=question, demos=dsp.sample(train, k=7))
        elif self.has_demos == True:
            demos = train[-(self.PLAN_DEMOS+self.REWRITE_DEMOS):-self.REWRITE_DEMOS]
            if self.has_context == False:
                x = dsp.Example(question=question, demos=demos)
            else:
                context = dsp.retrieve(question, k=3)
                
                print("context:",context)
                
                if self.annotator is not None:
                    #self.annotator.iloc[len(self.annotator)-1, 1]=context
                    self.annotator._set_value(len(self.annotator)-1, 'Context', copy.deepcopy(context))
                    
                x = dsp.Example(question=question, demos=demos, context=context)
        else:
            raise NotImplementedError()
        print("="*35 + " PLAN " + "="*35)
        x = self.plan(x)
        
        if self.annotator is not None:
            self.annotator._set_value(len(self.annotator)-1, 'Plan', x.plan)
            self.annotator._set_value(len(self.annotator)-1, 'Dependencies', x.dependencies)
            
        if self.has_demos == True:
            demos = dsp.sample(train[:-(self.PLAN_DEMOS+self.REWRITE_DEMOS)], k=7)
            x.demos=demos
        print("="*35 + " SEARCH " + "="*35)
        x = self.multistep_search(train, x)
        print("="*35 + " PREDICT " + "="*35)
        x = self.QA_predict(x)
        return x.answer
    
    def annotate(self, train, save_path):
        self.annotator = pd.DataFrame(columns=['Question', 'Context', 'Plan', 'Dependencies', 'Rewrite Questions', 'Rewrite'])
        train, test = train[:len(train)//2], train[len(train)//2:]
        
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[:2]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    
        train += (plan_demos + rewrite_demos)
    
        self.PLAN_DEMOS = len(plan_demos)
        self.REWRITE_DEMOS = len(rewrite_demos)
        
        test = dsp.sample(test, k=5)
        for example in test:
            self.annotator.loc[len(self.annotator)] = [example.question,None, None, None, None, None]
            self.__call__(train, example.question)
        self.annotator.to_csv(save_path, index=False)
        self.annotator = None

def _eval(s):
    try:
        r = eval(s)
        if isinstance(r, list):
            return r
        else:
            return [s]
    except:
        return [s]

def df_to_dsp(df):
    return [dsp.Example(question=row["Question"], answer=_eval(row["Answer"]), history=_eval(row["Context"])) if "Context" in row else dsp.Example(question=row["Question"], answer=_eval(row["Answer"])) for index, row in df.iterrows()]

def df_to_dsp_augmented(df):
    return [dsp.Example(question=row["Question"], context=_eval(row["Context"]), plan=row["Plan"], dependencies=row["Dependencies"], rewrite_questions=row["Rewrite Questions"], rewrite=row["Rewrite"], augmented=True) for index, row in df.iterrows()]


class Metric:
    def __init__(self):
        self.result = []
    def evaluate(self, prediction, answer):
        raise NotImplementedError()
    def average(self):
        r = np.mean(self.result)
        print(r)
        return r
    def clear(self):
        self.result = []
        
class Open_Squad_EM(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " EM " + "."*35)
        em = EM(prediction, answer)
        self.result.append(em)
        print(em)
    def average(self):
        print("."*35 + " EM " + "."*35)
        return super().average()
        
class Open_Squad_F1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = F1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
        
class Hotpot_EM(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " EM " + "."*35)
        em = EM(prediction, answer)
        self.result.append(em)
        print(em)
    def average(self):
        print("."*35 + " EM " + "."*35)
        return super().average()
        
class Hotpot_F1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = HotPotF1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
    
class Qrecc_F1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = F1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
    
class Qrecc_nF1(Metric):
    def evaluate(self, prediction, answer, history):
        print("."*35 + " nF1 " + "."*35)
        nf1 = nF1(" ".join(history), prediction, answer)
        self.result.append(nf1)
        print(nf1)
    def average(self):
        print("."*35 + " nF1 " + "."*35)
        return super().average()

class Queens_EM(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " EM " + "."*35)
        em = EM(prediction, answer)
        self.result.append(em)
        print(em)
    def average(self):
        print("."*35 + " EM " + "."*35)
        return super().average()
        
class Queens_F1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = F1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()

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
    if "gpt-4" in language_model:
        train, dev, test = train.sample(frac=0.1, random_state=seed), dev.sample(frac=0.1, random_state=seed), test.sample(frac=0.1, random_state=seed)
    
    train, dev, test = df_to_dsp(train), df_to_dsp(dev), df_to_dsp(test)
    

    if method == "vanilla":
        method_func = Vanilla_LM_QA()
        dsp.settings.configure(reranker=None)
    elif method == "retrieve_then_read_sc":
        method_func = Retrieve_then_Read_SC_QA()
        dsp.settings.configure(reranker=None)
    elif method == "multihop":
        method_func = Multihop_QA()
        dsp.settings.configure(reranker=None)
    elif method == "dsp+sample":
        method_func = DSP_QA(dsp.sample)
        dsp.settings.configure(reranker=None)
    elif method == "dsp+knn":
        method_func = DSP_QA(dsp.knn(train))
        dsp.settings.configure(reranker=None)
    elif method == "dsp+knn+nli-rr":
        method_func = DSP_QA(dsp.knn(train))
        dsp.settings.configure(reranker=partial(nli_reranker, retrieval_weight=0.5, nli_weight=0.5))
    elif method == "got":
        method_func = GoT_QA(has_demos=False, has_context=False, retrieve_ensemble_n=None)
        dsp.settings.configure(reranker=None)
    elif method == "got+demos":
        method_func = GoT_QA(has_demos=True, has_context=False, retrieve_ensemble_n=None)
        plan_demos, rewrite_demos = retrieve_demos(dataset)
        train += (plan_demos + rewrite_demos)
        method_func.PLAN_DEMOS = len(plan_demos)
        method_func.REWRITE_DEMOS = len(rewrite_demos)
        dsp.settings.configure(reranker=None)
    elif method == "got+demos+cx":
        method_func = GoT_QA(has_demos=True, has_context=True, retrieve_ensemble_n=None)
        plan_demos, rewrite_demos = retrieve_demos(dataset)
        train += (plan_demos + rewrite_demos)
        method_func.PLAN_DEMOS = len(plan_demos)
        method_func.REWRITE_DEMOS = len(rewrite_demos)
        dsp.settings.configure(reranker=None)
    elif method == "got+demos+cx+nli-rr-e1":
        method_func = GoT_QA(has_demos=True, has_context=True, retrieve_ensemble_n=1)
        plan_demos, rewrite_demos = retrieve_demos(dataset)
        train += (plan_demos + rewrite_demos)
        method_func.PLAN_DEMOS = len(plan_demos)
        method_func.REWRITE_DEMOS = len(rewrite_demos)
        dsp.settings.configure(reranker=partial(nli_reranker, retrieval_weight=0.5, nli_weight=0.5))
    elif method == "got+demos+cx+nli-rr-e3":
        method_func = GoT_QA()
        plan_demos, rewrite_demos = retrieve_demos(dataset)
        train += (plan_demos + rewrite_demos)
        method_func.PLAN_DEMOS = len(plan_demos)
        method_func.REWRITE_DEMOS = len(rewrite_demos)
        dsp.settings.configure(reranker=partial(nli_reranker, retrieval_weight=0.5, nli_weight=0.5))
    else:
        raise NotImplementedError()

    if "open-squad" in dataset:
        metrics = [Open_Squad_EM(), Open_Squad_F1()]
    elif "hotpotqa" in dataset:
        metrics = [Hotpot_EM(), Hotpot_F1()]
    elif "qrecc" in dataset:
        metrics = [Qrecc_F1(), Qrecc_nF1()]
    elif "queensqa" in dataset:
        metrics = [Queens_EM(), Queens_F1()]
    else:
        raise NotImplementedError()


    for example in test: 
        print("#"*10 + example.question + "#"*10)

        prediction = method_func(train, example.question)
        print("="*35 + " ANSWER " + "="*35)
        print("."*35 + " prediction " + "."*35)
        print(prediction)
        print("."*35 + " ground truth " + "."*35)
        print(example.answer)

        [metric.evaluate(prediction, example.answer, example.history) if isinstance(metric, Qrecc_nF1) else metric.evaluate(prediction, example.answer) for metric in metrics]
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
    
    #for method in ["vanilla", "retrieve_then_read_sc", "multihop", "dsp+sample", "dsp+knn", "dsp+knn+nli-rr", "got", "got+demos", "got+demos+cx", "got+demos+cx+nli-rr-e1", "got+demos+cx+nli-rr-e3"]:
    for method in ["got+demos+cx"]:
        #for dataset in ["open-squad-hard","open-squad-medium", "open-squad-easy", "hotpotqa-hard","hotpotqa-medium","hotpotqa-easy", "qrecc-hard", "qrecc-medium", "qrecc-easy"]:
        for dataset in ["queensqa-medium"]:
        
            evaluate(method, dataset)
    
def annotate():
    old_stdout = sys.stdout
    log_file = open("log/annotate.log","w")
    sys.stdout = log_file
    
    method_func = GoT_QA()
    dataset_dict = {"open-squad": "Open-SQuAD", "hotpotqa": "HotPotQA", "qrecc": "QReCC", "queensqa": "QueensQA"}
    for dataset in ["open-squad-hard", "open-squad-medium", "open-squad-easy", "hotpotqa-hard", "hotpotqa-medium", "hotpotqa-easy", "qrecc-hard", "qrecc-medium", "qrecc-easy", "queensqa-medium"]:
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
    dsp.settings.configure(reranker=partial(nli_reranker, retrieval_weight=0.5, nli_weight=0.5))
    

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
    