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
from metric import OpenSQuADEM, OpenSQuADF1, HotPotEM, HotPotF1, QReCCF1, QReCCnF1, FEVEREM, FELMF1, FELMBalAcc, FELMMetric, WysdomEM, WysdomF1, ElapsedTime
from pipeline import Vanilla_LM_QA, Retrieve_then_Read_SC_QA, Multihop_QA, DSP_QA, GoT_QA, ReAct
from judge import nli_electoral_college
import matplotlib.ticker as ticker
from functools import partial
from util import df_to_dsp, df_to_dsp_augmented
from model import init_langauge_model, init_retrieval_model
from data import load_data, preprocess_data, analyze_data

verbose = False
if verbose:
    from nli import _t5_nli_logged as _t5_nli
    from nli import _gpt_nli_logged as _gpt_nli
else:
    from nli import _t5_nli, _gpt_nli



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





def evaluate(method, dataset):
    
    if method == "react":
        language_model='text-davinci-002'
    else:
        language_model='gpt-3.5-turbo-1106'
    retrieval_model='google'

    init_langauge_model(language_model=language_model)
    init_retrieval_model(retrieval_model=retrieval_model)
    dsp.settings.configure(vectorizer=dsp.SentenceTransformersVectorizer())
    
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
        elif "fever" in dataset:
            return EM
        elif "felm" in dataset:
            '''
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
            '''
            return EM
        elif "wysdomqa" in dataset:
            return EM
        else:
            raise NotImplementedError()
        
    def _annot_max_pri_rationale_demos():
        if "fever" in dataset:
            return 3
        elif "hotpotqa" in dataset:
            if "long" in dataset:
                return 2
            else:
                return 3
        elif "open-squad" in dataset:
            if "medium" in dataset:
                return 3
            else:
                return 2
        else:
            return 2
        
    def _annot_min_cand_rationale_demos():
        if "fever" in dataset:
            return 64
        elif "hotpotqa" in dataset:
            if "long" in dataset:
                return 32
            else:
                return 128
        elif "open-squad" in dataset:
            if "medium" in dataset:
                return 128
            else:
                return 32
        else:
            return 32
        
    def _annot_step():
        if "fever" in dataset:
            return 300
        else:
            return 50
        
    def _annot_balance():
        if "fever" in dataset:
            return True
        else:
            return False
        
        
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
        params = {}
        if "fever" in dataset:
            folder = 'react/prompts/'
            prompt_file = 'fever.json'
            with open(folder + prompt_file, 'r') as f:
                prompt_dict = json.load(f)
            prompt = prompt_dict['webthink_simple3']
            
            q_prefix = "Claim:"
            
            params["prompt"] = prompt
            params["q_prefix"] = q_prefix
            params["q_transform"] = lambda x: x.replace(' Kindly answer with "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO".', '')
        
        method_func = ReAct(**params)
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
        W_R = eval(method.split('+')[-1])
        W_T = eval(method.split('+')[-2])
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True, W=W_T))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", p_context=True, W=W_R, annot_step=_annot_step(), annot_selector=_annot_selector(), annot_dump=annot_dump, annot_log=annot_log, annot_max_pri_rationale_demos=_annot_max_pri_rationale_demos(), annot_min_cand_rationale_demos=_annot_min_cand_rationale_demos(), annot_balance=_annot_balance())
    elif method.startswith("got-3+demos-sa-knn+cx+t5-nli-ec+ci"):
        W_R = eval(method.split('+')[-1])
        W_T = eval(method.split('+')[-2])
        dsp.settings.configure(nli=_t5_nli)
        dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True, W=W_T))
        method_func = GoT_QA(demo_flags="plan+rewrite+rationale", p_context=True, W=W_R, annot_step=_annot_step(), annot_selector=_annot_selector(), annot_dump=annot_dump, annot_log=annot_log, annot_max_pri_rationale_demos=_annot_max_pri_rationale_demos(), annot_min_cand_rationale_demos=_annot_min_cand_rationale_demos(), annot_balance=_annot_balance(), demo_sel_func=dsp.knn)
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
    elif "fever" in dataset:
        metrics = [FEVEREM(), ElapsedTime()]
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
    [preprocess_data(dataset) for dataset in ["open-squad", "hotpotqa", "qrecc", "fever", "felm", "wysdomqa"]]
    
    df_dict = {}
    df_dict["open-squad"] = load_data("open-squad")
    df_dict["hotpotqa"] = load_data("hotpotqa")
    df_dict["qrecc"] = load_data("qrecc")
    df_dict["fever"] = load_data("fever")
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
    #for method in ["got-3+demos-sa+cx+t5-nli-ec+ci+[0.3,0.6,0.1]"]:
    #for method in ["vanilla", "retrieve_then_read_sc", "multihop", "dsp+sample", "dsp+knn"]:   
    #for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.35,0.55,0.1]", "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.55,0.15]", "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.25,0.6,0.15]"]:   
    #for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.5,0.2]", "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.45,0.25]"]:  
    #for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.54,0.16]", "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.56,0.14]"]:  
    #for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.6,0.1]", "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.5,0.2]"]:  
    '''
    for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.3,0.6,0.1]", 
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.3,0.55,0.15]", 
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.3,0.5,0.2]", 
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.3,0.45,0.25]",
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.3,0.4,0.3]",
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.25,0.6,0.15]", 
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.25,0.55,0.2]",
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.25,0.5,0.25]",
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.25,0.45,0.3]",
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.2,0.6,0.2]",
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.2,0.55,0.25]",
                   "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.2,0.5,0.3]"]:  
    '''
    '''
    for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[1.0,0.0,0.0]+[0.2,0.55,0.25]",
               "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[1.0,0.0,0.0]"]:      
    '''
    '''
    for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.15,0.55,0.3]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.15,0.6,0.25]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.4,0.3,0.3]+[0.2,0.55,0.25]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[0.2,0.55,0.25]"]:    
    '''
    '''
    for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[1.0,0.0,0.0]+[0.3,0.6,0.1]",
               "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.4,0.3,0.3]+[0.3,0.6,0.1]",
               "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[0.3,0.6,0.1]"]:
    '''
    '''
    for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.1,0.45,0.45]+[0.3,0.6,0.1]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.1,0.45,0.45]+[0.2,0.55,0.25]"]:    
    '''
    '''
    for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.1,0.45,0.45]+[0.3,0.5,0.2]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[0.3,0.5,0.2]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.4,0.3,0.3]+[0.3,0.5,0.2]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[1.0,0.0,0.0]+[0.3,0.5,0.2]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.1,0.45,0.45]+[1.0,0.0,0.0]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[1.0,0.0,0.0]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.4,0.3,0.3]+[1.0,0.0,0.0]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[1.0,0.0,0.0]+[1.0,0.0,0.0]",]:
    '''
    
    for method in ["got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.1,0.45,0.45]+[0.15,0.55,0.3]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.15,0.55,0.3]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[0.15,0.55,0.3]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.4,0.3,0.3]+[0.15,0.55,0.3]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[1.0,0.0,0.0]+[0.15,0.55,0.3]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.1,0.45,0.45]+[0.2,0.55,0.25]",
            "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.2,0.55,0.25]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[0.2,0.55,0.25]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.4,0.3,0.3]+[0.2,0.55,0.25]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[1.0,0.0,0.0]+[0.2,0.55,0.25]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.1,0.45,0.45]+[0.3,0.6,0.1]",
            "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.3,0.6,0.1]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[0.3,0.6,0.1]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.4,0.3,0.3]+[0.3,0.6,0.1]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[1.0,0.0,0.0]+[0.3,0.6,0.1]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.1,0.45,0.45]+[0.3,0.5,0.2]",
            "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[0.3,0.5,0.2]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[0.3,0.5,0.2]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.4,0.3,0.3]+[0.3,0.5,0.2]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[1.0,0.0,0.0]+[0.3,0.5,0.2]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.1,0.45,0.45]+[1.0,0.0,0.0]",
            "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.2,0.4,0.4]+[1.0,0.0,0.0]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[1.0,0.0,0.0]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.4,0.3,0.3]+[1.0,0.0,0.0]",
           "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[1.0,0.0,0.0]+[1.0,0.0,0.0]"]:

        #for dataset in ["open-squad-long","open-squad-medium", "open-squad-short", "hotpotqa-long","hotpotqa-medium","hotpotqa-short", "qrecc-long", "qrecc-medium", "qrecc-short", "fever-long", "fever-medium", "fever-short"]:
        for dataset in ["hotpotqa-medium"]:   
        #for dataset in ["hotpotqa-long"]:   
        #for dataset in ["open-squad-long","open-squad-medium", "open-squad-short", "hotpotqa-long","hotpotqa-short", "qrecc-long", "qrecc-medium", "qrecc-short", "fever-long", "fever-medium", "fever-short"]: 
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
    log_file = open("log/temp/%s_%s_%s_%s.log"%(dataset, "test", language_model, retrieval_model),"w")
    sys.stdout = log_file
    
    train, dev, test = load_data(dataset)
    train, dev, test = df_to_dsp(train), df_to_dsp(dev), df_to_dsp(test)

    W = [0.3,0.6,0.1]
    dsp.settings.configure(nli=_t5_nli)
    dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True))
    method_func = GoT_QA(demo_flags="plan+rewrite+rationale", p_context=True, W=W, annot_selector=EM, annot_dump="log/temp/%s_%s_%s_%s.csv"%(dataset, "test", language_model, retrieval_model))

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
    #preprocess_data("fever")
    #annotate()
    '''
    df_dict = {}
    df_dict["fever"] = load_data("fever")
    df_dict["open-squad"] = load_data("open-squad")
    df_dict["hotpotqa"] = load_data("hotpotqa")
    
    df_dict["fever-long"] = load_data("fever-long")
    df_dict["fever-medium"] = load_data("fever-medium")
    df_dict["fever-short"] = load_data("fever-short")
    df_dict["open-squad-long"] = load_data("open-squad-long")
    df_dict["open-squad-medium"] = load_data("open-squad-medium")
    df_dict["open-squad-short"] = load_data("open-squad-short")
    df_dict["hotpotqa-long"] = load_data("hotpotqa-long")
    df_dict["hotpotqa-medium"] = load_data("hotpotqa-medium")
    df_dict["hotpotqa-short"] = load_data("hotpotqa-short")
    
    #df_dict["qrecc"] = load_data("qrecc")
    #df_dict["wysdomqa"] = load_data("wysdomqa")
    analyze_data(df_dict)
    '''
    