'''
Created on Sep. 7, 2023

@author: Yihao Fang
'''
import dsp
import numpy as np
import random
from dsp.primitives.demonstrate import Example
from pandas import DataFrame
import os
def _eval(s):
    if isinstance(s, list):
        return s
    try:
        r = eval(s)
        if isinstance(r, list):
            return r
        else:
            return [s]
    except:
        return [s]

def df_to_dsp(df):
    return [dsp.Example(question=row["Question"], answer=_eval(row["Answer"]), history=_eval(row["Context"])) if "Context" in row 
            else dsp.Example(question=row["Question"], answer=row["Answer"], labels=_eval(row["Labels"])) if "Labels" in row
            else dsp.Example(question=row["Question"], answer=_eval(row["Answer"])) for index, row in df.iterrows()]

def df_to_dsp_augmented(df, segment):
    if segment == "plan":
        return [dsp.Example(question=row["Question"], context=_eval(row["Plan Context"]), plan=row["Plan"], dependencies=row["Dependencies"], augmented=True) for index, row in df.iterrows() 
                if row["Question"] is not None and row["Plan Context"] is not None and row["Plan"] is not None and row["Dependencies"] is not None]
    elif segment == "rewrite":
        return [dsp.Example(question=row["Question"], rewrite_context=row["Rewrite Context"], rewrite=row["Rewrite"], augmented=True) for index, row in df.iterrows() 
                if row["Question"] is not None and row["Rewrite Context"] is not None and row["Rewrite"] is not None]
    elif segment == "rationale":
        return [dsp.Example(question=row["Question"], context=_eval(row["Rationale Context"]), rationale = row["Rationale"], answer = row["Answer"], augmented=True) for index, row in df.iterrows()
                if row["Question"] is not None and row["Rationale Context"] is not None and row["Rationale"] is not None and row["Answer"] is not None]
    else:
        raise NotImplementedError()
    
def sample_balancedly(demos:list[Example], k):
    if len(demos) <= k or 'answer' not in demos[0]:
        return dsp.sample(demos, k=k)
    
    bins = {}
    for demo in demos: 
        if isinstance(demo.answer, list):
            answer = demo.answer[0]
        else:
            answer = demo.answer
        if answer not in bins:
            bins[answer] = []
        bins[answer].append(demo)
    mean_k_per_bin = k / len(bins)
    floored_k_per_bin = int(np.floor(mean_k_per_bin))
    remainders = list(range(len(bins)))
    rng = random.Random(dsp.settings.branch_idx)
    rng.shuffle(remainders)
    remainders = remainders[:k-floored_k_per_bin*len(bins)]
    
    sampled_demos = []
    for i, key in enumerate(bins.keys()):
        if i in remainders:
            k_per_bin = floored_k_per_bin+1
        else:
            k_per_bin = floored_k_per_bin
        demos_per_bin = bins[key]
        sampled_demos.extend(dsp.sample(demos_per_bin, k = k_per_bin))

    assert(len(sampled_demos)==k)
    rng.shuffle(sampled_demos)
    return sampled_demos

def transform_balancedly(df:DataFrame):
    grouped = df.groupby(['Answer'], as_index=False)
    return grouped.apply(lambda x: x.sample(n=int(grouped['Answer'].count().min().iloc[0]))).reset_index(drop=True).sample(frac=1).reset_index(drop=True)

