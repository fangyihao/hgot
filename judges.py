'''
Created on Aug. 10, 2023

@author: Yihao Fang
'''
import time
from dsp.primitives.predict import Completions
from dsp import Example
import dsp
from collections import Counter
from dsp.utils import normalize_text
from metrics import citation_recall, citation_precision
import re
from nltk import sent_tokenize
import numpy as np
from ordered_set import OrderedSet
import string
#import spacy
#nlp = spacy.load('en_core_web_sm')
def nli_reranker(query, passages, retrieval_weight=0.5, nli_weight=0.5):
    _nli = dsp.settings.nli
    passages_scores = []
    
    for i, passage in enumerate(passages):
        start = time.time()
        entail = _nli(passage.long_text, query)
        end  = time.time()
        
        print("-"*35 + (" NLI RERANKING %d "%i) + "-"*35)
        print("."*35 + " passage " + "."*35)
        print(passage.long_text)
        print("."*35 + " claim " + "."*35)
        print(query)
        print("."*35 + " entailment " + "."*35)
        print("True" if entail == 1 else "False")
        print("."*35 + " score (before reranking) " + "."*35)
        print(passage.score)
        print("."*35 + " score (after reranking) " + "."*35)
        print(retrieval_weight * passage.score + nli_weight * entail)
        print("."*35 + " elapsed time " + "."*35)
        print(end - start)
        
        passage.score = retrieval_weight * passage.score + nli_weight * entail
        passages_scores.append(passage.score)
        
    return passages_scores


def nli_electoral_college(example: Example, completions: Completions, ci = False):
    prediction_field = completions.template.fields[-1].output_variable
    rationale_field = completions.template.fields[-2].output_variable
    template = completions.template
    
    if not dsp.settings.lm:
        raise AssertionError("No LM is loaded.")

    normalized_to_original = {}
    
    predictions = []
    rationales = []
    for completion in completions:
        if prediction_field in completion:
            predictions.append(normalize_text(completion[prediction_field]))
        else:
            predictions.append("")
            
        if rationale_field in completion:
            rationales.append(completion[rationale_field])
        else:
            rationales.append("")
    

    for completion, prediction in zip(completions, predictions):
        if prediction not in normalized_to_original:
            normalized_to_original[prediction] = completion
            
            

    def evaluate_rationale(rationale, context, recall_weight=0.5):
        _nli = dsp.settings.nli
        # Preprocess
        q2p_dict = {}
        
        #doc = nlp(rationale)
        #sents = [sent.text.strip() for sent in doc.sents]
        sents = sent_tokenize(rationale)
        sents = [sent for sent in sents if sent not in string.punctuation]
        citation_frequency = np.array([0] * len(context))
        for sent in sents:
            cite_indexes = OrderedSet([int(r[1:])-1 for r in re.findall(r"\[\d+", sent)])
            sent = re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")
            
            for i, passage in enumerate(context):
                if _nli(passage, sent) == 1:
                    cite_indexes.add(i)
            
            q2p_dict[sent] = [context[cite_index] for cite_index in cite_indexes if cite_index < len(context)]
            
            for cite_index in cite_indexes:
                if cite_index < len(citation_frequency):
                    citation_frequency[cite_index]+=1
 
        stats = {}
        stats["citation_recall"] = citation_recall(q2p_dict)
        stats["citation_precision"] = citation_precision(q2p_dict)
        stats["citation_frequency"] = citation_frequency
        
        stats["weight"] = recall_weight * stats["citation_recall"] + (1-recall_weight) * stats["citation_precision"]
        return stats

    evaluated_predictions = [(prediction, evaluate_rationale(rationale, example.context)) for prediction, rationale in zip(predictions, rationales) if prediction]
    
    def weighted_topk(evaluated_predictions):
        prediction_to_weight = {}
        for prediction, stats in evaluated_predictions:
            if prediction not in prediction_to_weight:
                prediction_to_weight[prediction] = 0
            
            prediction_to_weight[prediction] += stats["weight"]
            
        return sorted(prediction_to_weight.items(), key=lambda item: item[1], reverse=True)
    

    topk = weighted_topk(evaluated_predictions)
    prediction, _ = topk[0]
    
    if ci: 
        _, scores = np.split(np.array(topk), 2, axis = 1)
        scores = np.reshape(scores, (-1,)).astype(np.float32, copy=False)
        if np.sum(scores) == 0:
            ci_score = None
        else:
            ci_score = scores[0] / np.sum(scores)
    else:
        ci_score = None
        
    citation_frequency = np.sum(np.array([stats["citation_frequency"] for _, stats in evaluated_predictions]),axis=0)
    
    normalized_citation_frequency = citation_frequency/np.sum(citation_frequency) if np.sum(citation_frequency) > 0 else citation_frequency

    #if ci and ci_score is not None:
    #    normalized_citation_frequency *= ci_score
    
    completion = normalized_to_original[prediction]

    dsp.settings.lm.history.append(
        {**dsp.settings.lm.history[-1], "topk": topk, "completions": [completion]}
    )
    
    return Completions([completion.copy(citation_frequency = normalized_citation_frequency, confidence = ci_score)], template=template,)
