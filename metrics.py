'''
Created on Sep. 6, 2023

@author: Yihao Fang
'''
from nli import _run_nli
import copy
from dsp.utils.metrics import EM, F1, nF1
from dsp.utils.metrics import HotPotF1 as HPF1
import numpy as np

class Metric:
    def __init__(self):
        self.result = []
    def evaluate(self, **kwargs):
        raise NotImplementedError()
    def average(self):
        r = np.mean(self.result)
        print(r)
        return r
    def clear(self):
        self.result = []
        
class OpenSQuADEM(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " EM " + "."*35)
        em = EM(prediction, answer)
        self.result.append(em)
        print(em)
    def average(self):
        print("."*35 + " EM " + "."*35)
        return super().average()
        
class OpenSQuADF1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = F1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
        
class HotPotEM(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " EM " + "."*35)
        em = EM(prediction, answer)
        self.result.append(em)
        print(em)
    def average(self):
        print("."*35 + " EM " + "."*35)
        return super().average()
        
class HotPotF1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = HPF1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
    
class QReCCF1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = F1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
    
class QReCCnF1(Metric):
    def evaluate(self, prediction, answer, history):
        print("."*35 + " nF1 " + "."*35)
        nf1 = nF1(" ".join(history), prediction, answer)
        self.result.append(nf1)
        print(nf1)
    def average(self):
        print("."*35 + " nF1 " + "."*35)
        return super().average()

class QueensEM(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " EM " + "."*35)
        em = EM(prediction, answer)
        self.result.append(em)
        print(em)
    def average(self):
        print("."*35 + " EM " + "."*35)
        return super().average()
    
class QueensF1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = F1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
    
class ElapsedTime(Metric):
    def evaluate(self, time):
        print("."*35 + " elapsed time " + "."*35)
        self.result.append(time)
        print(time)
    def average(self):
        print("."*35 + " elapsed time " + "."*35)
        return super().average()


def citation_recall(q2p_dict):
    
    entail = 0
    
    for question in q2p_dict:
        # calculate the recall score
        joint_entail = -1 # Undecided

        passages = q2p_dict[question]

        if len(passages) == 0:
            # No citations
            joint_entail = 0
        else:
            joint_passage = '\n'.join(passages)

        # If not directly rejected by citation format error, calculate the recall score
        if joint_entail == -1: 
            joint_entail = _run_nli(joint_passage, question)
            
        entail += joint_entail

    citation_rec = entail / len(q2p_dict)
    
    return citation_rec



def citation_precision(q2p_dict):
    
    entail_prec = 0
    total_citations = 0
    for question in q2p_dict:
        # calculate the recall score
        joint_entail = -1 # Undecided

        passages = q2p_dict[question]

        if len(passages) == 0:
            # No citations
            joint_entail = 0
        else:
            
            total_citations += len(passages)
            joint_passage = '\n'.join(passages)

        # If not directly rejected by citation format error, calculate the recall score
        if joint_entail == -1: 
            joint_entail = _run_nli(joint_passage, question)
            
            
        # calculate the precision score if applicable
        if joint_entail and len(passages) > 1:

            # Precision check: did the model cite any unnecessary documents?
            for passage in passages:
                # condition A
                nli_result = _run_nli(passage, question)

                # condition B
                if not nli_result:
                    subset_exclude = copy.deepcopy(passages)
                    subset_exclude.remove(passage)
                    joint_passage_exclude = '\n'.join(subset_exclude)
                    nli_result = _run_nli(joint_passage_exclude, question)
                    if nli_result:
                        flag = 0
                    else:
                        entail_prec += 1
                else:
                    entail_prec += 1
        else:
            entail_prec += joint_entail 

    citation_prec = entail_prec / total_citations if total_citations > 0 else 0
    
    return citation_prec
    
'''
def citation_quality(q2p_dict):
    entail = 0
    entail_prec = 0
    total_citations = 0
    for question in q2p_dict:
        # calculate the recall score
        joint_entail = -1 # Undecided

        passages = q2p_dict[question]

        if len(passages) == 0:
            # No citations
            joint_entail = 0
        else:
            
            total_citations += len(passages)
            joint_passage = '\n'.join(passages)

        # If not directly rejected by citation format error, calculate the recall score
        if joint_entail == -1: 
            joint_entail = _run_nli(joint_passage, question)
            
        entail += joint_entail


        # calculate the precision score if applicable
        if joint_entail and len(passages) > 1:

            # Precision check: did the model cite any unnecessary documents?
            for passage in passages:
                # condition A
                nli_result = _run_nli(passage, question)

                # condition B
                if not nli_result:
                    subset_exclude = copy.deepcopy(passages)
                    subset_exclude.remove(passage)
                    joint_passage_exclude = '\n'.join(subset_exclude)
                    nli_result = _run_nli(joint_passage_exclude, question)
                    if nli_result:
                        flag = 0
                    else:
                        entail_prec += 1
                else:
                    entail_prec += 1
        else:
            entail_prec += joint_entail 

    citation_rec = entail / len(q2p_dict)
    citation_prec = entail_prec / total_citations if total_citations > 0 else 0
    
    return citation_rec, citation_prec
'''   
