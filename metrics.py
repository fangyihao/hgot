'''
Created on Sep. 6, 2023

@author: Yihao Fang
'''
import copy
import dsp
from dsp.utils.metrics import EM, F1, nF1
from dsp.utils.metrics import HotPotF1 as HPF1
import numpy as np
import re
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
    def __str__(self):
        return self.__class__.__name__
        
class OpenSQuADEM(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " EM " + "."*35)
        em = EM(prediction, answer)
        self.result.append(em)
        print(em)
    def average(self):
        print("."*35 + " EM " + "."*35)
        return super().average()
    def __str__(self):
        return "EM"
    
class OpenSQuADF1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = F1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
    def __str__(self):
        return "F1"
        
class HotPotEM(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " EM " + "."*35)
        em = EM(prediction, answer)
        self.result.append(em)
        print(em)
    def average(self):
        print("."*35 + " EM " + "."*35)
        return super().average()
    def __str__(self):
        return "EM"
    
class HotPotF1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = HPF1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
    def __str__(self):
        return "F1"
    
class QReCCF1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = F1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
    def __str__(self):
        return "F1"
        
class QReCCnF1(Metric):
    def evaluate(self, prediction, answer, history):
        print("."*35 + " nF1 " + "."*35)
        nf1 = nF1(" ".join(history), prediction, answer)
        self.result.append(nf1)
        print(nf1)
    def average(self):
        print("."*35 + " nF1 " + "."*35)
        return super().average()
    def __str__(self):
        return "nF1"

class FELMMetric(Metric):
    def consolidate(self, result):
        TP=0
        TN=0
        FP=0
        FN=0
    
        for item in result:
            if item=='TP':
                TP+=1
            elif item=='TN':
                TN+=1
            elif item=='FP':
                FP+=1
            elif item=='FN':
                FN+=1
        '''
        print("TP:", TP,
        "TN:", TN, 
        "FP:", FP, 
        "FN:", FN)
        '''
        return {
            'class 1': TP/(TP+FN) if TP+FN!=0 else None,
            'class 0': TN/(TN+FP) if TN+FP!=0 else None,
            'true num': TP + FN,
            'false num': TN + FP,
            'balanced': 0.5*(TP/(TP+FN)+TN/(TN+FP)) if TP+FN!=0 and TN+FP!=0 else None,
            'TN,TP,FN,FP':(TN,TP,FN,FP),
            'P':TN/(TN+FN) if TN+FN!=0 else None,
            'R':TN/(TN+FP) if TN+FP!=0 else None,
            'F1':2*(TN/(TN+FP))*(TN/(TN+FN))/(TN/(TN+FP)+TN/(TN+FN)) if (TN+FP!=0 and TN+FN!=0) else None,
        }
        
    def evaluate(self, prediction, answer):
        split_pred=[True] * len(answer)
        if 'ALL_CORRECT' not in prediction:
            prediction=[int(x) for x in re.findall(r'\d+',prediction) if int(x)<=len(answer)]
            for _ in prediction:
                split_pred[_-1]=False
        
        result = []
        for i in range(len(answer)):
            if split_pred[i] and answer[i]:
                #print("TP")
                result.append('TP')
            elif not split_pred[i] and not answer[i]:
                #print("TN")
                result.append('TN')
            elif split_pred[i] and not answer[i]:
                #print("FP")
                result.append('FP')
            elif not split_pred[i] and answer[i]:
                #print('FN')
                result.append('FN')
        '''
        print("split_pred:", split_pred)
        print("answer:", answer)
        print("result:", result)
        '''
        self.result.extend(result)
        return result

class FELMF1(FELMMetric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        result = super().evaluate(prediction, answer)
        f1 = self.consolidate(result)["F1"]
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        f1 = self.consolidate(self.result)["F1"]
        print(f1)
        return f1
    def __str__(self):
        return "F1"
    
class FELMBalAcc(FELMMetric):
    def evaluate(self, prediction, answer):
        print("."*35 + " Balanced Accuracy " + "."*35)
        result = super().evaluate(prediction, answer)
        balanced = self.consolidate(result)["balanced"]
        print(balanced)
    def average(self):
        print("."*35 + " Balanced Accuracy " + "."*35)
        balanced = self.consolidate(self.result)["balanced"]
        print(balanced)
        return balanced
    def __str__(self):
        return "Balanced Accuracy"    

class WysdomEM(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " EM " + "."*35)
        em = EM(prediction, answer)
        self.result.append(em)
        print(em)
    def average(self):
        print("."*35 + " EM " + "."*35)
        return super().average()
    def __str__(self):
        return "EM"
    
class WysdomF1(Metric):
    def evaluate(self, prediction, answer):
        print("."*35 + " F1 " + "."*35)
        f1 = F1(prediction, answer)
        self.result.append(f1)
        print(f1)
    def average(self):
        print("."*35 + " F1 " + "."*35)
        return super().average()
    def __str__(self):
        return "F1"
        
class ElapsedTime(Metric):
    def evaluate(self, time):
        print("."*35 + " elapsed time " + "."*35)
        self.result.append(time)
        print(time)
    def average(self):
        print("."*35 + " elapsed time " + "."*35)
        return super().average()
    def __str__(self):
        return "elapsed time"

def citation_recall(s2p_dict):
    _nli = dsp.settings.nli
    entail = 0
    
    for statement in s2p_dict:
        # calculate the recall score
        joint_entail = -1 # Undecided

        passages = s2p_dict[statement]

        if len(passages) == 0:
            # No citations
            joint_entail = 0
        else:
            joint_passage = '\n'.join(passages)

        # If not directly rejected by citation format error, calculate the recall score
        if joint_entail == -1: 
            joint_entail = _nli(joint_passage, statement)
            
        entail += joint_entail

    citation_rec = entail / len(s2p_dict)
    
    return citation_rec



def citation_precision(s2p_dict):
    _nli = dsp.settings.nli
    entail_prec = 0
    total_citations = 0
    for statement in s2p_dict:
        # calculate the recall score
        joint_entail = -1 # Undecided

        passages = s2p_dict[statement]

        if len(passages) == 0:
            # No citations
            joint_entail = 0
        else:
            
            total_citations += len(passages)
            joint_passage = '\n'.join(passages)

        # If not directly rejected by citation format error, calculate the recall score
        if joint_entail == -1: 
            joint_entail = _nli(joint_passage, statement)
            
            
        # calculate the precision score if applicable
        if joint_entail and len(passages) > 1:

            # Precision check: did the model cite any unnecessary documents?
            for passage in passages:
                # condition A
                nli_result = _nli(passage, statement)

                # condition B
                if not nli_result:
                    subset_exclude = copy.deepcopy(passages)
                    subset_exclude.remove(passage)
                    joint_passage_exclude = '\n'.join(subset_exclude)
                    nli_result = _nli(joint_passage_exclude, statement)
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
def citation_quality(s2p_dict):
    entail = 0
    entail_prec = 0
    total_citations = 0
    for statement in s2p_dict:
        # calculate the recall score
        joint_entail = -1 # Undecided

        passages = s2p_dict[statement]

        if len(passages) == 0:
            # No citations
            joint_entail = 0
        else:
            
            total_citations += len(passages)
            joint_passage = '\n'.join(passages)

        # If not directly rejected by citation format error, calculate the recall score
        if joint_entail == -1: 
            joint_entail = _run_nli(joint_passage, statement)
            
        entail += joint_entail


        # calculate the precision score if applicable
        if joint_entail and len(passages) > 1:

            # Precision check: did the model cite any unnecessary documents?
            for passage in passages:
                # condition A
                nli_result = _run_nli(passage, statement)

                # condition B
                if not nli_result:
                    subset_exclude = copy.deepcopy(passages)
                    subset_exclude.remove(passage)
                    joint_passage_exclude = '\n'.join(subset_exclude)
                    nli_result = _run_nli(joint_passage_exclude, statement)
                    if nli_result:
                        flag = 0
                    else:
                        entail_prec += 1
                else:
                    entail_prec += 1
        else:
            entail_prec += joint_entail 

    citation_rec = entail / len(s2p_dict)
    citation_prec = entail_prec / total_citations if total_citations > 0 else 0
    
    return citation_rec, citation_prec
'''   
