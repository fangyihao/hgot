'''
Created on Nov. 3, 2023

@author: Yihao Fang
'''
from datasets import load_dataset
dataset=load_dataset(r"hkust-nlp/felm",'wk')
print(dataset['test'][0])

print(len(dataset['test']))