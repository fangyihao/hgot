'''
Created on Jan 15, 2025

@author: Yihao Fang
'''
import os
root_path = '.'
os.environ["DSP_CACHEDIR"] = os.path.join('../', 'hgot_cache')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

from functools import partial
from pipelines import GoT_QA

verbose = False
if verbose:
    from nli import _t5_nli_logged as _t5_nli
    from nli import _gpt_nli_logged as _gpt_nli
else:
    from nli import _t5_nli, _gpt_nli

from dsp.utils.metrics import EM, F1
from utils import df_to_dsp, df_to_dsp_augmented
from judges import nli_electoral_college
from run import init_langauge_model
from run import init_retrieval_model
from run import load_data

from flask import Flask, render_template, jsonify, request
import networkx as nx
import pandas as pd

language_model='gpt-3.5-turbo-1106'
retrieval_model='google'

init_langauge_model(language_model=language_model)
init_retrieval_model(retrieval_model=retrieval_model)

dataset = "hotpotqa-short"
method = "got-3+demos-sa-knn+cx+t5-nli-ec+ci+[0.3,0.35,0.35]+[0.3,0.6,0.1]"
    
train, dev, test = load_data(dataset)
train, dev, test = df_to_dsp(train), df_to_dsp(dev), df_to_dsp(test)

annot_dump = "log/temp/%s_%s_%s_%s_annot.csv"%(dataset, method, language_model, retrieval_model)
annot_log = "log/temp/%s_%s_%s_%s_annot.log"%(dataset, method, language_model, retrieval_model)

W_R = eval(method.split('+')[-1])
W_T = eval(method.split('+')[-2])
dsp.settings.configure(nli=_t5_nli)
dsp.settings.configure(electoral_college=partial(nli_electoral_college, ci=True, W=W_T))
method_func = GoT_QA(demo_flags="plan+rewrite+rationale", p_context=True, W=W_R, B=[0.9,0.7,0.9], annot_step=50, annot_selector=EM, annot_dump=annot_dump, annot_log=annot_log, annot_max_pri_rationale_demos=3, annot_min_cand_rationale_demos=128, annot_balance=False, demo_sel_func=dsp.knn)

def warmup():
    default_stdout = sys.stdout
    log_file = open("log/temp/%s_%s_%s_%s.log"%(dataset, method, language_model, retrieval_model),"w")
    sys.stdout = log_file
    
    df = pd.read_csv("data/MIRA/warmup.csv")
    for index, row in df.iterrows():
        question = row['Question']
        #question = test[0].question
        try:
            print("#"*10 + question + "#"*10)
            prediction = method_func(train, question)
            
            print("="*35 + " ANSWER " + "="*35)
            #print("."*35 + " prediction " + "."*35)
            print(prediction)
            #print("."*35 + " ground truth " + "."*35)
            #print(test[0].answer)
        except:
            raise
            #print("Erroneous question: ", question, file=sys.stderr)
            
    sys.stdout = default_stdout
    log_file.close()
    
warmup()


def chat(question):
    
    default_stdout = sys.stdout
    log_file = open("log/temp/%s_%s_%s_%s.log"%(dataset, method, language_model, retrieval_model),"w")
    sys.stdout = log_file
    
    #question = test[0].question
    
    print("#"*10 + question + "#"*10)
    prediction = method_func(train, question)
    
    print("="*35 + " ANSWER " + "="*35)
    #print("."*35 + " prediction " + "."*35)
    print(prediction)
    #print("."*35 + " ground truth " + "."*35)
    #print(test[0].answer)
    
    sys.stdout = default_stdout
    log_file.close()
    return prediction



def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=1.0, xcenter=0.5, depth=0):
    """
    Compute the hierarchical layout for a tree.
    Assumes that G is a directed tree (a DiGraph) with no cycles.
    
    Parameters:
        G (nx.DiGraph): The tree graph.
        root: The root node of the current branch.
        width (float): Horizontal space allocated for this branch.
        vert_gap (float): Gap between levels.
        vert_loc (float): Vertical location of the root.
        xcenter (float): Horizontal center of the root.
    
    Returns:
        dict: Mapping node -> (x, y) position.
    """
    pos = {root: (xcenter, vert_loc)}
    children = list(G.successors(root))
    children = [child for child in children if child.split(':')[0]!=root.split(':')[0]]
    if children:
        vert_sub_gap = np.array(range(-1000 * (depth+1),1000* (depth+1),2000* (depth+1)//len(children)))/10000
        random.shuffle(vert_sub_gap)
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for i, child in enumerate(children):
            nextx += dx
            pos.update(hierarchy_pos(G, child, width=dx, vert_gap=vert_gap + 0.1,
                                       vert_loc=vert_loc - vert_gap + vert_sub_gap[i], xcenter=nextx, depth=depth+1))
    return pos


app = Flask(__name__, template_folder='static', static_url_path='', static_folder='static')

@app.route('/api/rest/v1/chat', methods=['POST'])
def create_chat():
    req_json = request.get_json()
    question = req_json['content']
    #logger.info('%s - QUESTION: %s'%(request.remote_addr, question))
    answer = chat(question)
    #logger.info('%s - ANSWER: %s'%(request.remote_addr, answer))
    answer['confidence'] = f"{answer['confidence']*100}%"
    
    
    G = answer['graph_of_thoughts']
    pos = hierarchy_pos(G, 'L1:'+question, width=5.0, vert_gap=0.2, vert_loc=1.0, xcenter=0.5, depth=0)
    scale = 350
    nodes = []
    for node in G.nodes:
        x, y = pos[node]
        nodes.append({
            'data': {'id': node, 
                     'label': node, 
                     'question': nx.get_node_attributes(G, "question")[node] if node in nx.get_node_attributes(G, "question") else '',
                     'rationale': nx.get_node_attributes(G, "rationale")[node] if node in nx.get_node_attributes(G, "rationale") else '',
                     'context': nx.get_node_attributes(G, "context")[node] if node in nx.get_node_attributes(G, "context") else [],
                     'context_links': nx.get_node_attributes(G, "context_links")[node] if node in nx.get_node_attributes(G, "context_links") else [],
                     'answer': nx.get_node_attributes(G, "answer")[node] if node in nx.get_node_attributes(G, "answer") else '',
                     'confidence': f"{nx.get_node_attributes(G, 'confidence')[node]*100}%" if node in nx.get_node_attributes(G, 'confidence') else ''},
            'position': {'x': x * scale, 'y': (1 - y) * scale}
        })
    edges_list = []
    for source, target in G.edges:
        edges_list.append({
            'data': {'source': source, 'target': target}
        }) 

    
    answer['graph_of_thoughts'] = {'nodes': nodes, 'edges': edges_list}

    resp = jsonify(answer)
    
    return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9080, debug=True)

