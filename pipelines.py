'''
Created on Sep. 7, 2023

@author: Yihao Fang
'''
import sys
import dsp
import re
import openai
import time
import numpy as np
import functools
import random
import copy
import networkx as nx
from networkx.exception import NetworkXNoCycle
from collections import OrderedDict
from dsp.modules.cache_utils import CacheMemory, cache_turn_on
from dsp.utils import deduplicate
from utils import df_to_dsp_augmented, sample_balancedly, transform_balancedly
import pandas as pd
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
#from langchain.agents import load_tools
#from langchain.agents import initialize_agent
#from langchain.agents import AgentType
#from langchain.llms import OpenAI
#from langchain.chat_models import ChatOpenAI
from react import wikienv, wrappers
import requests
import json

from functools import partial

seed = 42
np.random.seed(seed)
random.seed(seed)

from sentence_transformers import SentenceTransformer
bert = SentenceTransformer('bert-base-nli-mean-tokens')

import spacy
nlp = spacy.load("en_core_sci_lg")

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
def paraphrase(passage, n, temperature=0.9, language_model='gpt-3.5-turbo'):
    start = time.time()
            
    if n == 1:
        instruction = "Please paraphrase the sentence below:\n"
    else:
        instruction = "Please generate %d paraphrases of the sentence below:\n"%n
       
    if n == 1:
        paraphrases = []   
        response = completions_with_backoff(
            model=language_model, 
            messages=[{"role": "user", "content": instruction + passage}],
            temperature=temperature
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
                max_tokens=1920,
                temperature=temperature
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

'''
class ReAct:
    def __init__(self):
        llm = OpenAI(temperature=0)
        tools = load_tools(["serpapi", "llm-math"], llm=llm)
        chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.agent = initialize_agent(
            tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
    def __call__(self, train, question):
        answer=self.agent.run(
            question
        )
        return {"answer":answer}
'''

class ReAct:
    def __init__(self, prompt = None, q_prefix = "Question:", q_transform = lambda x: x):
        self.env = wikienv.WikiEnv()
        self.env = wrappers.QAWrapper(self.env)
        
        if prompt is None:
            folder = 'react/prompts/'
            prompt_file = 'prompts_naive.json'
            with open(folder + prompt_file, 'r') as f:
                prompt_dict = json.load(f)
            
            webthink_examples = prompt_dict['webthink_simple6']
            instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
            (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
            (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
            (3) Finish[answer], which returns the answer and finishes the task.
            Here are some examples.
            """
            self.webthink_prompt = instruction + webthink_examples
        else:
            self.webthink_prompt = prompt
            
        self.q_prefix = q_prefix
        self.q_transform = q_transform
    
    def generate(self, prompt, stop=("\n",)):
        '''
        response = openai.Completion.create(
          model="text-davinci-002",
          prompt=prompt,
          temperature=0,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0.0,
          presence_penalty=0.0,
          stop=stop
        )
        return response["choices"][0]["text"]
        '''
        if not dsp.settings.lm:
            raise AssertionError("No LM is loaded.")
        completions = dsp.settings.lm(prompt=prompt, stop=stop)
        return completions[0]
    
    def step(self, env, action):
        attempts = 0
        while attempts < 10:
            try:
                return env.step(action)
            except requests.exceptions.Timeout:
                attempts += 1
    
    def webthink(self, question=None, prompt=None, to_print=True):
        question = self.env.reset(question=question, q_prefix = self.q_prefix)
        question = self.q_transform(question)
        if to_print:
            print(question)
        prompt += question + "\n"
        n_calls, n_badcalls = 0, 0
        for i in range(1, 8):
            n_calls += 1
            thought_action = self.generate(prompt + f"Thought {i}:", stop=(f"\nObservation {i}:",))
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except:
                print('ohh...', thought_action)
                n_badcalls += 1
                n_calls += 1
                thought = thought_action.strip().split('\n')[0]
                action = self.generate(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=(f"\n",)).strip()
            obs, r, done, info = self.step(self.env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')
            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prompt += step_str
            if to_print:
                print(step_str)
            if done:
                break
        if not done:
            obs, r, done, info = self.step(self.env, "finish[]")
        if to_print:
            print(info, '\n')
        info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
        return r, info

    def __call__(self, train, question):
        r, info = self.webthink(question=question, prompt=self.webthink_prompt, to_print=False)
        #print('-----------')
        #print()
        return {"answer":info['answer']}

class Vanilla_LM_QA:
    def __init__(self):
        Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
        Answer = dsp.Type(prefix="Answer:", desc="${a short factoid answer, often between 1 and 5 words}", format=dsp.format_answers)
        
        self.qa_template = dsp.Template(instructions="Answer questions with short factoid answers.", question=Question(), answer=Answer())
    
    def __call__(self, train, question):
        demos = dsp.sample(train, k=7)
        example = dsp.Example(question=question, demos=demos)
    
        example, completions = dsp.generate(self.qa_template)(example, stage='qa')
        #return completions.answer
        return {"answer":completions.answer}
    
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
    
        #return completions.answer
        return {"answer":completions.answer}
    
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
            desc="${a step-by-step deduction that identifies the correct response, which will be provided below%s}"%("" if not dsp.settings.electoral_college else """. Every statement in the "Rationale" section should be attributable to the passages provided in the "Context" section. """)
        )
        
        self.qa_template_with_CoT = dsp.Template(
            instructions=qa_template.instructions,
            context=Context(), question=Question(), rationale=Rationale(), answer=Answer()
        )
        
    @dsp.transformation
    def QA_predict(self, example: dsp.Example, sc=True):
        if sc:
            example, completions = dsp.generate(self.qa_template_with_CoT, n=20, temperature=0.7)(example, stage='qa')
            if dsp.settings.electoral_college:
                completions, citation_frequency = dsp.settings.electoral_college(example, completions)
            else:
                completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(self.qa_template_with_CoT)(example, stage='qa')
        
        return example.copy(answer=completions.answer) if not sc or not dsp.settings.electoral_college else example.copy(answer=completions.answer, citation_frequency=citation_frequency)
    
    def __call__(self, train, question):
        demos = dsp.sample(train, k=7)
        passages = dsp.retrieve(question, k=5)
        example = dsp.Example(question=question, context=passages, demos=demos)
        
        #return self.QA_predict(example).answer
        return {"answer":self.QA_predict(example).answer}
    
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
            desc="${a step-by-step deduction that identifies the correct response, which will be provided below%s}"%("" if not dsp.settings.electoral_college else """. Every statement in the "Rationale" section should be attributable to the passages provided in the "Context" section. """)
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
            if dsp.settings.electoral_college:
                completions, citation_frequency = dsp.settings.electoral_college(example, completions)
            else:
                completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(self.qa_template_with_CoT)(example, stage='qa')
        
        return example.copy(answer=completions.answer) if not sc or not dsp.settings.electoral_college else example.copy(answer=completions.answer, citation_frequency=citation_frequency)
    
    def __call__(self, train, question):
        demos = dsp.sample(train, k=7)
        x = dsp.Example(question=question, demos=demos)
        
        x = self.multihop_search_v1(x)
        x = self.QA_predict(x, sc=False)
    
        #return x.answer
        return {"answer":x.answer}


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
        #return x.answer
        return {"answer":x.answer}


class GoT_QA:

    def __init__(self, demo_flags = None, p_context: bool = True, retrieve_ensemble_n = None, depth = 3, **kwargs):
        self.EDGE_PATTERN = r'\s*([Ss]tep [0-9]+)\s*->\s*([Ss]tep [0-9]+)\s*'
        self.NODE_PATTERN = r'\s*([Ss]tep [0-9]+):\s*(.*)'
        
        self.demo_flags = demo_flags
        self.p_context = p_context
        self.retrieve_ensemble_n = retrieve_ensemble_n
        self.depth = depth
        self.annotator = None
        
        if "demos" in kwargs:
            self.demos = kwargs["demos"]
        else:
            self.demos = None
        
        if "demo_sel_func" in kwargs:
            self.demo_sel_func = kwargs["demo_sel_func"]
        else:
            self.demo_sel_func = sample_balancedly #dsp.sample
        
        if "annot_selector" in kwargs:
            self.annot_selector = kwargs["annot_selector"]
        else:
            self.annot_selector = None
            
        if "annot_dump" in kwargs:
            self.annot_dump = kwargs["annot_dump"]
        else:
            self.annot_dump = None
            
        if "annot_log" in kwargs:
            self.annot_log = kwargs["annot_log"]
        else:
            self.annot_log = None
        
        if "annot_step" in kwargs:
            self.annot_step = kwargs["annot_step"]
        else:
            self.annot_step = 50
            
        if "annot_max_pri_plan_demos" in kwargs:
            self.MAX_PRI_PLAN_DEMOS = kwargs["annot_max_pri_plan_demos"]
        else:
            self.MAX_PRI_PLAN_DEMOS = 2
        
        if "annot_max_pri_rewrite_demos" in kwargs:
            self.MAX_PRI_REWRITE_DEMOS = kwargs["annot_max_pri_rewrite_demos"]
        else:
            self.MAX_PRI_REWRITE_DEMOS = 2
        
        if "annot_max_pri_rationale_demos" in kwargs:
            self.MAX_PRI_RATIONALE_DEMOS = kwargs["annot_max_pri_rationale_demos"]
        else:
            self.MAX_PRI_RATIONALE_DEMOS = 2
        
        if "annot_min_cand_plan_demos" in kwargs:
            self.MIN_CAND_PLAN_DEMOS = kwargs["annot_min_cand_plan_demos"]
        else:
            self.MIN_CAND_PLAN_DEMOS = 32
        
        if "annot_min_cand_rewrite_demos" in kwargs:
            self.MIN_CAND_REWRITE_DEMOS = kwargs["annot_min_cand_rewrite_demos"]
        else:
            self.MIN_CAND_REWRITE_DEMOS = 16
        
        if "annot_min_cand_rationale_demos" in kwargs:
            self.MIN_CAND_RATIONALE_DEMOS = kwargs["annot_min_cand_rationale_demos"]
        else:
            self.MIN_CAND_RATIONALE_DEMOS = 32
        
        if "annot_balance" in kwargs:
            self.annot_balance = kwargs["annot_balance"]
        else:
            self.annot_balance = False
        
        if "B" in kwargs:
            self.B = kwargs["B"]
        else:
            self.B = [0.9,0.7,0.9]
            
        if dsp.settings.electoral_college:
            self.W = kwargs["W"]

        Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
        Answer = dsp.Type(prefix="Answer:", desc="${a short factoid answer, often between 1 and 5 words}", format=dsp.format_answers)
        
        qa_template = dsp.Template(instructions="Answer questions with short factoid answers.", question=Question(), answer=Answer())
    
        '''
        Context = dsp.Type(
            prefix="Context:\n",
            desc="${sources that may contain relevant content}",
            format=dsp.passages2text
        )
        '''
        Context = dsp.Type(
            prefix="Context:\n",
            desc="${sources that may contain relevant content. e.g., [1] Passage 1. [2] Passage 2. [3] Passage 3.}",
            format=dsp.passages2text
        )
        
        '''
        Rationale = dsp.Type(
            prefix="Rationale: Let's think step by step.",
            desc="${a step-by-step deduction that identifies the correct response, which will be provided below%s}"%("" if not dsp.settings.electoral_college else """. Every statement in the "Rationale" section should be attributable to the passages provided in the "Context" section. """)
        )
        '''
        Rationale = dsp.Type(
            prefix="Rationale: Let's think step by step.",
            desc="""${a step-by-step deduction that identifies the correct response, which will be provided below. Every statement in the "Rationale" section should be attributable to the passages provided in the "Context" section. e.g., ...[1][2].}"""
        )
        
        self.qa_template_with_CoT = dsp.Template(
            instructions=qa_template.instructions,
            context=Context(), question=Question(), rationale=Rationale(), answer=Answer()
        )
        
        Hints = dsp.Type(
            prefix="Hints:\n",
            desc="${answers to sub-questions that may contain relevant content}",
            format=dsp.hints2text
        )
        
        self.qa_template_with_CoT_n_hints = dsp.Template(
            instructions=qa_template.instructions,
            hints = Hints(), context=Context(), question=Question(), rationale=Rationale(), answer=Answer()
        )
        '''
        Plan = dsp.Type(
            prefix="Plan:\n",
            desc="Step 1: ${a standalone search question} Step 2: ${a standalone search question} ... Step n: ${a standalone search question}"
        )
        '''
        Plan = dsp.Type(
            prefix="Plan:\n",
            desc="Step 1: ${a standalone search question. e.g., ...?} Step 2: ${a standalone search question. e.g., ...?} ... Step n: ${a standalone search question. e.g., ...?}"
        )
        '''
        Plan = dsp.Type(
            prefix="Plan:\n",
            desc="Step 1: ${a more generic search question} Step 2: ${a more specific search question} ... Step n: ${a more specific search question}"
        )
        '''
        '''
        Dependencies = dsp.Type(
            prefix="Dependencies: ",
            desc="${interdependencies among multiple steps}"
        )
        '''
        Dependencies = dsp.Type(
            prefix="Dependencies: ",
            desc="${interdependencies among multiple steps. e.g., Step ... depends on Step ... .}"
        )
        
        self.plan_template = dsp.Template(
            instructions="Sketch a plan to answer the following question with the provided context. List only the essential steps which can be answered by search engines. Express each step as a standalone search question. Highlight interdependencies if any. Higher number steps can depend on lower number steps, while the reverse is not possible.",
            context=Context(), question=Question(), plan = Plan(), dependencies = Dependencies()
        )
        
        self.plan_template_wo_cx = dsp.Template(
            instructions="Sketch a plan to answer the following question. List only the essential steps which can be answered by search engines. Express each step as a standalone search question. Highlight interdependencies if any. Higher number steps can depend on lower number steps, while the reverse is not possible.",
            question=Question(), plan = Plan(), dependencies = Dependencies()
        )
        '''
        self.plan_template = dsp.Template(
            instructions="Sketch a plan to answer the following question with the provided context. List only the essential steps which can be answered by search engines. Express each step as a standalone search question. Step 1 serves as a more generic question. Subsequent steps are more specific questions. Highlight interdependencies if any. Higher number steps can depend on lower number steps, while the reverse is not possible.",
            context=Context(), question=Question(), plan = Plan(), dependencies = Dependencies()
        )
        
        self.plan_template_wo_cx = dsp.Template(
            instructions="Sketch a plan to answer the following question. List only the essential steps which can be answered by search engines. Express each step as a standalone search question. Step 1 serves as a more generic question. Subsequent steps are more specific questions. Highlight interdependencies if any. Higher number steps can depend on lower number steps, while the reverse is not possible.",
            question=Question(), plan = Plan(), dependencies = Dependencies()
        )
        '''
        Rewrite_Context = dsp.Type(
            prefix="Context:\n",
            desc="${previous questions and answers}"
        )
        
        Rewrite = dsp.Type(
            prefix="Rewrite: ",
            desc="${the last question after the rewrite}"
        )
        
        self.rewrite_template = dsp.Template(
            instructions="Rewrite the last question in a standalone manner by giving the answers to previous questions. Do not consider answers that were not specified. Only show the last question after the rewrite.",
            rewrite_context=Rewrite_Context(), rewrite = Rewrite()
        )
        
        Descriptions = dsp.Type(
            prefix="Descriptions: ",
            desc="${descriptions of dependencies}"
        )
        '''
        Dependencies_2 = dsp.Type(
            prefix="Dependencies: ",
            desc="${e.g. If Step 2 depends on Step 1, then write Step 1 -> Step 2; If Step 2 and Step 3 depend on Step 1, then write Step 1 -> (Step 2 and Step 3); If Step 3 depends on Step 1 and Step 2, then write (Step 1 and Step 2) -> Step 3}"
        )
        '''
        Dependencies_2 = dsp.Type(
            prefix="Dependencies: ",
            desc="${e.g., If Step 2 depends on Step 1, then write Step 1 -> Step 2; If Step 2 and Step 3 depend on Step 1, then write Step 1 -> (Step 2 and Step 3); If Step 3 depends on Step 1 and Step 2, then write (Step 1 and Step 2) -> Step 3}"
        )
        
        self.formalization_template = dsp.Template(
            instructions="Express the dependencies in formal language by giving the descriptions below.",
            descriptions=Descriptions(), dependencies = Dependencies_2()
        )
        
        self.reflection_template = dsp.Template(
            instructions="Highlight interdependencies among the steps below if any. Higher number steps can depend on lower number steps, while the reverse is not possible.",
            plan = Plan(), dependencies = Dependencies()
        )
        
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
    def predict(self, example: dsp.Example, sc=True, stage='qa', hints = False):
        default_demos = example.demos
        if self.demos and "rationale" in self.demos:
            example.demos = self.interpret_demos(self.demos["rationale"], example)
        
        qa_template = self.qa_template_with_CoT_n_hints if stage == 'qa' and hints else self.qa_template_with_CoT
        
        if sc:
            example, completions = dsp.generate(qa_template, n=20, temperature=0.7)(example, stage=stage)
            if dsp.settings.electoral_college:
                completions = dsp.settings.electoral_college(example, completions)
            else:
                completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(qa_template)(example, stage=stage)
            
        if self.annotator is not None and stage=='qa':
            self.annotator._set_value(len(self.annotator)-1, 'Rationale Context', copy.deepcopy(example.context))
            self.annotator._set_value(len(self.annotator)-1, 'Rationale', completions.rationale)
            self.annotator._set_value(len(self.annotator)-1, 'Answer', completions.answer)

        if dsp.settings.electoral_college:
            # TODO: optimize weights using derivative-free optimization
            def rerank(passages, scores, citation_frequency, confidence, W = [0.3, 0.6, 0.1]):
                assert (len(passages) == len(scores) and len(passages) == len(citation_frequency))
                if len(passages) == 0:
                    return passages, scores
                
                if confidence is not None:
                    scores = list(np.matmul(np.stack((scores, citation_frequency, [confidence] * len(passages)), axis=1), W))
                else:
                    scores = list(np.matmul(np.stack((scores, citation_frequency), axis=1), [W[0], sum(W[1:])]))

                passages, scores = np.split(np.array([[passage, score] for passage, score in sorted(zip(passages, scores), key=lambda item: item[1], reverse=True)]), 2, axis=1)
                passages = list(np.reshape(passages, (-1,)).astype(str, copy=False))
                scores = list(np.reshape(scores, (-1,)).astype(np.float32, copy=False))
                return passages, scores
            
            example.context, example.context_scores = rerank(example.context, example.context_scores, completions.citation_frequency, completions.confidence, W=self.W)
            confidence = completions.confidence
        else:
            confidence = None
            
        print("-"*35 + " ANSWER " + "-"*35)
        print(completions.answer)    
        print("-"*35 + " CONFIDENCE " + "-"*35)
        print(confidence)
        if dsp.settings.electoral_college:
            print("-"*35 + " CITATION FREQUENCY " + "-"*35)
            print(completions.citation_frequency)
        print("-"*35 + " PASSAGES WITH SCORES " + "-"*35)
        for passage, score in zip(example.context, example.context_scores):
            print(passage + " SCORE: " + str(score))
        
        return example.copy(answer=completions.answer, rationale = completions.rationale, confidence = confidence, demos = default_demos)

    
    @dsp.transformation
    def search(self, example: dsp.Example, depth, k=3) -> dsp.Example:
        
        def assess_hint_quality(hint_answer):
            bad_keywords=["not available","not specified","not provided","unavailable","not at hand","absent","not on hand","inaccessible","not up for grabs","out of reach","currently unattainable","not obtainable","unspecified","not detailed","not mentioned","not designated","not identified","not indicated","not set out","not defined","not listed","not supplied","unprovided","not given","absent","not furnished","not delivered","omitted","lacking","not included","not presented", "does not provide","fails to offer","doesn't supply","lacks provision of","omits giving","withholds","denies provision","doesn't present","does not make available","is devoid of providing","neglects to give"]
            quality = True # good quality
            for keyword in bad_keywords:
                if keyword in hint_answer.lower():
                    quality = False
            return quality
        
        if self.retrieve_ensemble_n:
            def retrieve_ensemble(query: str, k: int, return_dict:bool = False) -> list[str]:
                assert self.retrieve_ensemble_n >= 1
                
                if self.retrieve_ensemble_n > 1:
                    paraphrases = paraphrase(query, self.retrieve_ensemble_n-1)
                    return self.retrieve_ensemble([query]+paraphrases, k=k, return_dict=return_dict)
                elif self.retrieve_ensemble_n == 1:
                    return self.retrieve_ensemble([query], k=k, return_dict=return_dict)

        steps = self.find_steps(example.plan)

        _, completions = dsp.generate(self.formalization_template)(dsp.Example(descriptions=example.dependencies, demos=example.demos), stage='formalization')
        
        example = example.copy(dependencies=completions.dependencies)

        dependencies = self.format_dependencies(example.dependencies)
        
        if self.annotator is not None:
            if self.annotator._get_value(len(self.annotator)-1, 'Dependencies') is None:
                annot_dependencies = []
                for dependency in dependencies:
                    mo = re.match(self.EDGE_PATTERN, dependency)
                    if mo:
                        u = mo.group(1)
                        v = mo.group(2)
                        annot_dependencies.append("%s depends on %s."%(v, u))
                if len(annot_dependencies) == 0:
                    annot_dependencies = "None"
                else:
                    annot_dependencies = " ".join(annot_dependencies)
                self.annotator._set_value(len(self.annotator)-1, 'Dependencies', annot_dependencies)
        
        
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
        
        rev_G = G.reverse(copy=True)
        answers = OrderedDict()
        passages = OrderedDict()
        scores = OrderedDict()
        confidences = OrderedDict()
        for u in nx.topological_sort(G):
            if G.in_degree(u) == 0:
                print("~"*35 + u.capitalize() + "~"*35)

                if self.retrieve_ensemble_n:
                    passages_w_scores = retrieve_ensemble(self.find_step_question(questions[u]), k=k, return_dict = True)
                else:
                    passages_w_scores = self.retrieve(self.find_step_question(questions[u]), k=k, return_dict = True)
                passages[u], scores[u] = passages_w_scores["text"], passages_w_scores["score"]
                    
                sub_example = dsp.Example(question=self.find_step_question(questions[u]), demos=example.demos, context=passages[u], context_scores=scores[u])    
                #completions = self.predict(sub_example, stage='sub_qa')
                completions = self.predict_plan_search_predict(sub_example, depth=depth, stage='sub_qa')
                answers[u], passages[u], scores[u], confidences[u] = completions.answer, completions.context, completions.context_scores, completions.confidence

            else:
                all_rev_neighbors = True
                rewrite_context = []
                for v in rev_G.neighbors(u):
                    if v in answers:
                        rewrite_context.extend([questions[v], "ANSWER: " + answers[v] + "."])
                    else:
                        all_rev_neighbors = False
                if all_rev_neighbors:
                    print("~"*35 + u.capitalize() + "~"*35)
                    rewrite_context.append(questions[u])
                    
                    if self.annotator is not None:
                        if self.annotator._get_value(len(self.annotator)-1, 'Rewrite Context') is None:
                            self.annotator._set_value(len(self.annotator)-1, 'Rewrite Context', ' '.join(rewrite_context))
                    
                    if self.demos and "rewrite" in self.demos:
                        _, completions = dsp.generate(self.rewrite_template)(dsp.Example(rewrite_context=' '.join(rewrite_context), demos=self.interpret_demos(self.demos["rewrite"], example)), stage='rewrite')
                    else:
                        _, completions = dsp.generate(self.rewrite_template)(dsp.Example(rewrite_context=' '.join(rewrite_context), demos=example.demos), stage='rewrite')
                    rewrite = completions.rewrite
                    
                    if self.annotator is not None:
                        if self.annotator._get_value(len(self.annotator)-1, 'Rewrite') is None:
                            self.annotator._set_value(len(self.annotator)-1, 'Rewrite', rewrite)
                    
                    if not rewrite.lower().startswith("step"):
                        rewrite = u.capitalize() + ": " + rewrite
                    questions[u] = rewrite

                    if self.retrieve_ensemble_n:
                        passages_w_scores = retrieve_ensemble(self.find_step_question(rewrite), k=k, return_dict = True)
                    else:
                        passages_w_scores = self.retrieve(self.find_step_question(rewrite), k=k, return_dict = True)
                    passages[u], scores[u] = passages_w_scores["text"], passages_w_scores["score"]
                        
                    sub_example = dsp.Example(question=self.find_step_question(rewrite), demos=example.demos, context=passages[u], context_scores=scores[u])
                    #completions = self.predict(sub_example, stage='sub_qa')
                    completions = self.predict_plan_search_predict(sub_example, depth=depth, stage='sub_qa')
                    answers[u], passages[u], scores[u], confidences[u] = completions.answer, completions.context, completions.context_scores, completions.confidence 

        assert len(answers) == len(G.nodes)     
        
        # verbose
        retrieval_history=dict([(self.find_step_question(questions[u]), passages[u]) for u in questions])
        retrieval_history[example.question] = copy.deepcopy(example.context)
        
        # TODO: optimize weights using derivative-free optimization
        # weights = [0.8] + [1]*(len(questions))
        weights = [self.B[2]] + [1]*(len(questions))
        repeats = [len(example.context)]
        for u in questions:
            example.context.extend(passages[u])
            example.context_scores.extend(scores[u])
            repeats.append(len(passages[u]))
        weights = np.concatenate([[weight]*repeat for weight, repeat in zip(weights, repeats)])
        example.context_scores = list(np.multiply(example.context_scores, weights))
            
        print("-"*35 + " PASSAGES WITH SCORES " + "-"*35)
        for passage, score in zip(example.context, example.context_scores):
            print(passage + " SCORE: " + str(score))
        
        def topk(X, Y, k):
            if len(X) == 0 and len(Y) == 0:
                return X, Y
            
            assert(len(X)==len(Y) and len(X)>0)
            X_type = type(X[0])
            Y_type = type(Y[0])
            X, Y = np.split(np.array([[x, y] for x, y in sorted(zip(X, Y), key=lambda item: item[1], reverse=True)]), 2, axis=1)
            X = list(np.reshape(X, (-1,)).astype(X_type, copy=False))
            Y = list(np.reshape(Y, (-1,)).astype(Y_type, copy=False))
            return X[:k], Y[:k]
        #example.context, example.context_scores = np.split(np.array([[passage, score] for passage, score in sorted(zip(example.context, example.context_scores), key=lambda item: item[1], reverse=True)]), 2, axis=1)
        #example.context = list(np.reshape(example.context, (-1,)).astype(str, copy=False))
        #example.context_scores = list(np.reshape(example.context_scores, (-1,)).astype(np.float32, copy=False))
        #example.context = example.context[:int(len(example.context)*0.6)]
        #example.context_scores = example.context_scores[:int(len(example.context_scores)*0.6)]
        
        # Miller, G. A. (1956). "The magical number seven, plus or minus two: Some limits on our capacity for processing information". Psychological Review. 63 (2): 81–97.
        clip = 7
        example.context, example.context_scores = topk(example.context, example.context_scores, clip)
        # The passing percentage is set to 60% aligning to most of the academic grading systems
        # passing_percentage = 0.6
        # example.context, example.context_scores = topk(example.context, example.context_scores, min([int(len(example.context)*passing_percentage), clip]))
        
        hints = []
        print("-"*35 + " STEPS WITH ANSWERS " + "-"*35)
        for u in questions:
            print(questions[u] + " ANSWER: " + answers[u] + " CONFIDENCE: " + str(confidences[u]))
            if assess_hint_quality(answers[u]):
                hints.append(self.find_step_question(questions[u]) + " ANSWER: " + answers[u] + " CONFIDENCE: " + str(confidences[u]))
                
        if len(hints) == 0:
            hints.append('Refer to "Context".')

        return example.copy(hints=hints, retrieval_history=retrieval_history)
    
    
    @dsp.transformation
    def plan(self, example: dsp.Example, self_reflect=True, shortlist=False) -> dsp.Example:
        def extract_plan(plan):
            return re.sub(r"(.+)Context:.+", r"\1", plan, flags=re.DOTALL)
    
        default_demos = example.demos
        default_context = example.context
        default_context_scores = example.context_scores
        
        if self.demos and "plan" in self.demos:
            example.demos = self.interpret_demos(self.demos["plan"], example)
        
        if self.p_context == True:
            filtered_context = []
            filtered_context_scores = []
            for passage, score in zip(example.context, example.context_scores):
                #if score > 0.7:
                if score > self.B[0]:
                    filtered_context.append(passage)
                    filtered_context_scores.append(score)
            if len(filtered_context) > 0:
                example.context = filtered_context
                example.context_scores = filtered_context_scores
                template = self.plan_template
            else:
                template = self.plan_template_wo_cx
        else:
            template = self.plan_template_wo_cx
        
        # shortlist is currently an experimental feature
        if shortlist:
            from dsp.primitives.predict import Completions
            example, completions = dsp.generate(template, n=30, temperature=0.9)(example, stage='plan')
            plan_field = completions.template.fields[-2].output_variable
            bad_keywords = ["other","different","additional","subsequent","another"]
            
            completions = Completions([completion for completion in completions if plan_field in completion and all(bad_keyword not in word_tokenize(completion[plan_field]) for bad_keyword in bad_keywords)], 
                                      template=completions.template,)
            completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(template)(example, stage='plan')

        plan = extract_plan(completions.plan)
        if self_reflect:
            _, completions = dsp.generate(self.reflection_template)(dsp.Example(plan=plan, demos=example.demos), stage='plan')
        
        if self.annotator is not None:
            if self.p_context == True:
                if self.annotator._get_value(len(self.annotator)-1, 'Plan Context') is None:
                    self.annotator._set_value(len(self.annotator)-1, 'Plan Context', copy.deepcopy(example.context))
            if self.annotator._get_value(len(self.annotator)-1, 'Plan') is None:
                self.annotator._set_value(len(self.annotator)-1, 'Plan', plan)
            #if self.annotator._get_value(len(self.annotator)-1, 'Dependencies') is None:
            #    self.annotator._set_value(len(self.annotator)-1, 'Dependencies', completions.dependencies)    
        
        return example.copy(plan = plan, dependencies = completions.dependencies, demos = default_demos, context = default_context, context_scores=default_context_scores)
    
    
    @dsp.transformation
    def predict_plan_search_predict(self, example: dsp.Example, depth=0, stage = 'qa'):
        print("="*35 + " PREDICT (%d-0) "%(depth) + "="*35)
        example_0 = self.predict(example, stage=stage)
        
        example_1 = dsp.Example(question=example_0.question, demos=example_0.demos, context=example_0.context, context_scores = example_0.context_scores)
        print("="*35 + " PLAN (%d) "%(depth) + "="*35)
        example_1 = self.plan(example_1)
        
        def proceed(question, plan, sim_thr = self.B[1]):
            steps = self.find_steps(plan)
            if len(steps) == 1:
                sents = [
                    question,
                    self.find_step_question(steps[0])
                ]
                sent_embeddings = bert.encode(sents)
                sim = np.squeeze(cosine_similarity([sent_embeddings[0]], [sent_embeddings[1]]))
                print("-"*35 + " SIMILARITY " + "-"*35)
                print(sim)
                return sim < sim_thr
            elif len(steps) > 1:
                return True
            else: 
                return False
            
        if proceed(example_1.question, example_1.plan) and depth < self.depth - 1:
            example = example_1
            print("="*35 + " SEARCH (%d) "%(depth+1) + "="*35)
            example = self.search(example, depth+1)
            print("="*35 + " PREDICT (%d-1) "%(depth) + "="*35)
            example = self.predict(example, stage=stage)
        else:
            example = example_0
        return example
    
    def interpret_demos(self, demos, example):
        if isinstance(demos, list):
            return demos
        else:
            return demos(example)
    
    def retrieve(self, query:str, k:int, return_dict:bool=False)->list[str]:
        passages = dsp.retrieve(query, k, return_dict)
        if len(passages['text']) == 0: 
            doc = nlp(query)
            keyword_query = ", ".join([str(ent) for ent in doc.ents])
            passages = dsp.retrieve(keyword_query, k, return_dict)
        return passages
    
    def retrieve_ensemble(self, queries:list[str], k:int, by_prob:bool=False, return_dict:bool=False)->list[str]:
        passages = dsp.retrieveEnsemble(queries=queries, k=k, by_prob=by_prob, return_dict=return_dict)
        if len(passages['text']) == 0: 
            keyword_queries = []
            for query in queries:
                doc = nlp(query)
                keyword_queries.append(", ".join([str(ent) for ent in doc.ents]))
            passages = dsp.retrieveEnsemble(queries=keyword_queries, k=k, by_prob=by_prob, return_dict=return_dict)
        return passages
    
    def start(self, train, question):
        passages_w_scores = self.retrieve(question, k=3,  return_dict = True)
        
        if (self.demos is None or "default" not in self.demos):
            if self.demos is None:
                self.demos = {}
            if "sample" in self.demo_sel_func.__name__:
                self.demos["default"] = self.demo_sel_func(train, k = 7)
            elif "knn" in self.demo_sel_func.__name__:
                self.demos["default"] = partial(self.demo_sel_func(train), k = 7)
            else:
                raise NotImplementedError()
        
        example = dsp.Example(question=question, demos=None, context=passages_w_scores["text"], context_scores = passages_w_scores["score"])
        
        example.demos = self.interpret_demos(self.demos["default"], example)
        
        example = self.predict_plan_search_predict(example)
        
        return {"answer":example.answer, "confidence": example.confidence}
    
    def __call__(self, train, question):
        self.annotate(train)
        return self.start(train, question)
    
    def annotate(self, train):
        
        if (self.demo_flags is not None 
            and ("plan" in self.demo_flags 
                 and self.demos is not None and "plan" in self.demos #and len(self.demos["plan"]) > 0
                 or "plan" not in self.demo_flags) 
            and ("rewrite" in self.demo_flags 
                 and self.demos is not None and "rewrite" in self.demos #and len(self.demos["rewrite"]) > 0
                 or "rewrite" not in self.demo_flags)
            and ("rationale" in self.demo_flags 
                 and self.demos is not None and "rationale" in self.demos #and len(self.demos["rationale"]) > 0
                 or "rationale" not in self.demo_flags)
            or self.demo_flags is None):
            return
        
        if self.annot_log:
            default_stdout = sys.stdout
            log_file = open(self.annot_log,"w")
            sys.stdout = log_file
        
        default_electoral_college = dsp.settings.electoral_college
        dsp.settings.configure(electoral_college=None)
        default_depth = self.depth
        self.depth = 2
        default_B = self.B
        self.B = copy.deepcopy(self.B)
        self.B[0] = 0
        
        shuffled_train = [example.copy() for example in train]
        rng = random.Random(dsp.settings.branch_idx)
        rng.shuffle(shuffled_train)
        train, test = shuffled_train[:len(shuffled_train)//10], shuffled_train[len(shuffled_train)//10:]
        #test = dsp.sample(test, k=len(test))
        
        annot_columns = ['Question', 'Plan Context', 'Plan', 'Dependencies', 'Rewrite Context', 'Rewrite', 'Rationale Context', 'Rationale', 'Answer', 'GT Answer']
        _annotator = pd.DataFrame(columns=annot_columns)
        self.demos=None
        self.annotator = pd.DataFrame(columns=annot_columns)
        annot_examples = None
        for i, example in enumerate(test):
            print("#"*10 + example.question + "#"*10)
            self.annotator.loc[len(self.annotator)] = [example.question,None, None, None, None, None, None, None, None, example.answer]
            self.start(train, example.question)
            
            if ((i % self.annot_step) == self.annot_step - 1) or i == len(test) - 1:
                
                #self.annotator["Dependencies"] = ["None" if "no interdependencies" in dependencies else dependencies for dependencies in self.annotator["Dependencies"].values]
                self.annotator["Dependencies"] = ["None" if dependencies is None else dependencies for dependencies in self.annotator["Dependencies"].values]
                
                self.annotator["Rationale"] = [re.sub(r"((\[[0-9]+\])+)\s*([^a-z\[\.\s][^\[\.]*)\.[\n\s]", r"\3 \1. ", rationale) for rationale in self.annotator["Rationale"].values]
                self.annotator["Rationale"] = [re.sub(r"((\[[0-9]+\])+)\s*([^a-z\[\.\s][^\[\.]*)\.$", r"\3 \1.", rationale) for rationale in self.annotator["Rationale"].values]
                self.annotator["Rationale"] = [rationale.replace('\n', ' ') for rationale in self.annotator["Rationale"].values]
                
                self.annotator["Pass"] = [self.annot_selector(prediction, answer) for prediction, answer in zip(self.annotator['Answer'].values, self.annotator['GT Answer'].values)]
                
                #self.annotator["Pass"] &= [all([step.endswith('?') and re.search(r"[Ii]s the query.+accurately addressed by the response.+", step, flags=re.DOTALL) is None for step in self.find_steps(plan)]) for plan in self.annotator['Plan'].values]
                self.annotator["Pass"] &= [all([step.endswith('?') for step in self.find_steps(plan)]) for plan in self.annotator['Plan'].values]
                
                mean_step_lens = pd.Series([np.mean([len(step) for step in self.find_steps(plan)]) for plan in self.annotator['Plan'].values])
                
                # IQR
                Q1 = np.percentile(mean_step_lens, 25, method='midpoint')
                Q3 = np.percentile(mean_step_lens, 75, method='midpoint')
                IQR = Q3 - Q1
                self.annotator["Pass"] &= [np.mean([len(step) for step in self.find_steps(plan)]) < Q3 + 1.5*IQR for plan in self.annotator['Plan'].values]
                
                self.annotator["Pass"] &= [re.match(r"None|((\s*([Ss]tep [0-9]+) depends on ([Ss]tep [0-9]+)\.\s*)+)", dependencies) is not None for dependencies in self.annotator['Dependencies'].values]
                
                self.annotator["Pass"] &= ['\n' not in rationale and re.match(r"^([^\[\.]+(\[[0-9]+\])*\.)+$", rationale) is not None and re.search(r"\[[0-9]+\]", rationale) is not None for rationale in self.annotator['Rationale'].values]
                
                if self.annot_dump:
                    self.annotator.to_csv(self.annot_dump+"_%d-%d"%(i//self.annot_step*self.annot_step, i), index=False)
                
                self.annotator = self.annotator[self.annotator["Pass"]==True]
                self.annotator.drop("Pass", axis=1, inplace=True)
                
                if len(self.annotator.index) > 0:
                    if self.annot_balance:
                        self.annotator = transform_balancedly(self.annotator)
                    
                    _annotator = pd.concat([_annotator, self.annotator], ignore_index=True)
                    if self.annot_dump:
                        _annotator.to_csv(self.annot_dump, index=False)
                    self.annotator = pd.DataFrame(columns=annot_columns)
                    
                    plan_demos = df_to_dsp_augmented(_annotator, segment="plan")
                    rewrite_demos = df_to_dsp_augmented(_annotator, segment="rewrite")
                    rationale_demos = df_to_dsp_augmented(_annotator, segment="rationale")
                    
                    if len(plan_demos) >= self.MIN_CAND_PLAN_DEMOS and len(rewrite_demos) >= self.MIN_CAND_REWRITE_DEMOS and len(rationale_demos) >= self.MIN_CAND_RATIONALE_DEMOS:
                        annot_examples = i + 1
                        break
                    
                    self.demos = {}
                    if len(plan_demos) > 0:
                        self.demos["plan"] = dsp.sample(plan_demos, k=self.MAX_PRI_PLAN_DEMOS) #plan_demos[:self.MAX_PRI_PLAN_DEMOS]
                    if len(rewrite_demos) > 0:
                        self.demos["rewrite"] = dsp.sample(rewrite_demos, k=self.MAX_PRI_REWRITE_DEMOS) #rewrite_demos[:self.MAX_PRI_REWRITE_DEMOS]
                    if len(rationale_demos) > 0:
                        self.demos["rationale"] = sample_balancedly(rationale_demos, k=self.MAX_PRI_RATIONALE_DEMOS) #rationale_demos[:self.MAX_PRI_RATIONALE_DEMOS]
                    
        print("="*35 + " ANNOTATION: EXAMPLES USED " + "="*35)
        if annot_examples:
            print(annot_examples)
        else:
            print(len(test))
            
        self.annotator = None
                
        self.demos = {}
        if self.demo_flags is not None and "plan" in self.demo_flags:
            #self.demos["plan"] = df_to_dsp_augmented(_annotator, segment="plan")[:MAX_PLAN_DEMOS]
            if "sample" in self.demo_sel_func.__name__:
                self.demos["plan"] = self.demo_sel_func(df_to_dsp_augmented(_annotator, segment="plan"), k = self.MAX_PRI_PLAN_DEMOS)
            elif "knn" in self.demo_sel_func.__name__:
                self.demos["plan"] = partial(self.demo_sel_func(df_to_dsp_augmented(_annotator, segment="plan")), k = self.MAX_PRI_PLAN_DEMOS)
            else:
                raise NotImplementedError()
        if self.demo_flags is not None and "rewrite" in self.demo_flags:
            #self.demos["rewrite"] = df_to_dsp_augmented(_annotator, segment="rewrite")[:MAX_REWRITE_DEMOS]
            if "sample" in self.demo_sel_func.__name__:
                self.demos["rewrite"] = self.demo_sel_func(df_to_dsp_augmented(_annotator, segment="rewrite"), k = self.MAX_PRI_REWRITE_DEMOS)
            elif "knn" in self.demo_sel_func.__name__:
                self.demos["rewrite"] = partial(self.demo_sel_func(df_to_dsp_augmented(_annotator, segment="rewrite")), k = self.MAX_PRI_REWRITE_DEMOS)
            else:
                raise NotImplementedError()
        if self.demo_flags is not None and "rationale" in self.demo_flags:
            #self.demos["rationale"] = df_to_dsp_augmented(_annotator, segment="rationale")[:MAX_RATIONALE_DEMOS]
            if "sample" in self.demo_sel_func.__name__:
                self.demos["rationale"] = self.demo_sel_func(df_to_dsp_augmented(_annotator, segment="rationale"), k = self.MAX_PRI_RATIONALE_DEMOS)
            elif "knn" in self.demo_sel_func.__name__:
                self.demos["rationale"] = partial(self.demo_sel_func(df_to_dsp_augmented(_annotator, segment="rationale")), k = self.MAX_PRI_RATIONALE_DEMOS)
            else:
                raise NotImplementedError()
        if self.annot_dump:
            _annotator.to_csv(self.annot_dump, index=False)
        
        dsp.settings.configure(electoral_college=default_electoral_college)
        self.depth = default_depth
        self.B = default_B
        
        if self.annot_log:
            sys.stdout = default_stdout
            log_file.close()
    '''
    def annotate(self, train, save_path):
        self.annotator = pd.DataFrame(columns=['Question', 'Plan Context', 'Plan', 'Dependencies', 'Rewrite Context', 'Rewrite', 'Rationale Context', 'Rationale', 'Answer'])
        train, test = train[:len(train)//2], train[len(train)//2:]
        
        plan_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[:2]
        rewrite_demos = df_to_dsp_augmented(pd.read_csv("data/seed_augmented.csv",keep_default_na=False))[4:5]
    
        train += (plan_demos + rewrite_demos)
    
        self.PLAN_DEMOS = len(plan_demos)
        self.REWRITE_DEMOS = len(rewrite_demos)
        
        test = dsp.sample(test, k=10)
        for example in test:
            self.annotator.loc[len(self.annotator)] = [example.question,None, None, None, None, None, None, None, None]
            self.__call__(train, example.question)
        self.annotator.to_csv(save_path, index=False)
        self.annotator = None
    '''