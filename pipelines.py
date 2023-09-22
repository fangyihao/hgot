'''
Created on Sep. 7, 2023

@author: Yihao Fang
'''
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
from utils import df_to_dsp_augmented
import pandas as pd
seed = 42
np.random.seed(seed)
random.seed(seed)
language_model='gpt-3.5-turbo'
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
def paraphrase(passage, n, temperature=0.9):
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
            prefix="Rationale: Let's think step by step." if not dsp.settings.electoral_college else """Rationale: Let's think step by step. Every statement in the "Rationale" section should be attributable to the citations provided in the "Context" section.""",
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
            if dsp.settings.electoral_college:
                completions, citation_frequency = dsp.settings.electoral_college(example, completions)
            else:
                completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(self.qa_template_with_CoT)(example, stage='qa')
        
        return example.copy(answer=completions.answer) if not dsp.settings.electoral_college else example.copy(answer=completions.answer, citation_frequency=citation_frequency)
    
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
            prefix="Rationale: Let's think step by step." if not dsp.settings.electoral_college else """Rationale: Let's think step by step. Every statement in the "Rationale" section should be attributable to the citations provided in the "Context" section.""",
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
            if dsp.settings.electoral_college:
                completions, citation_frequency = dsp.settings.electoral_college(example, completions)
            else:
                completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(self.qa_template_with_CoT)(example, stage='qa')
        
        return example.copy(answer=completions.answer) if not dsp.settings.electoral_college else example.copy(answer=completions.answer, citation_frequency=citation_frequency)
    
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

    def __init__(self, demos = ['plan', 'rewrite'], p_context: bool = True, retrieve_ensemble_n = None):
        self.EDGE_PATTERN = r'\s*([Ss]tep [0-9]+)\s*->\s*([Ss]tep [0-9]+)\s*'
        self.NODE_PATTERN = r'\s*([Ss]tep [0-9]+):\s*(.*)'
        
        self.demos = demos
        self.p_context = p_context
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
            prefix="Rationale: Let's think step by step." if not dsp.settings.electoral_college else """Rationale: Let's think step by step. Every statement in the "Rationale" section should be attributable to the citations provided in the "Context" section.""",
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
            if dsp.settings.electoral_college:
                completions, citation_frequency = dsp.settings.electoral_college(example, completions)
            else:
                completions = dsp.majority(completions)
        else:
            example, completions = dsp.generate(self.qa_template_with_CoT)(example, stage='qa')

        return example.copy(answer=completions.answer, rationale = completions.rationale) if not dsp.settings.electoral_college else example.copy(answer=completions.answer, rationale = completions.rationale, citation_frequency = citation_frequency)
    
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
            
            def retrieve_ensemble(query: str, k: int, return_dict:bool = False) -> list[str]:
                #psgs = dsp.retrieve(query, k=k*3)
                #return psgs[:k]
                assert self.retrieve_ensemble_n >= 1
                
                if self.retrieve_ensemble_n > 1:
                    paraphrases = paraphrase(query, self.retrieve_ensemble_n-1)
                    return dsp.retrieveEnsemble([query]+paraphrases, k=k, return_dict=return_dict)
                elif self.retrieve_ensemble_n == 1:
                    return dsp.retrieveEnsemble([query], k=k, return_dict=return_dict)

        if dsp.settings.electoral_college:
            
            def rerank(passages, scores, citation_frequency, citation_weight = 0.7):
                assert (len(passages) == len(scores) and len(passages) == len(citation_frequency))
                for i, passage in enumerate(passages): 
                    scores[i] = (1-citation_weight)*scores[i] + citation_weight*citation_frequency[i]
                return [passage for passage, _ in sorted(zip(passages, scores), key=lambda item: item[1], reverse=True)]

        if self.p_context == False:
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
                    passages_w_scores = retrieve_ensemble(self.find_step_question(questions[u]), k=k, return_dict = True)
                else:
                    passages_w_scores = dsp.retrieve(self.find_step_question(questions[u]), k=k, return_dict = True)
                passages[u], scores = passages_w_scores["text"], passages_w_scores["score"]
                    
                completions = self.QA_predict(dsp.Example(question=self.find_step_question(questions[u]), demos=example.demos, context=passages[u]))
                answers[u] = completions.answer
                rationale = completions.rationale
                if dsp.settings.electoral_college:
                    passages[u] = rerank(passages[u], scores, completions.citation_frequency)
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
                    
                    if self.demos:
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
                        passages_w_scores = retrieve_ensemble(self.find_step_question(rewrite), k=k, return_dict = True)
                    else:
                        passages_w_scores = dsp.retrieve(self.find_step_question(rewrite), k=k, return_dict = True)
                    passages[u], scores = passages_w_scores["text"], passages_w_scores["score"]
                        
                    completions = self.QA_predict(dsp.Example(question=self.find_step_question(rewrite), demos=example.demos, context=passages[u]))
                    answers[u] = completions.answer
                    rationale = completions.rationale
                    if dsp.settings.electoral_college:
                        passages[u] = rerank(passages[u], scores, completions.citation_frequency)
                    #example.context.extend(passages[u])
                    #example.context.extend([questions[u] + " | " + answers[u]])
        assert len(answers) == len(G.nodes)     
        
        # verbose
        retrieval_history=dict([(self.find_step_question(questions[u]), passages[u]) for u in questions])
        if self.p_context:
            retrieval_history[example.question] = copy.deepcopy(example.context)
        
        for u in questions:
            #example.context.extend([questions[u] + " | " + answers[u]])
            example.context.extend(passages[u][:1])
            
        print("-"*35 + " STEPS WITH ANSWERS " + "-"*35)
        for u in questions:
            print(questions[u] + " ANSWER: " + answers[u])

        #return example
        return example.copy(retrieval_history=retrieval_history)
    
    def extract_plan(self, plan):
        return re.sub(r"(.+)Context:.+", r"\1", plan, flags=re.DOTALL)
    
    @dsp.transformation
    def plan(self, example: dsp.Example, self_reflect=True) -> dsp.Example:
        if self.p_context == True:
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
        if not self.demos and self.p_context == False:
            x = dsp.Example(question=question, demos=dsp.sample(train, k=7))
        elif self.demos:
            demos = train[-(self.PLAN_DEMOS+self.REWRITE_DEMOS):-self.REWRITE_DEMOS]
            if self.p_context == False:
                x = dsp.Example(question=question, demos=demos)
            else:
                context = dsp.retrieve(question, k=3)
                
                #print("context:",context)
                
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
            
        if self.demos:
            demos = dsp.sample(train[:-(self.PLAN_DEMOS+self.REWRITE_DEMOS)], k=7)
            x.demos=demos
        print("="*35 + " SEARCH " + "="*35)
        x = self.multistep_search(train, x)
        print("="*35 + " PREDICT " + "="*35)
        x = self.QA_predict(x)
        return {"answer":x.answer}
    
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
