import functools
from typing import Optional, Union, Any
import requests

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.utils import dotdict

import urllib.request, json 
from serpapi import GoogleSearch
import sys
import backoff

def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )

class Google:
    """Wrapper for the Google Retrieval."""

    def __init__(
        self,
        serpapi_key: str
    ):
        self.serpapi_key = serpapi_key

    def __call__(
        self, query: str, k: int = 10, simplify: bool = False
    ) -> Union[list[str], list[dotdict]]:
        
        topk: list[dict[str, Any]] = google_request(self.serpapi_key, query, k)
        
        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@CacheMemory.cache
@backoff.on_exception(
    backoff.expo,
    (json.decoder.JSONDecodeError),
    max_time=1000,
    on_backoff=backoff_hdlr,
)
def google_request_v2(serpapi_key, query, k):
    assert (
        k <= 100
    )

    params = {
        "api_key": serpapi_key,
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }
    
    print("calling google ...", file=sys.stderr)
  
    search = GoogleSearch(params)
    res = search.get_dict()

    topk = []

    if 'answer_box' in res.keys() and 'title' in res['answer_box'].keys() and \
        ('answer' in res['answer_box'].keys() or 'snippet' in res['answer_box'].keys() or 'snippet_highlighted_words' in res['answer_box'].keys()):
        psg = {}
        psg["long_text"] = res['answer_box']['title'] + " | "
        phrases = []
        if 'answer' in res['answer_box'].keys():
            phrases.append(res['answer_box']['answer'])
        if 'snippet' in res['answer_box'].keys():
            phrases.append(res['answer_box']['snippet'])
        if 'snippet_highlighted_words' in res['answer_box'].keys():
            phrases.append(", ".join(res['answer_box']["snippet_highlighted_words"]))
        psg["long_text"] += ", ".join(phrases)
        
        ### Major updates - August 30, 2023
        #psg["prob"] = 0.5
        psg["score"] = 1
        ###
        psg["link"] = res['answer_box']['link']
        topk.append(psg)
        
    if 'organic_results' in res.keys():
        for organic_result in res['organic_results']:
            if 'snippet' in organic_result.keys() and 'title' in organic_result.keys() :
                psg = {}
                psg["long_text"] = organic_result['title'] + " | " + organic_result['snippet'] 
                
                ### Major updates - August 30, 2023
                if 'rich_snippet' in organic_result.keys() and 'top' in organic_result['rich_snippet'].keys() and 'extensions' in organic_result['rich_snippet']['top'].keys():
                    psg["long_text"] += (" " + (", ".join(organic_result['rich_snippet']['top']['extensions'])))
                #psg["prob"] = 0.5
                psg["score"] = 1/float(organic_result['position'])
                ###
                psg["link"] = organic_result['link']
                topk.append(psg)
    #print(topk, file=sys.stderr)
    return topk[:k]

'''
@functools.lru_cache(maxsize=None)
@NotebookCacheMemory.cache
def google_request_v2_wrapped(*args, **kwargs):
    return google_request_v2(*args, **kwargs)
'''

google_request = google_request_v2



