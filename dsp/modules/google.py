import functools
from typing import Optional, Union, Any
import requests

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.utils import dotdict

import urllib.request, json 
from serpapi import GoogleSearch
import sys


class Google:
    """Wrapper for the ColBERTv2 Retrieval."""

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


@CacheMemory.cache
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
        psg["prob"] = 0.5
        topk.append(psg)
        
    if 'organic_results' in res.keys():
        for organic_result in res['organic_results']:
            if 'snippet' in organic_result.keys() and 'title' in organic_result.keys() :
                psg = {}
                psg["long_text"] = organic_result['title'] + " | " + organic_result['snippet'] 
                psg["prob"] = 0.5
                topk.append(psg)

    return topk[:k]


@functools.lru_cache(maxsize=None)
@NotebookCacheMemory.cache
def google_request_v2_wrapped(*args, **kwargs):
    return google_request_v2(*args, **kwargs)


google_request = google_request_v2_wrapped


