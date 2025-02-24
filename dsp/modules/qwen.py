'''
Created on Feb 20, 2025

@author: Yihao Fang
'''
import functools
import json
from typing import Any, Literal, Optional, cast

import backoff

from dsp.modules.cache_utils import CacheMemory, cache_turn_on
from dsp.modules.lm import LM

import sys

from ollama import chat
from ollama import ChatResponse

def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )

class Qwen(LM):
    """Wrapper around Qwen. 

    Args:
        model (str, optional): Defaults to "qwen2.5:72b".
        api_provider (Literal["openai", "azure"], optional): The API provider to use. Defaults to "ollama".
        model_type (Literal["chat", "reasoner"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "chat".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        model: str = "qwen2.5:72b",
        api_provider: Literal["ollama", "huggingface"] = "ollama",
        model_type: Literal["chat", "reasoner"] = "chat",
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "qwen"
        self.model_type = model_type

        self.kwargs = {
            **self.kwargs,
            **kwargs,
            "api_provider":api_provider
        }
        
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        kwargs = {
            "stringify_request": json.dumps(kwargs)
        }
        response = cached_qwen_request(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        (),
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retreival of Qwen completions whilst handling rate limiting and caching."""
        
        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        if self.model_type == "chat":
            return choice["message"]["content"]
        else:
            raise NotImplementedError()

    def __call__(
        self,
        prompt: str,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from Qwen.

        Args:
            prompt (str): prompt to send to Qwen
            
        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        response = self.request(prompt, **kwargs)
        choices = response["choices"]

        completions = [self._get_choice_text(c) for c in choices]

        return completions


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@CacheMemory.cache
def _cached_qwen_request(**kwargs):
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    response = {}
    choices = []
    for _ in range(kwargs["n"]):
        if kwargs["api_provider"] == "ollama":
            choice: ChatResponse = chat(model=kwargs["model"], messages=kwargs["messages"])
        else:
            raise NotImplementedError()
        choices.append(choice)
    response["choices"] = choices
    return response


def _cached_qwen_request_logged(**kwargs):
    print("calling qwen ...", file=sys.stderr)
    print("~"*35 + " PROMPT " + "~"*35)
    for message in json.loads(kwargs["stringify_request"])["messages"]:
        print("."*35 + message['role'] + "."*35)
        print(message['content'])
    response = _cached_qwen_request(**kwargs)
    print("-"*35 + " RESPONSE " + "-"*35)
    for i, choice in enumerate(response["choices"]):
        print("-"*35 + (" CHOICE %d "%i) + "-"*35)
        print("."*35 + choice['message']['role'] + "."*35)
        print(choice['message']['content'])
    return response

cached_qwen_request = _cached_qwen_request_logged