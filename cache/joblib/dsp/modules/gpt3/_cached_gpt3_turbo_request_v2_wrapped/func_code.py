# first line: 210
@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def _cached_gpt3_turbo_request_v2_wrapped(**kwargs) -> OpenAIObject:
    response = _cached_gpt3_turbo_request_v2(**kwargs)
    return response
