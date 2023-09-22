# first line: 95
@functools.lru_cache(maxsize=None)
@NotebookCacheMemory.cache
def google_request_v2_wrapped(*args, **kwargs):
    return google_request_v2(*args, **kwargs)
