# first line: 34
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
        
        ### Major updates - August 30, 2023
        #psg["prob"] = 0.5
        psg["score"] = 1
        ###
        
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
                
                topk.append(psg)

    return topk[:k]