import numpy as np
import dsp


def retrieve(query: str, k: int, return_dict: bool = False) -> list[str]:
    """Retrieves passages from the RM for the query and returns the top k passages."""
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = dsp.settings.rm(query, k=k)

    if dsp.settings.reranker:
        passages_cs_scores = dsp.settings.reranker(query, passages)
        passages_cs_scores_sorted = np.argsort(passages_cs_scores)[::-1]
        #passages = [passages[idx] for idx in passages_cs_scores_sorted]
        if not return_dict:
            results = [passages[idx].long_text for idx in passages_cs_scores_sorted]
        else:
            results = {"text":[passages[idx].long_text for idx in passages_cs_scores_sorted], "score":[passages[idx].score for idx in passages_cs_scores_sorted]}
    else:
        if not return_dict:
            results = [psg.long_text for psg in passages]
        else:
            results = {"text":[psg.long_text for psg in passages], "score":[psg.score for psg in passages]}
    return results


def retrieveRerankEnsemble(queries: list[str], k: int, return_dict: bool = False) -> list[str]:
    if not (dsp.settings.rm and dsp.settings.reranker):
        raise AssertionError("Both RM and Reranker are needed to retrieve & re-rank.")
    queries = [q for q in queries if q]
    passages = {}
    for query in queries:
        retrieved_passages = dsp.settings.rm(query, k=k*3)
        #passages_cs_scores = dsp.settings.reranker(query, [psg.long_text for psg in retrieved_passages])
        passages_cs_scores = dsp.settings.reranker(query, retrieved_passages)
        for idx in np.argsort(passages_cs_scores)[::-1]:
            psg = retrieved_passages[idx]
            passages[psg.long_text] = passages.get(psg.long_text, []) + [
                passages_cs_scores[idx]
            ]

    passages = [(np.average(score), text) for text, score in passages.items()]
    if not return_dict:
        return [text for _, text in sorted(passages, reverse=True)[:k]]
    else:
        results = sorted(passages, reverse=True)[:k]
        return {"text":[text for _, text in results], "score":[score for score, _ in results]}


def retrieveEnsemble(queries: list[str], k: int, by_prob: bool = False, return_dict: bool = False) -> list[str]:
    """Retrieves passages from the RM for each query in queries and returns the top k passages
    based on the probability or score.
    """
    if not dsp.settings.rm:
        raise AssertionError("No RM is loaded.")
    if dsp.settings.reranker:
        return retrieveRerankEnsemble(queries, k)
    queries = [q for q in queries if q]

    passages = {}
    for q in queries:
        for psg in dsp.settings.rm(q, k=k * 3):
            if by_prob:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.prob
            else:
                passages[psg.long_text] = passages.get(psg.long_text, 0.0) + psg.score

    passages = [(score, text) for text, score in passages.items()]
    results = sorted(passages, reverse=True)[:k]
    if not return_dict:
        return [text for _, text in results]
    else:
        return {"text":[text for _, text in results], "score":[score for score, _ in results]}
