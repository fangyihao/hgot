# HGOT
In the realm of large language models (LLMs), the challenge of factuality and the propensity for hallucinations raises significant concerns. To address this issue, particularly in retrieval-augmented in-context learning, we introduce the hierarchical graph of thoughts (HGOT), a structured, multi-layered graph approach designed to enhance the retrieval of pertinent passages during in-context learning. The framework utilizes the emergent planning capabilities of LLMs, employing the divide-and-conquer strategy to break down complex queries into manageable sub-queries. It refines self-consistency majority voting for answer selection, which incorporates the recently proposed citation recall and precision metrics to assess the quality of thoughts, linking an answer's credibility intrinsically to the thought's quality. This methodology introduces a weighted system in majority voting, prioritizing answers based on the citation quality of their thoughts. Additionally, we propose a scoring mechanism for evaluating retrieved passages, considering factors such as citation frequency, citation quality, and the retrieval module's ranking. Our experiments and analyses reveal that HGOT markedly surpasses other retrieval-augmented in-context learning methods, including Demonstrate-Search-Predict (DSP), ReAct, Self-Ask, and Retrieve-then-Read on various datasets, demonstrating its efficacy in enhancing the factuality of LLMs.

## Download Datasets

Open-SQuAD: 
```
curl -o data/Open-SQuAD/biencoder-squad1-train.json.gz https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz
curl -o data/Open-SQuAD/biencoder-squad1-dev.json.gz https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz
curl -o data/Open-SQuAD/squad1-test.qa.csv https://dl.fbaipublicfiles.com/dpr/data/retriever/squad1-test.qa.csv
```
HotPotQA: 
```
curl -o data/HotPotQA/hotpot_train_v1.1.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
curl -o data/HotPotQA/hotpot_dev_fullwiki_v1.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
```
QReCC:
```
curl -o data/QReCC/qrecc_data.zip https://github.com/apple/ml-qrecc/blob/main/dataset/qrecc_data.zip
```
