# P-GoT
We explore the underexamined planning as an emergent reasoning capability of large language models. This research specifically addresses the hitherto uninvestigated role of planning in retrieval-augmented in-context learning. Furthermore, we carefully evaluate the planning capabilities of ChatGPT, in the context of question-answering tasks. The results of our experimental evaluation reveal that, when equipped with planning capabilities, the pipeline of retrieval-augmented in-context learning can match or even surpass the performance of the state-of-the-art approaches such as Demonstrate-Search-Predict, Self-ask, and Retrieve-then-Read. This provides insightful evidence on the potential and effectiveness of ChatGPT's planning abilities in enhancing question-answering performance.

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
