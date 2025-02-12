# HGOT: Hierarchical Graph of Thoughts for Retrieval-Augmented In-Context Learning in Factuality Evaluation
With the widespread adoption of large language models (LLMs) in numerous applications, the challenge of factuality and the propensity for hallucinations has emerged as a significant concern. To address this issue, particularly in retrieval-augmented in-context learning, we introduce the hierarchical graph of thoughts (HGOT), a structured, multi-layered graph approach designed to enhance the retrieval of pertinent passages during in-context learning. The framework utilizes the emergent planning capabilities of LLMs, employing the divide-and-conquer strategy to break down complex queries into manageable sub-queries. It refines self-consistency majority voting for answer selection, which incorporates the recently proposed citation recall and precision metrics to assess the quality of thoughts, linking an answer's credibility intrinsically to the thought's quality. This methodology introduces a weighted system in majority voting, prioritizing answers based on the citation quality of their thoughts. Additionally, we propose a scoring mechanism for evaluating retrieved passages, considering factors such as citation frequency and quality, self-consistency confidence, and the retrieval module's ranking. Experiments indicate that HGOT excels as a versatile approach, outperforming competing models in FEVER by up to 7% and matching leading models such as Retrieve-then-Read in Open-SQuAD, and DSP in HotPotQA, demonstrating its efficacy in enhancing LLMs' factuality.

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
## Citation
If you use our source code in a research paper, please cite our work as follows:

```
@inproceedings{fang-etal-2024-hgot,
    title = "{HGOT}: Hierarchical Graph of Thoughts for Retrieval-Augmented In-Context Learning in Factuality Evaluation",
    author = "Fang, Yihao  and
      Thomas, Stephen  and
      Zhu, Xiaodan",
    editor = "Ovalle, Anaelia  and
      Chang, Kai-Wei  and
      Cao, Yang Trista  and
      Mehrabi, Ninareh  and
      Zhao, Jieyu  and
      Galstyan, Aram  and
      Dhamala, Jwala  and
      Kumar, Anoop  and
      Gupta, Rahul",
    booktitle = "Proceedings of the 4th Workshop on Trustworthy Natural Language Processing (TrustNLP 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.trustnlp-1.12/",
    doi = "10.18653/v1/2024.trustnlp-1.12",
    pages = "118--144"
}
```
