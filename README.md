> **Note** This repo is a snapshot of the code used to produce our results up to December 2022 (v2, including Codex 5-shot CoT results). The code won't be updated further to ensure maximum reproducibility

# Medical Reasoning using GPT-3.5

Official repository for the paper [Can large language models reason about medical questions?](https://arxiv.org/abs/2207.08143)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/can-large-language-models-reason-about/question-answering-on-medqa-usmle)](https://paperswithcode.com/sota/question-answering-on-medqa-usmle?p=can-large-language-models-reason-about)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/can-large-language-models-reason-about/multiple-choice-question-answering-mcqa-on-21)](https://paperswithcode.com/sota/multiple-choice-question-answering-mcqa-on-21?p=can-large-language-models-reason-about)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/can-large-language-models-reason-about/question-answering-on-pubmedqa)](https://paperswithcode.com/sota/question-answering-on-pubmedqa?p=can-large-language-models-reason-about)

## Abstract

> Although large language models (LLMs) often produce impressive outputs, it remains unclear how they perform in real-world scenarios requiring strong reasoning skills and expert domain knowledge. We set out to investigate whether GPT-3.5 (Codex and InstructGPT) can be applied to answer and reason about difficult real-world-based questions. We utilize two multiple-choice medical exam questions (USMLE and MedMCQA) and a medical reading comprehension dataset (PubMedQA). We investigate multiple prompting scenarios: Chain-of-Thought (CoT, think step-by-step), zero- and few-shot (prepending the question with question-answer exemplars) and retrieval augmentation (injecting Wikipedia passages into the prompt). For a subset of the USMLE questions, a medical expert reviewed and annotated the model's CoT. We found that InstructGPT can often read, reason and recall expert knowledge. Failure are primarily due to lack of knowledge and reasoning errors and trivial guessing heuristics are observed, e.g.\ too often predicting labels A and D on USMLE. Sampling and combining many completions overcome some of these limitations. Using 100 samples, Codex 5-shot CoT not only gives close to well-calibrated predictive probability but also achieves human-level performances on the three datasets. USMLE: 60.2%, MedMCQA: 57.5% and PubMedQA: 78.2%.

## CoT Samples

Samples of generated CoTs for the USMLE, MedMCQA and PubMedQA datasets can be accessed [here](https://vlievin.github.io/medical-reasoning).

More samples will be made available through [ThoughtSource ⚡](https://github.com/OpenBioLink/ThoughtSource).

## Setup

<details>
<summary>Install poetry</summary>


```shell
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

</details>
<details>
<summary>Install dependencies</summary>

```shell
poetry install
```

</details>
<details>
<summary>Setup Elasticsearch</summary>

```shell
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.14.1-linux-x86_64.tar.gz
tar -xzf elasticsearch-7.14.1-linux-x86_64.tar.gz
```
To run ElasticSearch navigate to the `elasticsearch-7.14.1` folder in the terminal and run `./bin/elasticsearch`.

</details>


## Running one experiment

Use `poetry run` to load and run using the `poetry` environment.

```shell
poetry run experiment <args>
# Example
poetry run experiment engine=code dataset.name=medqa_us dataset.subset=10
```

## Running a group of experiments

Groups of experiments are defined in `pyproject.toml`

```shell
poetry run poe medqa_test
poetry run poe medmcqa_valid
poetry run poe pubmedqa_test
poetry run poe mmlu_test_code
```

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2207.08143,
  doi = {10.48550/ARXIV.2207.08143},
  url = {https://arxiv.org/abs/2207.08143},
  author = {Liévin, Valentin and Hother, Christoffer Egeberg and Winther, Ole},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.1; I.2.7},
  title = {Can large language models reason about medical questions?},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
