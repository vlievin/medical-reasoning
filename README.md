> **Warning** Work in progress

# Medical Reasoning using GPT-3

Official repository for the paper [Can large language models reason about medical questions?](https://arxiv.org/abs/2207.08143)

> Although large language models (LLMs) often produce impressive outputs, they also fail to reason and be factual. We set out to investigate how these limitations affect the LLM's ability to answer and reason about difficult real-world based questions. We applied the human-aligned GPT-3 (InstructGPT) to answer multiple-choice medical exam questions (USMLE and MedMCQA) and medical research questions (PubMedQA). We investigated Chain-of-thought (think step by step) prompts, grounding (augmenting the prompt with search results) and few-shot (prepending the question with question-answer exemplars). For a subset of the USMLE questions, a medical domain expert reviewed and annotated the model's reasoning. Overall, GPT-3 achieved a substantial improvement in state-of-the-art machine learning performance. We observed that GPT-3 is often knowledgeable and can reason about medical questions. GPT-3, when confronted with a question it cannot answer, will still attempt to answer, often resulting in a biased predictive distribution. LLMs are not on par with human performance but our results suggest the emergence of reasoning patterns that are compatible with medical problem-solving. We speculate that scaling model and data, enhancing prompt alignment and allowing for better contextualization of the completions will be sufficient for LLMs to reach human-level performance on this type of task.


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

## Samples

Samples of generated CoTs for the USMLE, MedMCQA and PubMedQA datasets can be accessed [here](https://vlievin.github.io/medical-reasoning).

## Cached GPT-3 predictions

All GPT-3 completions are automatically cached (see `medical_reasoning/models/cache.py`) and re-used whenever the API is called using the same parameters. You can find all the cached completions (all experiments, all the chain-of-thoughts) in the following zip files;

```shell
https://f001.backblazeb2.com/file/FindZebraData/medical-reasoning/cached_funcs.zip
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
