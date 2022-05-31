# Medical Reasoning using GPT-3

## Setup

<details>
<summary>Installation</summary>

1. Install poetry

```shell
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

2. Install dependencies

```shell
poetry install
poetry run pre-commit install
```

</details>

## Running experiments

```shell
poetry run experiment <args>
# Example
poetry run experiment engine=ada dataset.name=medqa_us dataset.subset=10
```

```shell
poetry run poe debug_exps
```
