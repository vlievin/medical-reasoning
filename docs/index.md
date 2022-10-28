# Can large language models reason about medical questions?

Generated chain-of-thought samples from the paper [Can large language models reason about medical questions?](https://arxiv.org/abs/2207.08143) The github repository can be found [here](https://github.com/vlievin/medical-reasoning). Abstract:

> Although large language models (LLMs) often produce impressive outputs, they also fail to reason and be factual. We set out to investigate how these limitations affect the LLM's ability to answer and reason about difficult real-world based questions. We applied the human-aligned GPT-3 (InstructGPT) to answer multiple-choice medical exam questions (USMLE and MedMCQA) and medical research questions (PubMedQA). We investigated Chain-of-thought (think step by step) prompts, grounding (augmenting the prompt with search results) and few-shot (prepending the question with question-answer exemplars). For a subset of the USMLE questions, a medical domain expert reviewed and annotated the model's reasoning. Overall, GPT-3 achieved a substantial improvement in state-of-the-art machine learning performance. We observed that GPT-3 is often knowledgeable and can reason about medical questions. GPT-3, when confronted with a question it cannot answer, will still attempt to answer, often resulting in a biased predictive distribution. LLMs are not on par with human performance but our results suggest the emergence of reasoning patterns that are compatible with medical problem-solving. We speculate that scaling model and data, enhancing prompt alignment and allowing for better contextualization of the completions will be sufficient for LLMs to reach human-level performance on this type of task.


## Samples of Chain-of-Thoughts

### Main samples

6 prompts, full test set (USMLE, PubMedQA), 1k validation samples (MedMCQA)

- [USMLE](samples/usmle.html)
- [MedMCQA without context](samples/medmcqa-wo-context.html)
- [MedMCQA with context](samples/medmcqa-w-context.html)
- [PubMedQA](samples/pubmedqa.html)

### Validation samples

30 prompts, 100 USMLE questions

- [USMLE](samples/usmle-validation.html)
