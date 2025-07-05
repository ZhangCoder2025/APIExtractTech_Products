# User Guide
- This repository stores the data and code used in the paper titled **Improving API Knowledge Comprehensibility: A Context-Dependent Entity Detection and Context Completion Approach using LLM**.
- The `API_sentences_with_LVCDE_and_RCDE.jsonl` is the dataset constructed by us, containing 1023 API sentences and 567 context-dependent entities.
- `CEDCC_pipline.py` contains the code of full process code for detecting context-dependent entities in sentences and completing contexts, but you need to provide the LLM API keys.
- `LLM_prompt.py` contains the prompt for the LLM used in **CEDCC**.
- `utils.py` provides some functions used in the process.
- `envs.txt` is the environment configuration required to reproduce the package.
- `Questionaire_of_50_cases_with_3_approaches.jsonl` is the questionnaire used for human evaluation in the paper, where sentence 1 is the original API sentence, sentence 2 is the API sentence processed by **AllenNLP-baseline**, sentence 3 is the sentence processed by **CEDCC**, and sentence 4 is the sentence processed by **LLM-baseline**. In addition, each case also provides the ID number of the SO post to which the sentence belongs, so that the respondents can determine the correctness of the completed content.

