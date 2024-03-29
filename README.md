### Code for "Few-shot Dialogue Strategy Learning for Motivational Interviewing via Inductive Reasoning"

- File diretory structure:
  - train_agent.py
    - for learning dialogue strategies
  - inference.py
    - for evaluating dialogue strategies
  - tom_detector.py
    - util functions for indexing experiences (strategies) with user mental state, that inferred user mental state
  - utils
    - building blocks, should be readable
  - AnnoMI
    - The dataset (also, the code we used to split the data)

- Getting Started
  - Fill out the assets.py in appendix, it needs
    - your OPENAI api key
    - path to a huggingface-compatible classifier (like BERT, see below)


- Obtaining the dialogue act classifier:
  - Use [Huggingface Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) to finetune [mental-bert](https://huggingface.co/mental/mental-bert-base-uncased) on [dataset by Welivita et. al](https://aclanthology.org/2022.coling-1.293/). We report hyper-parameters in the appendix (and feel free to use your favorite fine-tuning codebase). 
  - Unless otherwise noted in our paper, all parameters are default parameters.
