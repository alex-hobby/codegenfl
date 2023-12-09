# CodeGenFL
Fine-tuning Code Generation models on separate clients holding different datasets, then applying federated learning techniques to learn global fine-tuned model

## Code Organization

`experiments/`

This folder contains all of the Google Colab notebooks within which we conducted our model experiments.

1. `CodeParrot Experiments.ipynb`: experiments on CodeParrot models w/ IBM Project CodeNet and StarCoder datasets
2. `CodeGen Experiments.ipynb`: experiments on CodeGen model w/ IBM Project CodeNet dataset
3. `MDPP Eval Dataset.ipynb`: experiments on DeciCoder model w/ MBPP dataset
4. `Data and Visualizations [for Exp2].ipynb`: creating visualizations for usage in report (loss functions, etc)
5. `Aggregate and Save.ipynb`: for testing with aggregating the model
6. `IBM Data Preview.ipynb`: for investigation of IBM dataset
7. `model_sizes.ipynb`: for computing the size of models before/after quantization, and trainable params before/after lora

`helpers/`

This folder contains the helper functions used across our experiments. 

We standardized commonly used code (training loop, filtered generations, etc)

Please reference docstrings of functions themselves for more details.

1. `train.py`: main training loop which trains and saves models to local directories.
2. `train_alt.py`: alternative version of training loop
3. `fl_impl.py`: our implementation of FedAvg on provided clients
4. `evaluation.py`: contains helper functions for generations, humaneval, rouge, etc.


## References

### Models:
- CodeParrot-Small: https://huggingface.co/codeparrot/codeparrot-small
- CodeParrot: https://huggingface.co/codeparrot/codeparrot
- CodeGen: https://huggingface.co/Salesforce/codegen-2B-mono
- DeciCoder: https://huggingface.co/Deci/DeciCoder-1b

### Datasets:
- StarCoder: https://huggingface.co/bigcode/starcoder 
- IBM Project CodeNet: https://developer.ibm.com/exchanges/data/all/project-codenet/
- HumanEval: https://github.com/openai/human-eval
- MBPP: https://huggingface.co/datasets/mbpp
