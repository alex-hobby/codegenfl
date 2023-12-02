from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
)

from peft import prepare_model_for_kbit_training, get_peft_model

import torch
from torch.utils.data import Dataset, DataLoader

from bs4 import BeautifulSoup
import re

from rouge import Rouge

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    
def generate_batch_completions(prompts, model, tokenizer, max_new_tokens=128):
    """
    Generate a completion given a prompt using next-token prediction.

    :param List prompts: contains prompts in string list form
    :param model: model variable to use for generation
    :param tokenizer: tokenizer for turning tokens into one-hot vectors
    :param max_new_tokens: maximum tokens to generate
    :return: list of response strings to given prompts
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type: {device}")

    config = GenerationConfig(
        max_new_tokens=max_new_tokens
    )

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        pad_token_id=tokenizer.eos_token_id,
        generation_config=config
    )

    completions = [tokenizer.decode(output, clean_up_tokenization_spaces=True) for output in outputs]
    completions = clean_samples(completions)

    return completions

def get_humaneval_dataloader(problems, batch_size):
    """
    Get dataloader of HumanEval problems

    :param problems: list of HumanEval problems
    :param int batch_size: number of problems per batch in dataloader
    :return: dataloader of HumanEval problems
    """
    class CustomDataset(Dataset):
        def __init__(self, data_dict):
            self.data = list(data_dict.values())

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            # Convert data to PyTorch tensors if needed
            tensor_sample = {
                'task_id': sample['task_id'],
                'prompt': sample['prompt'],
                # Add more fields as needed
            }
            return tensor_sample

    probs_dataloader = DataLoader(CustomDataset(problems), batch_size=batch_size, shuffle=True)

    return probs_dataloader

def generate_model_on_problems(model, probs_dataloader, tokenizer):
    """
    Generate sample responses for problems dataloader

    :param model: model used for generation
    :param probs_dataloader: PyTorch dataloader containing problem strings
    :param tokenizer: tokenizer used for tokenization
    :return: list of sample dicts containing task id and completion
    """
    samples = []

    for i, batch in enumerate(probs_dataloader):
        task_id = batch['task_id']
        prompts = batch['prompt']
        for _ in range(1):
            completions = generate_batch_completions(prompts, model, tokenizer)
            print(f'Generated completions for batch {i} of {len(probs_dataloader)}')

            for i in range(len(completions)):
                samples.append(dict(task_id=task_id[i], completion=completions[i]))

            del completions
            torch.cuda.empty_cache()

    return samples


def get_prompt_from_descr(meta_path):
    """
    Fetches prompt from description file in Google Drive

    :param str meta_path: absolute path to problem description file in Drive
    :return: parsed prompt from problem description
    """
    with open(meta_path) as f:
        metadata_string = f.read()

    metadata_string = metadata_string[:metadata_string.find("<H2>")]
    soup = BeautifulSoup(metadata_string, 'html.parser')
    prompt = "\"\"\"\n" + soup.get_text() + "\n" + "\"\"\"\n"

    return prompt


def cut_off_after_function(input_code):
    """
    Helper function to remove extra generations since model does not know when to stop

    Removes additional function definitions that may be generated, we only care about the first one

    If no additional line breaks w/o indentations are detected, nothing happens

    :param str input_code: raw generated code from model
    :return: truncated code
    """
    found = list([i.end() for i in re.finditer('def.*?(\n\n)[^\s]', input_code, re.DOTALL)])

    if found:
        cut_off_code = input_code[:found[0]-1]
        return cut_off_code
    else:
        return input_code

def clean_samples(samples):
    """
    Function to clean samples by removing <|endoftext|> tokens and truncating

    :param List samples: list of raw generated code samples
    :return: cleaned samples
    """
    samples_new = []
    for val in samples:
        completion = val.replace('<|endoftext|>', '')

        completion = cut_off_after_function(completion)

        samples_new.append(completion)
    return samples_new


def compute_average_rouge_scores(references, hypotheses):
    """
    Function to return ROUGE scores for provided references and hypotheses

    :param List references: the expected responses (from benchmark dataset)
    :param List hypotheses: the generated responses (from trained model)
    :return: ROUGE scores
    """
    rouge = Rouge()
    total_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                    'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                    'rouge-l': {'f': 0, 'p': 0, 'r': 0}}

    num_prompts = len(references)

    for i in range(num_prompts):
        reference = references[i]
        hypothesis = hypotheses[i]

        scores = rouge.get_scores(hypothesis, reference)[0]

        for metric in total_scores.keys():
            for measure in ['f', 'p', 'r']:
                total_scores[metric][measure] += scores[metric][measure]

    # Compute averages
    for metric in total_scores.keys():
        for measure in ['f', 'p', 'r']:
            total_scores[metric][measure] /= num_prompts

    return total_scores


def compute_bleu_score(reference, hypothesis):
    """
    Function to return BLEUE score for provided references and hypotheses

    :param List references: the expected responses (from benchmark dataset)
    :param List hypotheses: the generated responses (from trained model)
    :return: BLEU score
    """
    # Convert the reference and hypothesis sentences into lists of tokens
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()

    # Compute BLEU score with smoothing
    smooth = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smooth)

    return bleu_score


def print_gens_on_problem_all_models(SAVE_PATH, problem_desc_path, model_info, global_agg_model, num_clients=2):
    """
    Given an IBM problem description path, generate code samples for global model, clients, and aggregated model

    Samples are printed and not returned

    :param str SAVE_PATH: Location to save model checkpoint
    :param str problem_desc_path: absolute path to problem description file in Drive
    :param model_info: object containing info about model
    :param global_agg_model: aggregated model pre-computed in notebook
    :param int num_clients: number of participating clients
    :return: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type: {device}")

    prompt = get_prompt_from_descr(problem_desc_path)

    # Original model
    original_model = AutoModelForCausalLM.from_pretrained(
        model_info['model_name'], quantization_config=model_info['quant_config'], device_map={"": 0}
    )

    print(generate_batch_completions([prompt], original_model, model_info['tokenizer'])[0])
    print('------------------------')
    del original_model
    torch.cuda.empty_cache()

    # Client models
    for client_idx in range(num_clients):
        client_model = AutoModelForCausalLM.from_pretrained(
            SAVE_PATH + f"_{client_idx}", quantization_config=model_info['quant_config'], device_map={"": 0}
        )
        print("Loaded model")
        client_model = prepare_model_for_kbit_training(client_model)
        client_model = get_peft_model(client_model, model_info['lora_config']).to(device)

        print(generate_batch_completions([prompt], client_model, model_info['tokenizer'])[0])
        print('------------------------')

        del client_model
        torch.cuda.empty_cache()
    
    # FedAvg model
    print(generate_batch_completions([prompt], global_agg_model, model_info['tokenizer'])[0])
    print('------------------------')


