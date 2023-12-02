from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import torch

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import pickle

import time
from tqdm import tqdm


def generic_training_runner(
    SAVE_PATH,
    LOSSES_PATH,
    TIMES_PATH,
    model_info,
    training_args,
):
    """
    Function to perform finetuning on provided model_name from Hugging Face

    Models are checkpointed and saved at provided locations in Drive.

    :param str SAVE_PATH: Location to save model checkpoint
    :param str LOSSES_PATH: Location to log loss results
    :param str TIMES_PATH: Location to log elapsed time results
    :param model_info: Object containing information about model
    :param training_args: Object containing training arguments
    :return: None
    """
    # Args
    clients, MAX_LENGTH, conduct_logging, EPOCHS, lr = (
        training_args['clients'],
        training_args['MAX_LENGTH'],
        training_args['conduct_logging'],
        training_args['EPOCHS'],
        training_args['lr'],
    )
    model_name, tokenizer, client_dataloaders, quant_config, lora_config = (
        model_info['model_name'],
        model_info['tokenizer'],
        model_info['client_dataloaders'],
        model_info['quant_config'],
        model_info['lora_config'],
    )

    assert len(client_dataloaders) == clients

    # Device setup
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type: {device}")


    # Loggings of losses setup.
    def write_to_file(epoch_number, loss_values, avg_loss, file_path):
        # Open the file in binary append mode
        with open(file_path + ".txt", "ab") as file:
            # Serialize and write the epoch number
            pickle.dump(f"Epoch {epoch_number}", file)
            # Serialize and write the list of loss values
            pickle.dump(loss_values, file)
            # Average loss for the epoch
            pickle.dump(avg_loss, file)

    # Simple test to log.
    conduct_logging = True
    if conduct_logging:
        write_to_file(-1, [-1, -2, -3], [-2], LOSSES_PATH)

    # Begin training process
    # Traditional training loop requires modifications for FL...
    # To do this, we perform the following. Assume only 1 epoch is done.
    """
    Create a variable to aggregate gradients

    For Each client
        Copy the global model
        Fetch the client's dataloader
        Train across the data in one iteration.
        Log losses carefully
        Save model
        Aggregate change in weights to variable
    """

    def train_loop(client_idx):
        """
        Main training loop
        
        :param int client_idx: The index of the client being trained
        :return: None
        """
        global_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, device_map={"": 0}
        )
        global_model.gradient_checkpointing_enable()

        global_model = prepare_model_for_kbit_training(global_model)
        global_model = get_peft_model(global_model, lora_config).to(device)
        client_model = global_model
        print("Loaded global model.")

        # Set up loss and optimization unique to model.
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(client_model.parameters(), lr=lr)

        losses = []
        # Log a 10K losses
        dataloader_client = client_dataloaders[client_idx]
        log_every = max(1, (EPOCHS * len(dataloader_client)) // 10_000)
        print("Number of batches:", len(dataloader_client))
        client_model.train()
        start_time = time.time()
        print("Starting training")
        for epoch in range(EPOCHS):
            total_loss = 0
            for c, batch in tqdm(enumerate(dataloader_client)):
                # Batch size x Max Seq LEn
                sample = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=MAX_LENGTH,
                )["input_ids"].to(device)
                target = sample.detach()[:, 1:]
                sample = sample[:, :-1]

                # Batch size x Max Seq Len x Vocab Size
                optimizer.zero_grad()
                prediction = client_model(sample).logits

                # Ensure swapping of axes
                loss = criterion(prediction.transpose(1, 2), target)
                loss.backward()

                # Loss logging
                total_loss += loss.item()
                if c % log_every == 0:
                    print(f"Step: {c}, Loss: {loss.item():.4f}")
                    losses.append(loss.item())

                # Change model weights
                optimizer.step()

                # Explicit destruction (may not be needed after previous debugging)
                del loss, prediction, sample, target
                torch.cuda.empty_cache()

            print(f"Epoch {epoch} Complete")
            avg_loss = total_loss / len(dataloader_client)
            if conduct_logging:
                write_to_file(epoch, losses, avg_loss, LOSSES_PATH + f"_{client_idx}")
            losses.clear()

        print("Ending training")

        end_time = time.time()

        optimizer.zero_grad()

        elapsed_time = end_time - start_time
        if conduct_logging:
            with open(TIMES_PATH + f"_{client_idx}.txt", "a") as file:
                file.write(f"{elapsed_time}")

        print(elapsed_time)
        if conduct_logging:
            client_model.save_pretrained(SAVE_PATH + f"_{client_idx}")
            print("MODEL SAVED")
            del client_model
            torch.cuda.empty_cache()

    # Execute training for all clients
    for i in range(clients):
        print("Beginning training iteration")
        train_loop(i)
