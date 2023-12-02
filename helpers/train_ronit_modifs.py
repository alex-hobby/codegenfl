from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import torch

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import os
import pickle

import time
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR


def generic_training_runner(
    SAVE_PATH,
    LOSSES_PATH,
    TIMES_PATH,
    model_info,
    training_args,
):
    """
    Run training procedure across multiple clients and generate client models

    :param SAVE_PATH: Directory to save checkpoints to.
    :param LOSSES_PATH: Directory to save all losses in.
    :param TIMES_PATH: Directory to save how long each client took to train.


    """
    # Ensure the files can be written without program crash.
    assert os.path.exists(os.path.dirname(SAVE_PATH))
    assert os.path.exists(os.path.dirname(LOSSES_PATH))
    assert os.path.exists(os.path.dirname(TIMES_PATH))

    # Args
    clients, MAX_LENGTH, BATCH_SIZE, conduct_logging, EPOCHS, lr, t_max = (
        training_args["clients"],
        training_args["MAX_LENGTH"],
        training_args["BATCH_SIZE"],
        training_args["conduct_logging"],
        training_args["EPOCHS"],
        training_args["lr"],
        training_args["t_max"],
    )
    (
        model_name,
        tokenizer,
        datasets_list,
        quant_config,
        lora_config,
    ) = (
        model_info["model_name"],
        model_info["tokenizer"],
        model_info["datasets_list"],
        model_info["quant_config"],
        model_info["lora_config"],
    )

    # Device setup
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type: {device}")

    # Obtain data loaders for each client (separate partitions of data)
    print("Loading Data and Producing DataLoader objects")
    data_loaders_clients = [
        torch.utils.data.DataLoader(
            d, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        for d in datasets_list
    ]
    print("Finished loading data")

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
        Train across the data.
        Log losses carefully
        Save model
    """

    def train_loop(client_idx):
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
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

        losses = []
        # Log a 10K losses
        dataloader_client = data_loaders_clients[client_idx]
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
            scheduler.step()
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
        print(f"Beginning training iteration for client {i}")
        train_loop(i)
