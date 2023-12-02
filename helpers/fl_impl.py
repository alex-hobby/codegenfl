from transformers import AutoModelForCausalLM
import torch
from peft import prepare_model_for_kbit_training, get_peft_model


def get_fed_avg_model(SAVE_PATH, model_info, num_clients=2):
    """
    Get the aggregated model from the clients' weights

    :param str SAVE_PATH: location to load model checkpoint from w/o client index appended
    :param model_info: object containing information about model
    :param num_clients: number of clients to load
    :return: the aggregated model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type: {device}")

    model_name, lora_config, quant_config = (
        model_info['model_name'],
        model_info['lora_config'],
        model_info['quant_config']
    )

    aggregated_weight_dict = aggregate_weights(SAVE_PATH, lora_config, quant_config, device, num_clients)

    aggregated_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map={"": 0}
    )
    aggregated_model = prepare_model_for_kbit_training(aggregated_model)
    global_aggregated_model = get_peft_model(aggregated_model, lora_config).to(device)

    global_aggregated_model.load_state_dict(aggregated_weight_dict)

    del aggregated_weight_dict
    torch.cuda.empty_cache()

    return global_aggregated_model


def aggregate_weights(SAVE_PATH, lora_config, quant_config, device, num_clients=2):
    """
    Function to aggregate weights using FedAvg method

    :param str SAVE_PATH: location to load model checkpoint from w/o client index appended
    :param lora_config: config object with lora information
    :param quant_config: config object with quantization information
    :param device: cuda or cpu
    :param num_clients: number of clients to load
    :return: the weights as a dictionary
    """
    weight_dict = {}

    for client_idx in range(num_clients):
        print("--------------------------------")
        print(f"Processing client {client_idx}")
        # Load client
        print(f"Loading model from {SAVE_PATH}_{client_idx}")
        client_model = AutoModelForCausalLM.from_pretrained(
            SAVE_PATH + f"_{client_idx}", quantization_config=quant_config, device_map={"": 0}
        )
        print("Loaded model")
        client_model = prepare_model_for_kbit_training(client_model)
        client_model = get_peft_model(client_model, lora_config).to(device)

        print("Updating state dict")
        for k,v in client_model.state_dict().items():
            if isinstance(v, str):
                  weight_dict[k] = v
            else:
                if client_idx == 0:       # For the first client, we need to initialize the value.
                    weight_dict[k] = 0
                weight_dict[k] += v/torch.tensor(num_clients)

        del client_model
        torch.cuda.empty_cache()

    return weight_dict