import os
import io
import shutil
import logging
from collections import OrderedDict
import numpy as np
from omegaconf import OmegaConf
import torch

def statistic_model_parameters(model, prefix=None):
    var_dict = model.state_dict()
    numel = 0
    for i, key in enumerate(
        sorted(list([x for x in var_dict.keys() if "num_batches_tracked" not in x]))
    ):
        if prefix is None or key.startswith(prefix):
            numel += var_dict[key].numel()
    return numel

def int2vec(x, vec_dim=8, dtype=np.int32):
    b = ("{:0" + str(vec_dim) + "b}").format(x)

    return (np.array(list(b)[::-1]) == "1").astype(dtype)

def seq2arr(seq, vec_dim=8):
    return np.row_stack([int2vec(int(x), vec_dim) for x in seq])

def load_scp_as_dict(scp_path, value_type="str", kv_sep=" "):
    with io.open(scp_path, "r", encoding="utf-8") as f:
        ret_dict = OrderedDict()
        for one_line in f.readlines():
            one_line = one_line.strip()
            pos = one_line.find(kv_sep)
            key, value = one_line[:pos], one_line[pos + 1 :]
            if value_type == "list":
                value = value.split(" ")
            ret_dict[key] = value
        return ret_dict

def load_scp_as_list(scp_path, value_type="str", kv_sep=" "):
    with io.open(scp_path, "r", encoding="utf8") as f:
        ret_dict = []
        for one_line in f.readlines():
            one_line = one_line.strip()
            pos = one_line.find(kv_sep)
            key, value = one_line[:pos], one_line[pos + 1 :]
            if value_type == "list":
                value = value.split(" ")
            ret_dict.append((key, value))
        return ret_dict

def deep_update(original, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            if len(value) == 0:
                original[key] = value
            deep_update(original[key], value)
        else:
            original[key] = value

def prepare_model_dir(**kwargs):

    os.makedirs(kwargs.get("output_dir", "./"), exist_ok=True)

    yaml_file = os.path.join(kwargs.get("output_dir", "./"), "config.yaml")
    OmegaConf.save(config=kwargs, f=yaml_file)
    logging.info(f"kwargs: {kwargs}")
    logging.info("config.yaml is saved to: %s", yaml_file)

    model_path = kwargs.get("model_path", None)
    if model_path is not None:
        config_json = os.path.join(model_path, "configuration.json")
        if os.path.exists(config_json):
            shutil.copy(
                config_json, os.path.join(kwargs.get("output_dir", "./"), "configuration.json")
            )

def extract_filename_without_extension(file_path):

    filename_with_extension = os.path.basename(file_path)

    filename, extension = os.path.splitext(filename_with_extension)

    return filename

def smart_remove(path):

    if not os.path.exists(path):
        print(f"{path} does not exist.")
        return

    if os.path.isfile(path):
        os.remove(path)
        print(f"File {path} has been deleted.")

    elif os.path.isdir(path):
        try:

            os.rmdir(path)
            print(f"Empty directory {path} has been deleted.")
        except OSError:

            shutil.rmtree(path)
            print(f"Non-empty directory {path} has been recursively deleted.")

def tensor_to_scalar(x):
    if torch.is_tensor(x):
        return x.detach().item()
    return x

def get_dtype(dtype: str):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
