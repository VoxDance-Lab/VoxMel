import logging
import os.path

import torch
from omegaconf import ListConfig

from voxcore.utils.misc import deep_update
from voxcore.register import tables
from voxcore.train_utils.set_all_random_seed import set_all_random_seed
from voxcore.train_utils.load_pretrained_model import load_pretrained_model


class AutoModel:

    def __init__(self, **kwargs):

        log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
        logging.basicConfig(level=log_level)

        model, kwargs = self.build_model(**kwargs)

        self.kwargs = kwargs
        self.model = model
        self.model_path = kwargs.get("model_path")

    def build_model(self, **kwargs):
        assert "model" in kwargs

        set_all_random_seed(kwargs.get("seed", 0))

        from voxcore.train_utils.device_funcs import resolve_device
        device = resolve_device("cpu" if kwargs.get("ngpu", 1) == 0 else kwargs.get("device", "auto"))
        kwargs["device"] = str(device)

        torch.set_num_threads(kwargs.get("ncpu", 4))

        tokenizer = kwargs.get("tokenizer", None)
        kwargs["tokenizer"] = tokenizer

        if tokenizer is not None:
            kwargs["vocab_size"] = -1
            tokenizers = (
                tokenizer.split(",") if isinstance(tokenizer, str) else tokenizer
            )
            tokenizers_conf = kwargs.get("tokenizer_conf", {})
            tokenizers_build = []
            vocab_sizes = []
            token_lists = []

            token_list_files = kwargs.get("token_lists", [])
            seg_dicts = kwargs.get("seg_dicts", [])

            if not isinstance(tokenizers_conf, (list, tuple, ListConfig)):
                tokenizers_conf = [tokenizers_conf] * len(tokenizers)

            for i, tokenizer in enumerate(tokenizers):
                tokenizer_class = tables.tokenizer_classes.get(tokenizer)
                tokenizer_conf = tokenizers_conf[i]

                if len(token_list_files) > 1:
                    tokenizer_conf["token_list"] = token_list_files[i]
                if len(seg_dicts) > 1:
                    tokenizer_conf["seg_dict"] = seg_dicts[i]

                tokenizer = tokenizer_class(**tokenizer_conf)

                tokenizers_build.append(tokenizer)
                token_list = tokenizer.token_list if hasattr(tokenizer, "token_list") else None
                token_list = (
                    tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else token_list
                )
                vocab_size = -1
                if token_list is not None:
                    vocab_size = len(token_list)

                if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
                    vocab_size = tokenizer.get_vocab_size()
                token_lists.append(token_list)
                vocab_sizes.append(vocab_size)

            if len(tokenizers_build) <= 1:
                tokenizers_build = tokenizers_build[0]
                token_lists = token_lists[0]
                vocab_sizes = vocab_sizes[0]

            kwargs["tokenizer"] = tokenizers_build
            kwargs["vocab_size"] = vocab_sizes
            kwargs["token_list"] = token_lists

        frontend = kwargs.get("frontend", None)

        if frontend is not None:
            kwargs["input_size"] = None
            frontend_class = tables.frontend_classes.get(frontend)
            frontend = frontend_class(**kwargs.get("frontend_conf", {}))
            kwargs["input_size"] = (
                frontend.output_size() if hasattr(frontend, "output_size") else None
            )
        kwargs["frontend"] = frontend

        model_class = tables.model_classes.get(kwargs["model"])
        assert model_class is not None, f'{kwargs["model"]} is not registered'
        model_conf = {}
        deep_update(model_conf, kwargs.get("model_conf", {}))
        deep_update(model_conf, kwargs)
        model = model_class(**model_conf)

        init_param = kwargs.get("init_param", None)
        if init_param is not None:
            if os.path.exists(init_param):
                logging.info(f"Loading pretrained params from {init_param}")
                load_pretrained_model(
                    model=model,
                    path=init_param,
                    ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
                    oss_bucket=kwargs.get("oss_bucket", None),
                    scope_map=kwargs.get("scope_map", []),
                    excludes=kwargs.get("excludes", None),
                    use_ema=kwargs.get("use_ema", False),
                )
            else:
                raise FileNotFoundError(f"init_param does not exist: {init_param}")

        if kwargs.get("fp16", False):
            model.to(torch.float16)
        elif kwargs.get("bf16", False):
            model.to(torch.bfloat16)
        model.to(device)

        if not kwargs.get("disable_log", True):
            tables.print()

        return model, kwargs
