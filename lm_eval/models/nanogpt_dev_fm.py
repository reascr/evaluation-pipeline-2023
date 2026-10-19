from lm_eval.api.model import TokenLM, TokenSequence
from lm_eval.api import utils
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
import os
import sys
import pickle
import torch
import transformers
from transformers import GPT2Tokenizer

FMT_LLM_ROOT = "/home/andsch1/fmt-llm"
sys.path.insert(0, FMT_LLM_ROOT)

from model import GPT, GPTConfig
import inspect

print(f"GPT implementation loaded from: {inspect.getfile(GPT)}")

class NanoGPTDevFMLM(TokenLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        out_dir=None,
        data_dir=None,
        checkpoint_name="ckpt.pt", # change here later
        batch_size=1,
        max_gen_toks=256,
        max_length=None,
        add_special_tokens=None,
        device="cuda",
        model_name=None,  # compatibility with babylm_eval.py
    ):
        super().__init__()

        if out_dir is None:
            raise ValueError("out_dir must be provided")
        if data_dir is None:
            raise ValueError("data_dir must be provided")
        if batch_size != 1:
            raise ValueError("Only batch_size=1 is supported")

        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._add_special_tokens = add_special_tokens
        self._device = device

        ckpt_path = os.path.join(out_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(f"Checkpoint file: {ckpt_path}")
        print(f"Checkpoint step: {checkpoint['iter_num']}")
        print(f"Checkpoint developmental: {checkpoint['config'].get('wm_developmental')}")
        print(f"Checkpoint seed: {checkpoint['config'].get('torch_seed_default')}")

        self.iter_num = checkpoint["iter_num"]
        model_args = checkpoint["model_args"].copy()

        # Backward compatibility with older checkpoints
        model_args.setdefault("head_size_qkv", model_args["n_embd"])
        model_args.setdefault("wm_decay_length", model_args["block_size"])
        model_args.setdefault("ffw_dim", model_args["n_embd"] * 4)
        model_args.setdefault("wm_developmental", False)
        model_args.setdefault("wm_decay_rate_initial", 6.0)
        model_args.setdefault("wm_decay_rate_adult", 1.0)
        model_args.setdefault("wm_decay_development_rate", 0.9999)

        self._config = GPTConfig(**model_args)
        self.model = GPT(self._config)

        state_dict = checkpoint["model"]
        prefix = "_orig_mod."
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)

        run_config = checkpoint.get("config", {})
        developmental = run_config.get(
            "wm_developmental",
            model_args.get("wm_developmental", False),
        )

        if developmental:
            alpha0 = run_config.get(
                "wm_decay_rate_initial",
                model_args["wm_decay_rate_initial"],
            )
            alpha_adult = run_config.get(
                "wm_decay_rate_adult",
                model_args["wm_decay_rate_adult"],
            )
            r = run_config.get(
                "wm_decay_development_rate",
                model_args["wm_decay_development_rate"],
            )
            alpha_value = alpha_adult + (alpha0 - alpha_adult) * (r ** self.iter_num)
            model_label = "developmental fleeting memory"
        else:
            alpha_value = run_config.get(
                "wm_decay_rate",
                model_args.get("wm_decay_rate"),
            )
            model_label = "static fleeting memory"


        if alpha_value is None:
            raise ValueError("Could not determine decay rate for checkpoint")

        # CPU scalar tensor, matching the training-time compile-compatible setup
        self.current_decay_rate = torch.tensor(alpha_value, dtype=torch.float32)
        
        print(f"Evaluation alpha: {self.current_decay_rate.item():.6f}")

        print(
            f"Loaded {model_label} | step={self.iter_num} | alpha={alpha_value:.6f}"
        )

        meta_path = os.path.join(data_dir, "meta.pkl")
        self.tokenizer = self._create_auto_tokenizer(meta_path, data_dir)
        self.tokenizer.model_max_length = self.max_length

        self.model.eval()
        torch.set_grad_enabled(False)
        self.model.to(self._device)

    def _create_auto_tokenizer(self, meta_path, data_dir):
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            if meta.get("custom_tokenizer", False):
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    data_dir,
                    use_fast=False,
                    local_files_only=True,
                )
            elif meta.get("stoi", False):
                raise NotImplementedError("stoi/itos tokenizer not supported")
            else:
                raise NotImplementedError("Unsupported tokenizer in meta.pkl")
        else:
            tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        if not tokenizer.eos_token:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    @property
    def add_special_tokens(self):
        return False if self._add_special_tokens is None else self._add_special_tokens

    @property
    def eot_token(self):
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def max_length(self):
        if self._max_length is not None:
            return self._max_length
        for attr in ("n_positions", "max_position_embeddings", "n_ctx", "block_size"):
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string):
        return self.tokenizer.encode(
            string,
            add_special_tokens=self.add_special_tokens,
        )

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inputs, labels=None):
        # BLiMP needs logits for every token position, so inputs are also passed
        # as targets. The developmental/static alpha for this checkpoint is
        # passed explicitly into fmt-llm/model.py.
        logits, _ = self.model(
            inputs,
            inputs,
            current_decay_rate=self.current_decay_rate,
        )
        return logits

    def greedy_until(self, requests):
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]

            stop_sequences = request_args["stop_sequences"]
            max_generation_length = request_args["max_generation_length"]
            num_fewshot = request_args["num_fewshot"]

            until = [self.eot_token] if stop_sequences is None or num_fewshot == 0 \
                else stop_sequences + [self.eot_token]

            max_tokens = (
                self.max_gen_toks
                if max_generation_length is None
                else max_generation_length
            )

            token_context = self.tok_encode_batch(context)
            responses = self._model_generate(token_context, max_tokens, until)
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                for term in until:
                    response = response.split(term)[0]
                self.cache_hook.add_partial(
                    "greedy_until",
                    (context, until),
                    response,
                )
                results.append(response)

        return reorder.get_original(results)

    def _model_generate(self, inputs, max_tokens, stop=None):
        input_ids = inputs["input_ids"][
            :, self.max_gen_toks - self.max_length:
        ].to(self.device)

        generations = self.model.generate(
            idx=input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            current_decay_rate=self.current_decay_rate,
        )

        return utils.select_continuation_from_batch_left_padding(
            generations,
            max_context_size=inputs["input_ids"].size(1),
        )