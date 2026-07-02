from lm_eval.api.model import TokenLM, TokenSequence
from lm_eval.api import utils
from tqdm import tqdm
import random
from typing import List, Mapping, NewType, Optional, Tuple, Union, Dict
from pathlib import Path
import sys
import os
import torch
import transformers
import pickle
from transformers import GPT2TokenizerFast, GPT2Tokenizer, AutoTokenizer
from huggingface_hub import login as hf_login

# NANOGPT_ROOT = str(Path(__file__).parents[4])
# sys.path.append(NANOGPT_ROOT)

NANOGPT_ROOT = "/home/fleetprobe/fmt-analysis"
TOKENIZER_ROOT = "/home/fleetprobe/fmt-analysis/data"
sys.path.insert(0, NANOGPT_ROOT)
from model_HF import GPT, GPTConfig


class NanoGPTLM(TokenLM):

    #AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    #AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
    #AUTO_MODEL_CLASS: transformers.AutoModel = None

    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        out_dir: str = None,
        data_dir: str = None,
        model_name: str = None,
        batch_size: Optional[int] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        device: Optional[Union[int, str]] = "cuda",
    ):
        super().__init__()
        assert isinstance(device, str)
        assert isinstance(batch_size, int)
        assert batch_size == 1, "Batch size > 1 not supported yet"

        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._device = device

        hf_token = os.environ.get("HF_ACCESS_TOKEN")
        if hf_token:
            hf_login(token=hf_token)

        HF_REPO_ROOT = "fmtmodels"
        if model_name is None:
            raise ValueError("model_name must be provided for HuggingFace loading.")
        if data_dir is None:
            raise ValueError("data_dir must be provided (e.g. babylm_full_bpe_8k).")

        model_id = f"{HF_REPO_ROOT}/{model_name}"
        self.model = GPT.from_pretrained(model_id, use_safetensors=False)
        self.model.eval().to(self._device)

        self._add_special_tokens = add_special_tokens
        self.tokenizer = self._create_auto_tokenizer(None, data_dir)

        self._vocab_size = self.model.transformer.wte.weight.size(0)
        self._block_size = self.model.transformer.wpe.weight.size(0)

        tok_vocab = getattr(self.tokenizer, "vocab_size", None) or len(self.tokenizer)
        if tok_vocab != self._vocab_size:
            raise ValueError(
                f"Tokenizer/model vocab mismatch: tokenizer={tok_vocab}, model={self._vocab_size}."
            )

        self.tokenizer.model_max_length = (
            self._block_size if self._max_length is None else min(self._max_length, self._block_size)
        )

    def load_tokenizer(self, data_dir):
        # Match notebook behavior:
        # - if relative, resolve under TOKENIZER_ROOT
        # - require meta.pkl + custom_tokenizer
        tok_dir = data_dir if os.path.isabs(data_dir) else os.path.join(TOKENIZER_ROOT, data_dir)
        meta_path = os.path.join(tok_dir, "meta.pkl")

        if not os.path.exists(meta_path):
            raise NotImplementedError(f"No meta.pkl found at: {meta_path}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        if not meta.get("custom_tokenizer", False):
            if meta.get("stoi", False):
                raise NotImplementedError("stoi/itos not supported yet")
            raise NotImplementedError("No custom tokenizer/stoi/itos found in meta.pkl")

        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=False, local_files_only=True)

        if not tokenizer.eos_token:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _create_auto_tokenizer(
        self,
        meta_path: str,
        data_dir: str,
    ) -> transformers.PreTrainedTokenizer:
        return self.load_tokenizer(data_dir)

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.

        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            #Set to False by default
            return False

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        # seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx", "block_size") #Add block_size because it is in the GPTConfig
        # for attr in seqlen_config_attrs:
        #     if hasattr(self._config, attr):
        #         return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH


    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)
        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False), self.batch_size
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop_sequences = request_args["stop_sequences"]
            max_generation_length = request_args["max_generation_length"]
            num_fewshot = request_args["num_fewshot"]

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None
            assert isinstance(num_fewshot, int) or num_fewshot is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None or num_fewshot == 0:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            token_context = self.tok_encode_batch(context)

            responses = self._model_generate(
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)
        return reorder.get_original(results)


    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        # lm-eval may pass BatchEncoding or tensor
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.long()

        # prevent position embedding OOB
        if input_ids.size(1) > self._block_size:
            input_ids = input_ids[:, -self._block_size:]

        # fail fast on token OOB
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0 or max_id >= self._vocab_size:
            raise ValueError(
                f"Token id out of range: min={min_id}, max={max_id}, vocab_size={self._vocab_size}. "
                f"Likely tokenizer/model mismatch."
            )

        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            out = self.model(input_ids, input_ids)
        logits = out["logits"]  # model_HF forward returns dict-like output

        return logits
    
    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        # Ensure that the context does not encroach into the `space`
        # for the generation.
        input_ids = inputs["input_ids"][:, self.max_gen_toks - self.max_length :]
        attention_mask = inputs["attention_mask"][
            :, self.max_gen_toks - self.max_length :
        ] #Understand this 2
        #Can't use it anyway, so don't pass because since batch size is always one it will never be padded anyway
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # stopping_criteria = stop_sequences_criteria(
        #     self.tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
        # ) #Understand this

        generations = self.model.generate(
            input_ids=input_ids,
            #attention_mask=attention_mask,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )


def _get_dtype(
    dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible."""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype



class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][
            :, -self.sequence_id_len :
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


#ADDED THESE TWO BUT I DOUBT IT WILL BE USED

def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )
