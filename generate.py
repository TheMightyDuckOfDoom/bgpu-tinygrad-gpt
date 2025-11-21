#!/usr/bin/env python3

import os
#os.environ["BGPU"] = "1"
os.environ["DEBUG"] = "7"
os.environ["NOOPT"] = "0"

from utils import load_state_dict_from_numpy
from tinygrad.nn.state import load_state_dict
from model import GPT
from tinygrad.nn.state import get_parameters
import tiktoken
from tinygrad import Tensor

Tensor.manual_seed(42)

model = GPT(
    vocab_size=50304,
    block_size=256,
    n_layers=4,
    n_heads=4,
    embed_size=64,
    hidden_size=64 * 4,
    bias=True,
)

state_dict = load_state_dict_from_numpy("model")
load_state_dict(model, state_dict)

print(f"model has {sum(p.numel() for p in get_parameters(model)):,} parameters")
tokenizer = tiktoken.get_encoding("gpt2")
input_ids = Tensor([tokenizer.encode("What is the horoscope for Tobias who is an gemini?")])
tokens = model.generate(idx=input_ids, max_new_tokens=1)
print(tokenizer.decode(tokens[0].numpy()))
