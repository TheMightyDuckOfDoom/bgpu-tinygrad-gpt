#!/usr/bin/env python3

from tinygrad import Tensor, nn, dtypes

import os
import numpy as np
#os.environ["BGPU"] = "1"
os.environ["DEBUG"] = "7"

is_bgpu = True #os.environ.get("BGPU") == "1"

first_dim = 4 #28
second_dim = 8

class LinearNet:
  def __init__(self):
    if is_bgpu:
        self.l1 = Tensor(np.load("l1.npy"))
        self.l2 = Tensor(np.load("l2.npy"))
    else:
        self.l1 = Tensor.rand(first_dim*first_dim, second_dim)
        self.l2 = Tensor.rand(second_dim, 10)
        np.save("l1.npy", self.l1.numpy())
        np.save("l2.npy", self.l2.numpy())
  def __call__(self, x:Tensor) -> Tensor:
    return x.flatten(1).dot(self.l1).relu().dot(self.l2)

model = LinearNet()
optim = nn.optim.Adam([model.l1, model.l2], lr=0.001)

x, y = None, None

if is_bgpu:
    x = Tensor(np.load("x.npy"))
    y = Tensor(np.load("y.npy"))
else:
    x, y = Tensor.rand(4, 1, first_dim, first_dim), Tensor([2,4,3,7])  # replace with real mnist dataloader
    np.save("x.npy", x.numpy())
    np.save("y.npy", y.numpy())

with Tensor.train():
  for i in range(20):
    optim.zero_grad()
    loss = model(x).sparse_categorical_crossentropy(y).backward()
    optim.step()
    print(i, loss.numpy())
