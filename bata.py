import numpy as np
import matplotlib.pyplot as plt
import torch


a = torch.tensor([[1,1,1],[2,2,2]], dtype=torch.float32)
b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
c = torch.mul(a,b)
print(c)
