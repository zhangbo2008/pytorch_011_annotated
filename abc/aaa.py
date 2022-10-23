#!/usr/bin/python
# -*- coding: utf-8 -*-  
print "你好"  
import torch

import torch

from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])            # build a tensor
variable = Variable(tensor, requires_grad=True) # build a variable, usually for compute gradients

print(tensor)       
print(variable) #可能是因为版本的问题，显示和视频中不太一样
v_out = torch.mean(variable*variable)   # x^2
v_out.backward()    # backpropagation from v_out
print(variable.grad)
