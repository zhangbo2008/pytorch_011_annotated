#encoding:utf-8
from collections import OrderedDict
from .variable import Variable
//一个函数有do_forward的方法.
class Function(object):

    def __init__(self):
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None
        self.backward_hooks = OrderedDict()

    def __call__(self, *input):
        return self._do_forward(*input)

    def _do_forward(self, *input): # 底层用forward来实现.
        unpacked_input = tuple(arg.data for arg in input)
        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output,)
        self.needs_input_grad = tuple(arg.creator.requires_grad for arg in input)
        self.requires_grad = any(self.needs_input_grad) # 有一个需要算,那么我们就计算.
        output = tuple(Variable(tensor, self) for tensor in raw_output) # 把结果用variable再包起来.

        self.previous_functions = [(arg.creator, id(arg)) for arg in input]   #arg的上一层函数.设置好.forward时候写入他的父节点到previous_func里面.
        self.output_ids = {id(var): i for i, var in enumerate(output)}
        return output

    def _do_backward(self, grad_output):
        grad_input = self.backward(grad_output)
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input,)  //数据tuple化
        assert len(grad_input) == len(self.previous_functions), \
            self.__class__.__name__ + ' returned an invalid number of gradient tensors'

        for hook, idx in self.backward_hooks.values(): # 运行hook,hook就是每一次back后,运行这些东西.挂载到backward之后出发. 可能debug用吧.
            gi = grad_input if idx is None else grad_input[idx]
            hook(grad_input, grad_output)

        return grad_input

    def register_hook(self, name, hook, variable=None):
        assert name not in self.backward_hooks, \
            "Trying to register a second hook with name {}".format(name)
        variable_idx = self.output_ids[id(variable)] if variable else None
        self.backward_hooks[name] = (hook, variable_idx)

    def remove_hook(self, name):
        assert name in self.backward_hooks, \
            "Trying to remove an inexistent hook with name {}".format(name)
        del self.backward_hooks[name]

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError
