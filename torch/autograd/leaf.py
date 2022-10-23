from collections import OrderedDict
from .function import Function

class Leaf(Function):

    def __init__(self, variable, requires_grad):
        self.variable = variable # leaf节点也是variable.
        self.output_ids = {id(variable): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad
        self.backward_hooks = OrderedDict()

    def _do_forward(self, *input):
        raise NotImplementedError  #这种写法就是如果一个东西调用这个必须覆盖这个方法否则会报错.

    def _do_backward(self, *grad_output): # grad_output是bp时候传进来的导数.
        assert len(grad_output) == 1
        for hook in self.backward_hooks.values():
            hook(grad_output, grad_output)
        self.variable.grad.add_(grad_output[0])  # self.variable.grad 是一个tensor. 所以加上他的值.给这个variable用即可.
        return tuple()
