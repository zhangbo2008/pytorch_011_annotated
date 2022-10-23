#encoding:utf-8
import unittest

from common import make_jacobian, TestCase, iter_tensors, get_numerical_jacobian
from torch.autograd.functions import *

PRECISION = 1e-3

def iter_gradients(x):
    if isinstance(x, Variable):
        yield x.grad
    else:
        for elem in x:
            for result in iter_gradients(elem):
                yield result

def zero_gradients(i):
    for t in iter_gradients(i):
        t.zero_()

def get_analytical_jacobian(input, output):
    jacobian = make_jacobian(input, output.numel())
    grad_output = output.data.clone().zero_()
    flat_grad_output = grad_output.view(-1)

    for i in range(flat_grad_output.numel()): # numel: number elements
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        zero_gradients(input)#让变量的梯度都变成0.
        output.backward(grad_output) 
        for jacobian_x, d_x in zip(jacobian, iter_gradients(input)):
            jacobian_x[:,i] = d_x

    return jacobian

class TestAutograd(TestCase):

    def test_hooks(self):
        x = Variable(torch.ones(5, 5))
        y = Variable(torch.ones(5, 5) * 4)

        counter = [0]
        def bw_hook(inc, grad_input, grad_output):
            counter[0] += inc

        z = x ** 2 + x * 2 + x * y + y
        z.register_hook('test', lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5))
        self.assertEqual(counter[0], 1)

        z.register_hook('test2', lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5))
        self.assertEqual(counter[0], 4)

        z.remove_hook('test2')
        z.backward(torch.ones(5, 5))
        self.assertEqual(counter[0], 5)


L = 20
M = 10
S = 5
tests = [
    (Add, (), ((M, M), (M, M))),
    (Sub, (), ((M, M), (M, M))),
    (Mul, (), ((M, M), (M, M))),
    (Div, (), ((M, M), torch.rand(M, M) + 1e-2)),
    (Pow, (), (torch.rand(M, M), torch.rand(M, M) + 0.1)),
    (AddConstant, (3.14,), ((L, L),)),
    (SubConstant, (3.14,), ((L, L),)),
    (SubConstant, (3.14, True), ((L, L),), 'from_tensor'),
    (MulConstant, (3.14,), ((L, L),)),
    (DivConstant, (3.14, True), (torch.rand(L, L) + 1e-2,), 'by_tensor'),
    (PowConstant, (3.14,), (torch.rand(L, L),)),
    (Transpose, (0, 1), (torch.rand(L, L),)),
    (Transpose, (2, 0), (torch.rand(S, S, S),), '3d'),
    (Index, (1, 2), (torch.rand(S, S, S),)),
    (Index, (slice(0, 3),), (torch.rand(S, S, S),), 'slice'),
    (View, (S*S, S), (torch.rand(S, S, S),)),
    (Exp,  (), (torch.rand(S, S, S),)),
    (Log,  (), (torch.rand(S, S, S) + 1e-2,)),
    (Log1p,  (), (torch.rand(S, S, S),)),
]

def create_input(call_args):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)
    def map_arg(arg): #把每一个变量都用Variable包起来.
        if isinstance(arg, tuple):
            return Variable(torch.randn(*arg))
        else:
            return Variable(arg)
    return tuple(map_arg(arg) for arg in call_args)

#下面这个for循环是测试的核心代码,打断点进入即可. do_test断点即可.
for test in tests:
    cls, constructor_args, call_args = test[:3] # cls是函数,constructor_args是函数的构造参数,call_args是函数的调用参数.
    def do_test(self, cls=cls, constructor_args=constructor_args, call_args=call_args):
        input = create_input(call_args)
        output = cls(*constructor_args)(*input)
        for i, o in enumerate(output):
            analytical = get_analytical_jacobian(input, o)
            def fn(input):
                return cls(*constructor_args)(*input)[i].data
            numerical = get_numerical_jacobian(fn, input, input)
            self.assertLessEqual(
                max(a.add(-1, n).abs().max() for a, n in zip(analytical, numerical)),
                PRECISION
            )

    test_name = 'test_' + cls.__name__ + ('_' + test[3] if len(test) == 4 else '')
    assert not hasattr(TestAutograd, test_name), 'Two tests have the same name: ' + test_name
    setattr(TestAutograd, test_name, do_test)


if __name__ == '__main__':
    unittest.main() #这个是test入口
