#encoding:utf-8
from collections import Counter

class ExecutionEngine(object):
    def __init__(self):
        pass

    def _compute_dependencies(self, function): # dependencies 里面 
        dependencies = {}
        seen = {function}
        queue = [function]
        while len(queue) > 0:
            fn = queue.pop()
            for prev_fn, arg_id in fn.previous_functions:
                if prev_fn not in dependencies:
                    dependencies[prev_fn] = [Counter() for _ in prev_fn.output_ids]
                output_idx = prev_fn.output_ids[arg_id]
                dependencies[prev_fn][output_idx][fn] += 1 #给定函数和输出idx才能位置指定一个函数调用,当然这种页可能重复,所以搞一个计数.因为函数可能重复利用.  #这个计数的含义是, 这个函数被多少个函数当做pre函数. 之后求导的时候, 没对这个求一次导数他就-1. 如果到0,就表示没有函数再逆向时候经过这个函数,也就可以析构这个函数了!!!!!!!!!本质就是引用计数.
                if prev_fn not in seen:
                    queue.append(prev_fn)
                    seen.add(prev_fn)
        return dependencies

    def _free_backward_dependency(self, dependencies, prev_fn, fn, arg_id):
        deps = dependencies[prev_fn]
        output_idx = prev_fn.output_ids[arg_id]
        output_deps = deps[output_idx]
        output_deps[fn] -= 1 #表示这个函数已经被他的一个pre函数访问了一次,所以引用计数-1
        if output_deps[fn] == 0:
            del output_deps[fn]
        return output_idx

#判断function的前置导数是否都算完了.  查看他的依赖
    def _is_ready_for_backward(self, dependencies, function):
        for deps in dependencies[function]:#看function对于各个偏导数是否还有引用计数即可.
            if len(deps) > 0:
                return False
        return True
#这个算法,是算variable的提付.
    def run_backward(self, variable, grad):
        ready = [(variable.creator, (grad,))]
        not_ready = {}

        dependencies = self._compute_dependencies(variable.creator)

        while len(ready) > 0:
            fn, grad = ready.pop()
            # TODO: double-buffering
            grad_input = fn._do_backward(*grad)# grad是逆向传导的传入导数.进行一步dp, grad_input是fn的梯度. 下面for循环来算fn之后的梯度,用的就是更新ready数组.
            for (prev_fn, arg_id), d_prev_fn in zip(fn.previous_functions, grad_input): # prev_fn : pre函数,  arg_id: pre函数的id,  d_prev_fn是pre函数的传入导数.  #下面代码用来更新ready , 主要就是判断prev_fn这个函数,对于arg_id的偏导数是否需要计算!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!这句话很重要!!!!!!!!!!!!对于理解.
                if not prev_fn.requires_grad:
                    assert d_prev_fn is None
                    continue
                output_nr = self._free_backward_dependency(dependencies, prev_fn, fn, arg_id) #析构掉没用的. 返回变量在函数里面的索引.
                is_ready = self._is_ready_for_backward(dependencies, prev_fn)
                if is_ready: #更新这2个数组.
                    if prev_fn in not_ready:#如果上一次遍历时候他在not_ready.
                        prev_grad = not_ready[prev_fn]#取出他上一个梯度值.加上这次的.
                        if not prev_grad[output_nr]:#如果之前没有这个值.
                            prev_grad[output_nr] = d_prev_fn
                        else:
                            prev_grad[output_nr].add_(d_prev_fn)
                        del not_ready[prev_fn]
                    else: # 如果上一次ready或者没有遍历到,那么就记录本次梯度即可.
                        assert output_nr == 0
                        prev_grad = (d_prev_fn,)
                    ready.append((prev_fn, prev_grad))#让他进入下次bfs
                else:#如果本次非ready
                    if prev_fn in not_ready: #如果上次在非ready
                        prev_grad = not_ready[prev_fn] #本次梯度写入notready里面.以后会在62行触发加法.
                    else:
                        prev_grad = [None for _ in prev_fn.output_ids]#说名这次也不要算.所以none
                        #把梯度放到not_ready里面.
                    if not prev_grad[output_nr]:
                        prev_grad[output_nr] = d_prev_fn
                    else:
                        prev_grad[output_nr].add_(d_prev_fn)

                    not_ready[prev_fn] = prev_grad


