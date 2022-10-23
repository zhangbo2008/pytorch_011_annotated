#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Abs.c"
#else
//THNN里面的函数都是2套, 比如下面这两个,一个是forward, 一个是backward函数.
void THNN_(Abs_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);
  THTensor_(abs)(output, input);
}
//gradInput 的梯度更新一下带上abs函数.   注意gradOutput是下一层网络传进来的梯度,gradInput是这一层传出去的梯度.注意命名.
void THNN_(Abs_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput)
{
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
    real z = *input_data;
    *gradInput_data = *gradOutput_data * (z >= 0 ? 1 : -1);
  );//其实就是data
}

#endif
