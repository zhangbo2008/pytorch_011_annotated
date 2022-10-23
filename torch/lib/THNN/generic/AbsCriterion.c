#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/AbsCriterion.c"
#else
//abs度量函数的forward. 也就是output和target的差.
void THNN_(AbsCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage)
{
  real sum = 0;

  TH_TENSOR_APPLY2(real, input, real, target,
    sum += fabs(*input_data - *target_data);
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(AbsCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage)
{
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    *gradInput_data = (*input_data - *target_data) >= 0 ? norm : -norm;
  );// 我们看abs度量的梯度算法: 大于等于0是1,小于0是-1. 这很显然, 就是导数而已.
}

#endif
