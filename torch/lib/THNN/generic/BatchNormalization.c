#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BatchNormalization.c"
#else
//看bn的forward函数.  我理解是.因为batch_size是0维度,所以我们在1维度上的计算可以并行,所以我们下面12行,取dim=1,然后并行,每一个计算都取维度1. 例子就是 [1,2] [2,3] [3,4] batchsize就是3, 那么我们就可以 1,2,3 算一个归一化,   2,3,4 一组算一个归一化,2个可以并行,  最后就是3,3,mean 
void THNN_(BatchNormalization_updateOutput)(
  THNNState *state, THTensor *input, THTensor *output,
  THTensor *weight, THTensor *bias,
  THTensor *running_mean, THTensor *running_var,   //running_mean ,running_var 是上一次吃bn运算的记录下来,我们那他跟档次做一个均匀化.
  THTensor *save_mean, THTensor *save_std,
  bool train, double momentum, double eps)
{
  long nInput = THTensor_(size)(input, 1);
  long f,n = THTensor_(nElement)(input) / nInput;

  #pragma omp parallel for //并行跑下面for循环.因为之前互不干扰.
  for (f = 0; f < nInput; ++f) {
    THTensor *in = THTensor_(newSelect)(input, 1, f);
    THTensor *out = THTensor_(newSelect)(output, 1, f);

    real mean, invstd;

    if (train) {
      // compute mean per input
      accreal sum = 0;
      TH_TENSOR_APPLY(real, in, sum += *in_data;);

      mean = (real) sum / n;
      THTensor_(set1d)(save_mean, f, (real) mean);

      // compute variance per input
      sum = 0;
      TH_TENSOR_APPLY(real, in,
        sum += (*in_data - mean) * (*in_data - mean););

      if (sum == 0 && eps == 0.0) {
        invstd = 0;
      } else {
        invstd = (real) (1 / sqrt(sum/n + eps));
      }
      THTensor_(set1d)(save_std, f, (real) invstd);

      // update running averages
      THTensor_(set1d)(running_mean, f,
        (real) (momentum * mean + (1 - momentum) * THTensor_(get1d)(running_mean, f)));

      accreal unbiased_var = sum / (n - 1);
      THTensor_(set1d)(running_var, f,
        (real) (momentum * unbiased_var + (1 - momentum) * THTensor_(get1d)(running_var, f)));
    } else {
      mean = THTensor_(get1d)(running_mean, f);
      invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
    }

    // compute output
    real w = weight ? THTensor_(get1d)(weight, f) : 1;
    real b = bias ? THTensor_(get1d)(bias, f) : 0;

    TH_TENSOR_APPLY2(real, in, real, out,
      *out_data = (real) (((*in_data - mean) * invstd) * w + b););//结果写到第1个维度上.  w是论文里面gamma, b是beta  https://arxiv.org/pdf/1502.03167.pdf

    THTensor_(free)(out);
    THTensor_(free)(in);
  }
}
//这个函数被调用在torch/legacy/nn/BatchNormalization.py:137  // 对于train 参数的理解!!!!!!因为bn层有时候我们需要不对他进行训练,而又同时需要对其他层进行训练需要传导.所以只锁bn层的参数,不进行训练,这时候均值方差就都是之前训练完的参数!!!!!!!!这时代码很简单.
void THNN_(BatchNormalization_backward)(   // weight 是 gamma, 
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput,
  THTensor *gradWeight, THTensor *gradBias, THTensor *weight,
  THTensor *running_mean, THTensor *running_var,
  THTensor *save_mean, THTensor *save_std,
  bool train, double scale, double eps)//running_mean ,running_var 是上一次吃bn运算的记录下来,我们那他跟档次做一个均匀化.
{
  long nInput = THTensor_(size)(input, 1);//分成多少线程
  long f,n = THTensor_(nElement)(input) / nInput;  //每个线程跑多少个元素.

  #pragma omp parallel for
  for (f = 0; f < nInput; ++f) {
    THTensor *in = THTensor_(newSelect)(input, 1, f);
    THTensor *gradOut = THTensor_(newSelect)(gradOutput, 1, f);
    real w = weight ? THTensor_(get1d)(weight, f) : 1;
    real mean, invstd;

    //获取数值.分train和test2种.
    if (train) {
      mean = THTensor_(get1d)(save_mean, f);
      invstd = THTensor_(get1d)(save_std, f);
    } else {
      mean = THTensor_(get1d)(running_mean, f);
      invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
    }

    // sum over all gradOutput in feature plane
    accreal sum = 0;
    TH_TENSOR_APPLY(real, gradOut, sum += *gradOut_data;);

    // dot product of the Q(X) and gradOuput 定义为dotp
    accreal dotp = 0;
    TH_TENSOR_APPLY2(real, in, real, gradOut,
      dotp += (*in_data - mean) * (*gradOut_data););

    if (gradInput) {
      THTensor *gradIn = THTensor_(newSelect)(gradInput, 1, f);
      if (train) {
        // when in training mode Bn:是Y函数, Q是均值化函数.
        // Q(X) = X - E[x] ; i.e. input centered to zero mean
        // Y = Q(X) / σ    ; i.e. BN output before weight and bias                dL/dY=gradOut_data       
        // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / σ * w      //116行  这个是最核心的函数!!!!!!!!!!!!!!!!!!!跟论文对不上???????   dL/dY=go

        // projection of gradOutput on to output scaled by std
        real k = (real) dotp * invstd * invstd / n;   //k= Q(x) * go/(sigma^2*n)
        TH_TENSOR_APPLY2(real, gradIn, real, in,
          *gradIn_data = (*in_data - mean) * k;);

        accreal gradMean = sum / n;
        TH_TENSOR_APPLY2(real, gradIn, real, gradOut,
          *gradIn_data = (*gradOut_data - gradMean - *gradIn_data) * invstd * w;); //*gradOut_data - gradMean =Q(dL/dY)

      } else {
        // when in evaluation mode
        // Q(X) = X - running_mean  ; i.e. input centered to zero mean
        // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
        // dL/dX = w / running_std
        TH_TENSOR_APPLY2(real, gradIn, real, gradOut,
          *gradIn_data = *gradOut_data * invstd * w;);
      }

      THTensor_(free)(gradIn);
    }
//最后我们要求的就是dL/dW    dL/dBias       就是论文第四页左边留个公式的后2个.
    if (gradWeight) {
      real val = THTensor_(get1d)(gradWeight, f);
      THTensor_(set1d)(gradWeight, f, val + scale * dotp * invstd);  //dotp * invstd =Q(x)/std*gradOutput    gradOutput=dL/dY  Q(x)=xi_hat0
    }   //   set1d:torch/lib/TH/generic/THTensor.c:737

    if (gradBias) {//这个对应最后一个十字.
      real val = THTensor_(get1d)(gradBias, f);
      THTensor_(set1d)(gradBias, f, val + scale * sum);
    }

    THTensor_(free)(gradOut);
    THTensor_(free)(in);
  }
}

#endif
