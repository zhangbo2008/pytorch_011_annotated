#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else
// src拷贝到tensor向量里面. 具体就是里面data拷贝过来即可.这个是float上
void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = (real)(*src_data);)
}
//下面是在其他类型上的推广.
#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

IMPLEMENT_THTensor_COPY(Byte, unsigned char)
IMPLEMENT_THTensor_COPY(Char, char)
IMPLEMENT_THTensor_COPY(Short, short)
IMPLEMENT_THTensor_COPY(Int, int)
IMPLEMENT_THTensor_COPY(Long, long)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)

#endif
