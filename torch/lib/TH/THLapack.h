#ifndef TH_LAPACK_INC
#define TH_LAPACK_INC

#include "THGeneral.h"
//把函数都定义过来.具体实现就是引用lapack库包.
#define THLapack_(NAME) TH_CONCAT_4(TH,Real,Lapack_,NAME)

#define THLapackCheck(fmt, func, info , ...)						\
if (info < 0) {														\
  THError("Lapack Error in %s : Illegal Argument %d", func, -info); \
} else if(info > 0) {												\
  THError(fmt, func, info, ##__VA_ARGS__);							\
}																	\

#define THLapackCheckWithCleanup(fmt, cleanup, func, info , ...)    \
if (info < 0) {                                                     \
  cleanup                                                           \
  THError("Lapack Error in %s : Illegal Argument %d", func, -info); \
} else if(info > 0) {                                               \
  cleanup                                                           \
  THError(fmt, func, info, ##__VA_ARGS__);                          \
}

#include "generic/THLapack.h"
#include "THGenerateAllTypes.h"

#endif
