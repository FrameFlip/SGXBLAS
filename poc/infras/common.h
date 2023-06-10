#ifndef COMMON_H
#define COMMON_H

#ifndef DTYPE
#define DTYPE double
#endif
#ifndef NEWLINE
#define NEWLINE '\n'
#endif
#ifndef DEBUG_FLAG
#define DEBUG_FLAG true
#endif
#ifndef CONSTRUCTOR_FLAG
#define CONSTRUCTOR_FLAG false
#endif
#ifndef TIMING_FLAG
#define TIMING_FLAG false
#endif
#ifndef DEBUG_NO1_FLAG
#define DEBUG_NO1_FLAG false
#endif

#define PRINT_DEBUG(msg) if (DEBUG_FLAG) printf(msg)
#define PRINT_CONSTRUCTOR(msg) if (DEBUG_FLAG && CONSTRUCTOR_FLAG) printf(msg)
#define PRINT_TIMING(msg, time) if (DEBUG_FLAG && TIMING_FLAG) printf("[timer] %s takes \t%.4f secs\n", msg, time)

#include "tensor.h"
#include "utils.h"

using TypedTensor = Tensor<DTYPE>;

#endif