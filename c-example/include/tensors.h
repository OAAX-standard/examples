// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#ifndef C_EXAMPLE_INCLUDE_TENSORS_H_
#define C_EXAMPLE_INCLUDE_TENSORS_H_

#include <stddef.h>

#include "interface.h"  // NOLINT[build/include]

/**
 * @brief Allocate a Tensors struct with `num_tensors` zeroed descriptors.
 * @param [in] id Request id; the runtime echoes it on the matching output
 * @return The new struct, or NULL if out of memory
 */
Tensors *alloc_tensors(int id, int num_tensors);

/**
 * @brief Fill a descriptor with copies of the given name, shape and data.
 * @note The data is copied because the runtime frees it after inference.
 * @return 0 on success, -1 if out of memory
 */
int fill_tensor(TensorDescriptor *tensor, const char *name,
                TensorElementType data_type, int rank, const int *shape,
                const void *data, size_t data_size);

/**
 * @brief Free a Tensors struct and everything it owns (names, shapes, data).
 * @note Use it on outputs you retrieved and on inputs that were NOT accepted
 * by `runtime_enqueue_input`. Accepted inputs belong to the runtime.
 */
void free_tensors(Tensors *tensors);

#endif  // C_EXAMPLE_INCLUDE_TENSORS_H_
