// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "tensors.h"  // NOLINT[build/include]

#include <stdlib.h>
#include <string.h>

Tensors *alloc_tensors(int id, int num_tensors) {
  Tensors *tensors = (Tensors *)calloc(1, sizeof(Tensors));
  if (tensors == NULL) return NULL;
  tensors->tensors =
      (TensorDescriptor *)calloc((size_t)num_tensors, sizeof(TensorDescriptor));
  if (tensors->tensors == NULL) {
    free(tensors);
    return NULL;
  }
  tensors->id = id;
  tensors->num_tensors = num_tensors;
  return tensors;
}

int fill_tensor(TensorDescriptor *tensor, const char *name,
                TensorElementType data_type, int rank, const int *shape,
                const void *data, size_t data_size) {
  size_t name_size = strlen(name) + 1;
  size_t shape_size = (size_t)rank * sizeof(int);
  tensor->name = (char *)malloc(name_size);
  tensor->shape = (int *)malloc(shape_size);
  tensor->data = malloc(data_size);
  if (tensor->name == NULL || tensor->shape == NULL || tensor->data == NULL) {
    return -1;  // Partial allocations are released by free_tensors
  }
  memcpy(tensor->name, name, name_size);
  memcpy(tensor->shape, shape, shape_size);
  memcpy(tensor->data, data, data_size);
  tensor->data_type = data_type;
  tensor->rank = rank;
  tensor->data_size = data_size;
  return 0;
}

void free_tensors(Tensors *tensors) {
  if (tensors == NULL) return;
  if (tensors->tensors != NULL) {
    for (int i = 0; i < tensors->num_tensors; i++) {
      free(tensors->tensors[i].name);
      free(tensors->tensors[i].shape);
      free(tensors->tensors[i].data);
    }
    free(tensors->tensors);
  }
  free(tensors);
}
