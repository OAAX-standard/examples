#ifndef INTERFACE_H
#define INTERFACE_H

#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

typedef enum tensor_data_type {
    DATA_TYPE_FLOAT = 1,
    DATA_TYPE_UINT8 = 2,
    DATA_TYPE_INT8 = 3,
    DATA_TYPE_UINT16 = 4,
    DATA_TYPE_INT16 = 5,
    DATA_TYPE_INT32 = 6,
    DATA_TYPE_INT64 = 7,
    DATA_TYPE_STRING = 8,
    DATA_TYPE_BOOL = 9,
    DATA_TYPE_DOUBLE = 11,
    DATA_TYPE_UINT32 = 12,
    DATA_TYPE_UINT64 = 13
} tensor_data_type;

typedef struct tensors_struct {
    size_t num_tensors;                 // Number of tensors
    char** names;                 // Names of the tensors
    tensor_data_type* data_types;       // Data types of the tensors
    size_t* ranks;                      // Ranks of the tensors
    size_t** shapes;                    // Shapes of the tensors
    void** data;                        // Data of the tensors
} tensors_struct;

#endif // INTERFACE_H