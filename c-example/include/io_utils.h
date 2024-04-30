#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "mpack.h"
#include "runtime_utils.h"

/**
 * Extract relevant values from input.
 *
 * @param [in] input_packed_message Input datas (NOT freed in this function)
 * @param [in] input_message_length Length of input datas
 * @param [out] returned_num_tensors Number of input tensors
 * @param [out] returned_inputs Array of input tensors. Should be cleaned by the caller
 * @param [out] returned_input_shapes Array of input shapes. Should be cleaned by the caller
 * @param [out] returned_input_ranks Array of input ranks. Should be cleaned by the caller
 * @return exit code: non-zero on error
 */
int parse_input_data(const char *input_packed_message,
                                size_t input_message_length,
                                size_t *num_tensors,
                                void ***returned_inputs,
                                size_t ***returned_input_shapes,
                                size_t **returned_input_ranks);
/**
 * Build output MessagePack from model's outputs.
 *
 * Schema:
 *
 * 1. "Outputs" ({OutputName:bin})
 * 2. "OutputRanks" ([num tensors]i32)
 * 3. "OutputShapes" ([num tensors][rank]i64)
 * 4. "OutputDataTypes" ([num tensors]i32)
 *
 * @param [in] output_names Array of output names
 * @param [in] number_outputs Number of outputs
 * @param [in] outputs Array of output tensors
 * @param [in] output_shapes Array of output shapes
 * @param [in] output_ranks Array of output ranks
 * @param [in] output_data_types Array of outputs data type
 * @param [out] output_json Output JSON string pointer.
 * @param [out] output_json_size length of Output JSON string
 * @return exit code: non-zero on error
 */
int build_output_mpack(char **output_names,
                                  size_t number_outputs,
                                  void **outputs,
                                  size_t **output_shapes,
                                  size_t *output_ranks,
                                  tensor_data_type *output_data_types,
                                  char **output_buffer,
                                  size_t *output_buffer_size);