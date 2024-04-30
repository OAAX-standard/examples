#include "io_utils.h"

int parse_input_data(const char *input_packed_message,
                                size_t input_message_length,
                                size_t *returned_num_tensors,
                                void ***returned_inputs,
                                size_t ***returned_input_shapes,
                                size_t **returned_input_ranks) {
    // Parse message schema
    mpack_reader_t reader;
    mpack_reader_init_data(&reader, input_packed_message, input_message_length);

    // **** Read data according to schema *****

    uint32_t num_tensors = mpack_expect_uint(&reader);
    if (mpack_reader_error(&reader) != mpack_ok) {
        printf("Warning: Error reading num_tensors\n");
        return EXIT_FAILURE;
    }

    returned_num_tensors[0] = num_tensors;

    // Read input tensors in place ( do not free )
    void **inputs = (void **) malloc(num_tensors * sizeof(void *));
    // Read each tensor separately
    for (size_t tensor_index = 0; tensor_index < num_tensors; tensor_index++) {
        uint32_t input_bin_lenght = mpack_expect_bin_max(&reader, 1024 * 1024 * 100);
        // Read bytes in place to avoid copying
        uint8_t *local_input_data = (uint8_t *) mpack_read_bytes_inplace(&reader, input_bin_lenght);
        if (mpack_reader_error(&reader) != mpack_ok) {
            printf("Error: RUNTIME - Error reading input data\n");
            return EXIT_FAILURE;
        }
        mpack_done_bin(&reader);
        inputs[tensor_index] = (uint8_t *) local_input_data;
    }

    // Read output type (assume ownership of pointer)
    char *output_type = mpack_expect_cstr_alloc(&reader, 100); // Max 100 chars
    if (mpack_reader_error(&reader) != mpack_ok) {
        return EXIT_FAILURE;
    }
    // This value is unused by ONNXRUNTIME runtimes, but is used by SCAILABLE runtimes
    free(output_type);

    // Read input shapes ( No need actually, for ONNXRUNTIME these values are alaways included)
    bool shapes_included = mpack_expect_bool(&reader);
    if (!shapes_included) {
        printf("Error: RUNTIME - shapes are not included\n");
        return EXIT_FAILURE;
    }

    size_t **input_shapes = (size_t **) malloc(num_tensors * sizeof(size_t *));
    size_t *input_ranks = (size_t *) malloc(num_tensors * sizeof(size_t));
    for (size_t index = 0; index < num_tensors; index++) {
        uint32_t tensor_rank = mpack_expect_u32(&reader);
        input_ranks[index] = tensor_rank;
    }
    for (size_t index = 0; index < num_tensors; index++) {
        input_shapes[index] = malloc(input_ranks[index] * sizeof(int64_t));
        for (int32_t rank_index = 0; rank_index < input_ranks[index]; rank_index++) {
            input_shapes[index][rank_index] = mpack_expect_u64(&reader);
        }
    }

    returned_inputs[0] = inputs;
    returned_input_shapes[0] = input_shapes;
    returned_input_ranks[0] = input_ranks;

    // Return
    return 0;
}

int build_output_mpack(char **output_names,
                                  size_t number_outputs,
                                  void **outputs,
                                  size_t **output_shapes,
                                  size_t *output_ranks,
                                  tensor_data_type *output_data_types,
                                  char **output_buffer,
                                  size_t *output_buffer_size) {
    // Initialize writer
    mpack_writer_t writer;
    char *mpack_buffer;
    size_t buffer_size;
    mpack_writer_init_growable(&writer, &mpack_buffer, &buffer_size);

    // Start building root map
    mpack_start_map(&writer, 4);

    // Write outputs ({OutputName} bin)
    // Map key
    mpack_write_cstr(&writer, "Outputs");
    // Map value
    mpack_start_map(&writer, number_outputs);
    // Compute size_t *output_sizes
    size_t *output_sizes = (size_t *) malloc(number_outputs * sizeof(size_t));
    for (size_t o = 0; o < number_outputs; o++) {
        output_sizes[o] = 1;
        for (size_t r = 0; r < output_ranks[o]; r++)
            output_sizes[o] *= output_shapes[o][r];
    }

    for (int index = 0; index < number_outputs; index++) {
        // Determine output size
        size_t tensor_byte_size = (size_t) output_sizes[index];
        switch (output_data_types[index]) {
            case 1: // onnx::TensorProto_DataType_FLOAT:
            {
                tensor_byte_size *= sizeof(float);
                break;
            }
            case 2: // onnx::TensorProto_DataType_UINT8:
            {
                tensor_byte_size *= sizeof(uint8_t);
                break;
            }
            case 3: //onnx::TensorProto_DataType_INT8:
            {
                tensor_byte_size *= sizeof(int8_t);
                break;
            }
            case 6: //onnx::TensorProto_DataType_INT32:
            {
                tensor_byte_size *= sizeof(int32_t);
                break;
            }
            case 7: //onnx::TensorProto_DataType_INT64:
            {
                tensor_byte_size *= sizeof(int64_t);
                break;
            }
            case 8: // onnx::TensorProto_DataType_STRING:
            {
                tensor_byte_size *= sizeof(char);
                break;
            }
            case 9: //onnx::TensorProto_DataType_BOOL:
            {
                tensor_byte_size *= sizeof(bool);
                break;
            }
            case 11: // TensorProto_DataType_DOUBLE:
            {
                tensor_byte_size *= sizeof(double);
                break;
            }
            default:continue;
        }
        mpack_write_cstr(&writer, output_names[index]);
        mpack_write_bin(&writer, (const char *) outputs[index], tensor_byte_size);
    }
    free(output_sizes);
    mpack_finish_array(&writer); // Finish "Outputs" array

    // Write output ranks ([num tensors]i32)
    // Map key
    mpack_write_cstr(&writer, "OutputRanks");
    // Map value
    mpack_start_array(&writer, number_outputs);
    for (int index = 0; index < number_outputs; index++) {
        mpack_write_i32(&writer, output_ranks[index]);
    }
    mpack_finish_array(&writer); // Finish "OutputRanks" array

    // Write output shapes ([num tensors][rank]i64)
    // Map key
    mpack_write_cstr(&writer, "OutputShapes");
    // Map value
    mpack_start_array(&writer, number_outputs);
    for (int output_index = 0; output_index < number_outputs; output_index++) {
        mpack_start_array(&writer, output_ranks[output_index]);
        for (int rank_index = 0; rank_index < output_ranks[output_index]; rank_index++) {
            mpack_write_i64(&writer, output_shapes[output_index][rank_index]);
        }
        mpack_finish_array(&writer); // Finish "OutputShapes" inner array
    }
    mpack_finish_array(&writer); // Finish "OutputShapes" outer array

    // Write output data types ([num tensors]i32)
    // Map key
    mpack_write_cstr(&writer, "OutputDataTypes");
    // Map value
    mpack_start_array(&writer, number_outputs);
    for (int index = 0; index < number_outputs; index++) {
        mpack_write_i32(&writer, output_data_types[index]);
    }
    mpack_finish_array(&writer); // Finish "OutputDataTypes" array

    // Finish building root map
    mpack_finish_map(&writer);

    // Finish writing
    if (mpack_writer_destroy(&writer) != mpack_ok) {
        fprintf(stderr, "An error occurred encoding the data!\n");
        // Free buffer since it was not succesful
        free(writer.buffer);
        // Reset buffer so that it's not used again
        *output_buffer = NULL;
        return 1;
    }

    *output_buffer_size = buffer_size;
    *output_buffer = mpack_buffer;

    return 0;
}
