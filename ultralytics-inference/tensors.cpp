#include "single_headers/tensors.hpp"

tensors_struct *create_tensors(cv::Mat &image, string &input_name,
                                 bool nchw = true,
                                 const string &input_dtype = "float32") {
  spdlog::info("Creating tensors for input image: {}", input_name);
  if (image.empty()) {
    spdlog::error("Input image is empty.");
    return nullptr;
  }

  tensors_struct *tensors =
      (tensors_struct *)malloc(sizeof(tensors_struct));
  tensors->num_tensors = 1;
  tensors->names = (char **)malloc(sizeof(char *));
  tensors->names[0] = strdup((char *)input_name.c_str());
  tensors->ranks = (size_t *)malloc(sizeof(size_t));
  tensors->ranks[0] = 4;
  tensors->shapes = (size_t **)malloc(sizeof(size_t *));
  tensors->shapes[0] = (size_t *)malloc(4 * sizeof(size_t));
  tensors->data = (void **)malloc(sizeof(void *));

  int channels = image.channels();
  int height = image.rows;
  int width = image.cols;

  if (input_dtype == "float32") {
    tensors->data_types = (tensor_data_type *)malloc(sizeof(tensor_data_type));
    tensors->data_types[0] = DATA_TYPE_FLOAT;
    cv::Mat float_image;
    image.convertTo(float_image, CV_32F);
    tensors->data[0] = malloc(channels * height * width * sizeof(float));
    if (nchw) { // NCHW format
      tensors->shapes[0][0] = 1;
      tensors->shapes[0][1] = channels;
      tensors->shapes[0][2] = height;
      tensors->shapes[0][3] = width;
      // Convert HWC to CHW
      vector<cv::Mat> channels_mats;
      cv::split(float_image, channels_mats);
      float *data_ptr = (float *)tensors->data[0];
      for (int c = 0; c < channels; ++c) {
        memcpy(data_ptr + c * height * width, channels_mats[c].data,
               height * width * sizeof(float));
      }
    } else { // NHWC format
        tensors->shapes[0][0] = 1;
        tensors->shapes[0][1] = height;
        tensors->shapes[0][2] = width;
        tensors->shapes[0][3] = channels;
        memcpy(tensors->data[0], float_image.data,
                channels * height * width * sizeof(float));
    }
  } else if (input_dtype == "uint8") {
    tensors->data_types = (tensor_data_type *)malloc(sizeof(tensor_data_type));
    tensors->data_types[0] = DATA_TYPE_UINT8;
    tensors->data[0] = malloc(channels * height * width * sizeof(uint8_t));
    if (nchw) { // NCHW format
        tensors->shapes[0][0] = 1;
        tensors->shapes[0][1] = channels;
        tensors->shapes[0][2] = height;
        tensors->shapes[0][3] = width;
        // Convert HWC to CHW
        vector<cv::Mat> channels_mats;
        cv::split(image, channels_mats);
        uint8_t *data_ptr = (uint8_t *)tensors->data[0];
        for (int c = 0; c < channels; ++c) {
            memcpy(data_ptr + c * height * width, channels_mats[c].data,
                    height * width * sizeof(uint8_t));
        }
    } else { // NHWC format
        tensors->shapes[0][0] = 1;
        tensors->shapes[0][1] = height;
        tensors->shapes[0][2] = width;
        tensors->shapes[0][3] = channels;
        memcpy(tensors->data[0], image.data,
                channels * height * width * sizeof(uint8_t));
    }
  } else {
    spdlog::error("Unsupported input data type.");
    return nullptr;
  }
  print_tensors_metadata(tensors);

  return tensors;
}

void print_tensors_metadata(tensors_struct *tensors){
    printf("Number of tensors: %ld\n", tensors->num_tensors);
    for (size_t i = 0; i < tensors->num_tensors; ++i) {
        printf("Tensor id=%ld:\n", i);
        printf("  Name: '%s'\n", tensors->names[i]);
        printf("  Data type: %d\n", tensors->data_types[i]);
        printf("  Rank: %ld\n", tensors->ranks[i]);
        printf("  Shape: ");
        for (size_t j = 0; j < tensors->ranks[i]; ++j) {
            printf("%ld ", tensors->shapes[i][j]);
        }
        printf("\n");
        printf("  Data Pointer: %p\n", tensors->data[i]);
        printf("\n");
    }
}
