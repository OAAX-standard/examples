tensors_struct *create_tensors(cv::Mat &image, string &input_name,
                               bool nchw = true,
                               const string &input_dtype = "float32") {
  spdlog::info("Creating tensors for input image: {}", input_name);
  if (image.empty()) {
    spdlog::error("Input image is empty.");
    exit(EXIT_FAILURE);
  }
  // Cast image to float
  image.convertTo(image, CV_32F);

  // Create a new tensors_struct for the input tensor
  tensors_struct *tensors =
      (tensors_struct *)malloc(1 * sizeof(tensors_struct));
  tensors->num_tensors = 1;  // Assuming a single tensor for the input
  // Allocate memory for the tensor fields
  tensors->data_types = (tensor_data_type *)malloc(tensors->num_tensors *
                                                   sizeof(tensor_data_type));
  tensors->names = (char **)malloc(tensors->num_tensors * sizeof(char *));
  tensors->ranks = (size_t *)malloc(tensors->num_tensors * sizeof(size_t));
  tensors->shapes = (size_t **)malloc(tensors->num_tensors * sizeof(size_t *));
  tensors->data = (void **)malloc(tensors->num_tensors * sizeof(void *));
  // Set the name of the tensor
  tensors->names[0] = strdup(input_name.c_str());
  // Set the data type for the tensor
  if (input_dtype == "uint8") {
    tensors->data_types[0] = DATA_TYPE_UINT8;
  } else if (input_dtype == "int8") {
    tensors->data_types[0] = DATA_TYPE_INT8;
  } else if (input_dtype == "float32") {
    tensors->data_types[0] = DATA_TYPE_FLOAT;
  } else {
    spdlog::error("Unsupported input data type.");
    exit(EXIT_FAILURE);
  }
  // Set the rank of the tensor (assuming 4D for NCHW format)
  tensors->ranks[0] = 4;  // NCHW format
  // Allocate memory for the shape of the tensor
  tensors->shapes[0] = (size_t *)malloc(4 * sizeof(size_t));
  // Set the shape of the tensor
  if (nchw) {
    // For UINT8 and INT8, we assume the image is in HWC format
    // Convert to NCHW format
    tensors->shapes[0][0] = 1;                 // Batch size N
    tensors->shapes[0][1] = image.channels();  // Number of channels
    tensors->shapes[0][2] = image.rows;        // Height H
    tensors->shapes[0][3] = image.cols;        // Width W
  } else {
    // For FLOAT32, we assume the image is already in NCHW format
    tensors->shapes[0][0] = 1;                 // Batch size N
    tensors->shapes[0][1] = image.rows;        // Height H
    tensors->shapes[0][2] = image.cols;        // Width W
    tensors->shapes[0][3] = image.channels();  // Number of channels C
  }
  // Allocate memory for the tensor data
  size_t tensor_size = image.total() * image.channels();
  if (input_dtype == "uint8") {
    tensors->data[0] = (uint8_t *)malloc(tensor_size * sizeof(uint8_t));
    // Copy the image data to the tensor data
    uint8_t *tensor_data = (uint8_t *)tensors->data[0];
    for (int h = 0; h < image.rows; ++h) {
      for (int w = 0; w < image.cols; ++w) {
        for (int c = 0; c < image.channels(); ++c) {
          tensor_data[c * image.rows * image.cols + h * image.cols + w] =
              static_cast<uint8_t>(image.at<cv::Vec3f>(h, w)[c]);
        }
      }
    }
    if (nchw) {
      // Convert to NCHW format
      for (int h = 0; h < image.rows; ++h) {
        for (int w = 0; w < image.cols; ++w) {
          for (int c = 0; c < image.channels(); ++c) {
            tensor_data[c * image.rows * image.cols + h * image.cols + w] =
                tensor_data[c * image.rows * image.cols + h * image.cols + w];
          }
        }
      }
    }
  } else if (input_dtype == "int8") {
    tensors->data[0] = (int8_t *)malloc(tensor_size * sizeof(int8_t));
    // Copy the image data to the tensor data
    int8_t *tensor_data = (int8_t *)tensors->data[0];
    for (int h = 0; h < image.rows; ++h) {
      for (int w = 0; w < image.cols; ++w) {
        for (int c = 0; c < image.channels(); ++c) {
          tensor_data[c * image.rows * image.cols + h * image.cols + w] =
              static_cast<int8_t>(image.at<cv::Vec3f>(h, w)[c]);
        }
      }
    }
    if (nchw) {
      // Convert to NCHW format
      for (int h = 0; h < image.rows; ++h) {
        for (int w = 0; w < image.cols; ++w) {
          for (int c = 0; c < image.channels(); ++c) {
            tensor_data[c * image.rows * image.cols + h * image.cols + w] =
                tensor_data[c * image.rows * image.cols + h * image.cols + w];
          }
        }
      }
    }
  } else if (input_dtype == "float32") {
    tensors->data[0] = (float *)malloc(tensor_size * sizeof(float));
    // Copy the image data to the tensor data
    float *tensor_data = (float *)tensors->data[0];
    for (int h = 0; h < image.rows; ++h) {
      for (int w = 0; w < image.cols; ++w) {
        for (int c = 0; c < image.channels(); ++c) {
          tensor_data[c * image.rows * image.cols + h * image.cols + w] =
              image.at<cv::Vec3f>(h, w)[c];
        }
      }
    }
    if (nchw) {
      // Convert to NCHW format
      for (int h = 0; h < image.rows; ++h) {
        for (int w = 0; w < image.cols; ++w) {
          for (int c = 0; c < image.channels(); ++c) {
            tensor_data[c * image.rows * image.cols + h * image.cols + w] =
                tensor_data[c * image.rows * image.cols + h * image.cols + w];
          }
        }
      }
    }
  }
  return tensors;  // Return the created tensor
}
