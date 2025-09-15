
enum ResizeMethod { LETTERBOX, CROP_THEN_RESIZE, SQUASH };

cv::Mat preprocess_image(const std::string &image_path, int target_width,
                         int target_height, ResizeMethod resize_method,
                         const cv::Scalar &mean, const cv::Scalar &stddev) {
    spdlog::info("Preprocessing image: {}", image_path);
    try {
        // Load the image from the given path
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        // Convert the image to RGB
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // Resize the image based on the chosen method
        if (resize_method == LETTERBOX) {
            int original_width = image.cols;
            int original_height = image.rows;
            float scale =
                std::min(static_cast<float>(target_width) / original_width,
                         static_cast<float>(target_height) / original_height);
            int new_width = static_cast<int>(original_width * scale);
            int new_height = static_cast<int>(original_height * scale);
            cv::Mat resized_image;
            cv::resize(image, resized_image, cv::Size(new_width, new_height));
            image = cv::Mat::zeros(target_height, target_width, image.type());
            resized_image.copyTo(image(cv::Rect(
                (target_width - new_width) / 2,
                (target_height - new_height) / 2, new_width, new_height)));
            // Free temporary images
            resized_image.release();
        } else if (resize_method == CROP_THEN_RESIZE) {
            int crop_size = std::min(image.cols, image.rows);
            cv::Rect crop_region((image.cols - crop_size) / 2,
                                 (image.rows - crop_size) / 2, crop_size,
                                 crop_size);
            cv::Mat cropped_image = image(crop_region);
            cv::resize(cropped_image, image,
                       cv::Size(target_width, target_height));
            // Free temporary images
            cropped_image.release();
        } else if (resize_method == SQUASH) {
            cv::resize(image, image, cv::Size(target_width, target_height));
        }
        // print mean and stddev values
        spdlog::info("Mean: {}, {}, {}", mean[0], mean[1], mean[2]);
        spdlog::info("Stddev: {}, {}, {}", stddev[0], stddev[1], stddev[2]);
        // Convert the image to float and normalize
        image.convertTo(image, CV_32F);
        // Apply mean and standard deviation normalization
        image = image - mean;    // Subtract mean
        image = image / stddev;  // Divide by stddev
        // print min and max value of image
        double min_val, max_val;
        cv::minMaxLoc(image, &min_val, &max_val);
        spdlog::info("Image min value: {}, max value: {}", min_val, max_val);

        return image;
    } catch (const std::exception &e) {
        spdlog::error("Error preprocessing image: {}", e.what());
        exit(EXIT_FAILURE);
    }
}
