using json = nlohmann::json;

json load_config(string &config_path) {
    ifstream config_file(config_path);
    if (!config_file.is_open()) {
        spdlog::error("Failed to open config file: {}", config_path);
        exit(EXIT_FAILURE);
    }

    json config;
    config_file >> config;

    return config;  // Return the loaded JSON configuration
}