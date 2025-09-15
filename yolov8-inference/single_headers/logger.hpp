
spdlog::logger initialize_logger(const string &log_file,
                                 int file_level = spdlog::level::info,
                                 int console_level = spdlog::level::info,
                                 const string prefix = "OAAX") {
    try {
        // Create a console logger
        auto console_sink = make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(
            static_cast<spdlog::level::level_enum>(console_level));

        // Create a rotating file logger
        auto file_sink = make_shared<spdlog::sinks::rotating_file_sink_st>(
            log_file, 1024 * 1024 * 5, 3);  // 5MB max size, 3 rotated files
        file_sink->set_level(
            static_cast<spdlog::level::level_enum>(file_level));

        // Configure the thread pool for async logging
        static auto thread_pool =
            make_shared<spdlog::details::thread_pool>(8192, 1);

        // Create the async logger with both sinks using the thread pool
        auto logger = make_shared<spdlog::async_logger>(
            prefix, spdlog::sinks_init_list{console_sink, file_sink},
            thread_pool, spdlog::async_overflow_policy::overrun_oldest);

        // Set the logging pattern
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [" + prefix +
                            "] [%^%l%$] %v");

        // Set the logger as the default logger
        spdlog::set_default_logger(logger);

        return *logger;  // Return the logger instance
    } catch (const spdlog::spdlog_ex &ex) {
        cerr << "Logger initialization failed: " << ex.what() << "\n";
        exit(EXIT_FAILURE);
    }
}

void destroy_logger() {
    // Flush and close the logger
    spdlog::shutdown();
    // Optionally, you can reset the default logger to nullptr
    spdlog::set_default_logger(nullptr);
}