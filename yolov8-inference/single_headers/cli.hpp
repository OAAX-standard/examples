
// Utility function to parse command line arguments
// This function uses the CLI11 library to handle command line options
int parse_command_line(int argc, char **argv, string &library_path,
                       string &model_path, string &input_path,
                       string &config_path, string &log_file, int &log_level) {
  CLI::App app{"OAAX inference engine command line tool"};

  app.add_option("-l,--library", library_path,
                 "Path to the OAAX runtime library")
      ->required();
  app.add_option("-m,--model", model_path, "Path to the model file")
      ->required();
  app.add_option("-i,--input", input_path, "Path to the input image file")
      ->required();
  app.add_option("--log-file", log_file, "Path to the log file")
      ->default_val("app.log");
  app.add_option(
         "--log-level", log_level,
         "Set file logging level (default: 2 for info)."
         "0: trace, 1: debug, 2: info, 3: warn, 4: err, 5: critical, 6: off")
      ->default_val(2);
  app.add_option("-c,--config", config_path,
                 "Path to the configuration JSON file")
      ->required();

  // Optional help flag
  app.set_help_flag("-h,--help", "Display this help message");

  CLI11_PARSE(app, argc, argv);

  return 0;  // Return 0 on successful parsing
}