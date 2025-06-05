// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "metrics.h"  // NOLINT[build/include]

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cJSON.h"  // NOLINT[build/include]

void save_metrics_json(const char *model_path, const char *image_path,
                       const char *library_path, int num_args,
                       const char **arg_keys, const void **arg_values,
                       const char *runtime_name, const char *runtime_version,
                       int number_of_inferences, const SystemInfo *system_info,
                       const RealTimeSystemInfo *real_time_info, float fps_rate,
                       const char *output_file_path) {
  cJSON *root = cJSON_CreateObject();
  if (root == NULL) {
    fprintf(stderr, "Failed to create JSON object\n");
    return;
  }
  // Add current date and time
  time_t now = time(NULL);
  char time_buffer[2506];
  strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%dT%H:%M:%SZ",
           gmtime(&now));
  cJSON_AddStringToObject(root, "datetime", time_buffer);

  // Add model, image, and library paths
  cJSON_AddStringToObject(root, "model_path", model_path);
  cJSON_AddStringToObject(root, "image_path", image_path);
  cJSON_AddStringToObject(root, "library_path", library_path);
  cJSON_AddNumberToObject(root, "num_args", num_args);
  for (int i = 0; i < num_args; i++) {
    // Assuming arg_values are ints for simplicity
    cJSON_AddNumberToObject(root, arg_keys[i], *(int *)arg_values[i]);
  }

  cJSON_AddStringToObject(root, "runtime_name", runtime_name);
  cJSON_AddStringToObject(root, "runtime_version", runtime_version);
  cJSON_AddNumberToObject(root, "number_of_inferences", number_of_inferences);
  cJSON_AddNumberToObject(root, "fps_rate", fps_rate);

  // write system information
  cJSON_AddStringToObject(root, "cpu_name", system_info->cpu_name);
  cJSON_AddNumberToObject(root, "cpu_cores", system_info->cpu_cores);
  cJSON_AddNumberToObject(root, "logical_processors",
                          system_info->logical_processors);
  cJSON_AddNumberToObject(root, "cpu_clock_mhz", system_info->cpu_clock_mhz);
  cJSON_AddBoolToObject(root, "hyperthreading_supported",
                        system_info->hyperthreading_supported);
  cJSON_AddStringToObject(root, "instruction_sets",
                          system_info->instruction_sets);
  cJSON_AddNumberToObject(root, "total_ram_mb", system_info->total_ram_mb);
  cJSON_AddStringToObject(root, "os_name", system_info->os_name);
  cJSON_AddStringToObject(root, "architecture", system_info->architecture);
  cJSON_AddBoolToObject(root, "is_virtual_machine",
                        system_info->is_virtual_machine);

  // write real-time system information
  cJSON_AddNumberToObject(root, "cpu_usage_percent",
                          real_time_info->cpu_usage_percent);
  cJSON_AddNumberToObject(root, "used_ram_mb", real_time_info->used_ram_mb);
  cJSON_AddNumberToObject(root, "ram_usage_percent",
                          real_time_info->ram_usage_percent);

  // Save the JSON object to a file
  FILE *file = fopen(output_file_path, "a");
  if (file != NULL) {
    char *json_string = cJSON_PrintUnformatted(root);
    if (json_string != NULL) {
      fprintf(file, "%s\n", json_string);
      free(json_string);
    }
    fclose(file);
  }

  cJSON_Delete(root);
}