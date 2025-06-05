// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#ifndef C_EXAMPLE_INCLUDE_METRICS_H_
#define C_EXAMPLE_INCLUDE_METRICS_H_

#include "sysinfo.h"  // NOLINT[build/include]

void save_metrics_json(const char *model_path, const char *image_path,
                       const char *library_path, int num_args,
                       const char **arg_keys, const void **arg_values,
                       const char *runtime_name, const char *runtime_version,
                       int number_of_inferences, const SystemInfo *system_info,
                       const RealTimeSystemInfo *real_time_info, float fps_rate,
                       const char *output_file_path);


#endif  // C_EXAMPLE_INCLUDE_METRICS_H_