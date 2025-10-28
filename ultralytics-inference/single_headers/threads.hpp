#pragma once

#include "runtime.hpp"
#include "tensors.hpp"
#include <iostream>
#include <spdlog/spdlog.h>
#include <thread>
#include <vector>

void send_input_tensors_routine(Runtime *runtime,
                                tensors_struct *original_tensors);
void receive_output_tensors_routine(Runtime *runtime,
                                    tensors_struct **output_tensors);