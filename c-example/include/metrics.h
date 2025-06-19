// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#ifndef C_EXAMPLE_INCLUDE_METRICS_H_
#define C_EXAMPLE_INCLUDE_METRICS_H_
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <psapi.h>
#include <windows.h>

ULONGLONG get_process_cpu_time() {
  FILETIME creation, exit, kernel, user;
  if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
    ULONGLONG ktime =
        ((ULONGLONG)kernel.dwHighDateTime << 32) | kernel.dwLowDateTime;
    ULONGLONG utime =
        ((ULONGLONG)user.dwHighDateTime << 32) | user.dwLowDateTime;
    return ktime + utime;  // in 100-nanosecond units
  }
  return 0;
}

void get_usage(float *cpu_percent, float *ram_kb) {
  // RAM usage
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    *ram_kb = (float)(pmc.WorkingSetSize) / 1024;
  } else {
    *ram_kb = 0.0f;
  }

  // CPU usage
  ULONGLONG start_time = get_process_cpu_time();
  Sleep(1000);  // wait 1 second
  ULONGLONG end_time = get_process_cpu_time();

  // Calculate % CPU usage over interval
  ULONGLONG delta = end_time - start_time;
  *cpu_percent = (float)(delta / 10000.0);  // 100ns units -> ms
}
#else
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

double get_process_cpu_time() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 +
         usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
}

void get_usage(float *cpu_percent, float *ram_kb) {
  // RAM usage
  FILE *f = fopen("/proc/self/status", "r");
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      *ram_kb = atof(line + 6);  // already in KB
      break;
    }
  }
  fclose(f);

  // CPU usage
  double start = get_process_cpu_time();
  sleep(1);  // wait 1 second
  double end = get_process_cpu_time();
  *cpu_percent = (float)((end - start) * 100.0);  // 1 sec interval
}
#endif

#endif  // C_EXAMPLE_INCLUDE_METRICS_H_
