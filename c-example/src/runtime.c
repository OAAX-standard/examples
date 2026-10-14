// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include "runtime.h"  // NOLINT[build/include]

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#define OPEN_LIBRARY(path) ((void *)LoadLibraryA(path))
#define FIND_SYMBOL(handle, name) \
  ((void *)GetProcAddress((HMODULE)(handle), name))
#define CLOSE_LIBRARY(handle) FreeLibrary((HMODULE)(handle))
#else
#include <dlfcn.h>
#define OPEN_LIBRARY(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define FIND_SYMBOL(handle, name) dlsym(handle, name)
#define CLOSE_LIBRARY(handle) dlclose(handle)
#endif

// Resolves `symbol` into the function pointer `member`. dlsym returns a
// `void *`, so the assignment goes through the member's own type.
#define RESOLVE(member, symbol)                                          \
  do {                                                                   \
    *(void **)(&runtime->member) = FIND_SYMBOL(runtime->handle, symbol); \
    if (runtime->member == NULL) {                                       \
      fprintf(stderr, "Error: missing symbol '%s'\n", symbol);           \
      unload_runtime_library(runtime);                                   \
      return -1;                                                         \
    }                                                                    \
  } while (0)

int load_runtime_library(const char *library_path, Runtime *runtime) {
  *runtime = (Runtime){0};
  runtime->handle = OPEN_LIBRARY(library_path);
  if (runtime->handle == NULL) {
#ifdef _WIN32
    fprintf(stderr, "Error: failed to load '%s' (code %lu)\n", library_path,
            GetLastError());
#else
    fprintf(stderr, "Error: failed to load '%s': %s\n", library_path,
            dlerror());
#endif
    return -1;
  }

  RESOLVE(init, "runtime_init");
  RESOLVE(load_models, "runtime_load_models");
  RESOLVE(enqueue_input, "runtime_enqueue_input");
  RESOLVE(retrieve_output, "runtime_retrieve_output");
  RESOLVE(cleanup, "runtime_cleanup");
  RESOLVE(get_error, "runtime_get_error");
  RESOLVE(get_version, "runtime_get_version");
  RESOLVE(get_name, "runtime_get_name");
  RESOLVE(get_info, "runtime_get_info");
  return 0;
}

void unload_runtime_library(Runtime *runtime) {
  if (runtime == NULL || runtime->handle == NULL) return;
  CLOSE_LIBRARY(runtime->handle);
  *runtime = (Runtime){0};
}

void report_error(const Runtime *runtime, const char *what,
                  RuntimeStatus status) {
  const char *message = runtime->get_error();
  fprintf(stderr, "Error: %s failed (status %d): %s\n", what, (int)status,
          message ? message : "n/a");
}
