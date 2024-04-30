#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>

#include <sys/file.h>
#include <sys/sysinfo.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <dlfcn.h>

#include "daemon.h"
#include "shm_utils.h"
#include "runtime_utils.h"
#include "io_utils.h"

// #####################################

// Runtime Interface
typedef int (*runtime_initialization_t)();
typedef int (*runtime_model_loading_t)(const char *);
typedef int (*runtime_inference_execution_t)(tensors_struct *, tensors_struct *);
typedef int (*runtime_inference_cleanup_t)();
typedef int (*runtime_destruction_t)();
typedef const char *(*runtime_error_message_t)();
typedef const char *(*runtime_version_t)();
typedef const char *(*runtime_name_t)();

runtime_initialization_t runtime_initialization;
runtime_initialization_t runtime_model_loading;
runtime_inference_execution_t runtime_inference_execution;
runtime_inference_cleanup_t runtime_inference_cleanup;
runtime_destruction_t runtime_destruction;
runtime_error_message_t runtime_error_message;
runtime_version_t runtime_version;
runtime_name_t runtime_name;
void *handle;

pthread_t daemon_listen_thread;
int interrupt_signal = 0;

// Constants to communicate with a client
char *engine_pipe_name;
char *module_pipe_name;
int shm_id;
key_t shm_key;

// Forward declaration of functions
static void daemon_exit_handler(int sig);

// Helper logging functions
static void printf_error(const char *error_message) {
    printf("Error: RUNTIME - %s\n", error_message);
}

static void printf_notice(const char *notice_message) {
    printf("Notice: RUNTIME - %s\n", notice_message);
}

static void printf_warning(const char *warning_message) {
    printf("Warning: RUNTIME - %s\n", warning_message);
}

static void print_error(const char *error_message) {
    printf_error(error_message);
    printf("{\"error\":\"%s\"}\n", error_message);
}

static void daemon_exit_handler(int sig) {
    // Set interrupt signal which will case engine_start_listen_thread to exit
    interrupt_signal = sig;
}

void daemon_init_signals(void) {

    set_module_sig_handler(SIGINT, daemon_exit_handler);// for Ctrl-C in terminal for debug (in debug mode)
    set_module_sig_handler(SIGTERM, daemon_exit_handler);

    set_module_sig_handler(SIGCHLD, SIG_IGN);// ignore child
    set_module_sig_handler(SIGTSTP, SIG_IGN);// ignore tty signals
    set_module_sig_handler(SIGTTOU, SIG_IGN);
    set_module_sig_handler(SIGTTIN, SIG_IGN);
    set_module_sig_handler(SIGHUP, SIG_IGN);

    // We expect write failures to occur but we want to handle them where
    // the error occurs rather than in a SIGPIPE handler.
    signal(SIGPIPE, SIG_IGN);

    // Set death signal when parent is terminated
    prctl(PR_SET_PDEATHSIG, SIGTERM);
}

void *engine_start_listen_thread(void *thread_arg) {
    tensors_struct input_tensors, output_tensors;

    (void) (thread_arg);

    size_t result_len;
    size_t input_data_length;
    char *input_message_buffer;
    size_t shm_capacity = shm_get_size(shm_id);
    int engine_pipe = open_pipe_reading(engine_pipe_name);
    int module_pipe = open_pipe_writing(module_pipe_name);
    // Send signal that inference engine is ready
    printf_notice("Sending ready signal to module.");
    pipe_send(module_pipe, 1);
    while (interrupt_signal == 0) {
        // Wait for signal from the module that the data is ready
//        printf_notice("Inference Engine is waiting for signal.");
        int error = pipe_timed_read(engine_pipe, 1);
        if (error <= 0) {
            // Wait timed out, restart loop ( effectively checking for interrupt signal )
            continue;
        }
        printf_notice("Inference Engine received signal.");

        // Read data from shared memory
        void *data_pointer = shm_read(shm_id, &input_data_length, &input_message_buffer);
        if (data_pointer == NULL) {
            // It's possible that shm was realloc'd by another process, this changes the ID. Reacquire ID
            shm_id = shm_get(shm_key);
            // Try to read again with new ID
            data_pointer = shm_read(shm_id, &input_data_length, &input_message_buffer);
            if (data_pointer == NULL) {
                printf_warning("Inference Engine could not read from SHM");
                continue;
            }
            shm_capacity = shm_get_size(shm_id);
        }

        printf("Notice: Inference Engine received message of length: %zu\n", input_data_length);

        // Decode the MsgPack message
        parse_input_data(input_message_buffer, input_data_length, &input_tensors.num_tensors,
                                    &input_tensors.data, &input_tensors.shapes, &input_tensors.ranks);
        input_tensors.data_types = NULL;
        input_tensors.names = NULL;

        printf("Number of input tenors: %zu\n", input_tensors.num_tensors);
        for (size_t i = 0; i < input_tensors.num_tensors; i++) {
            printf("Input tensor %zu has rank %zu\n", i, input_tensors.ranks[i]);
        }

        // Perform inference on the received data
        error = runtime_inference_execution(&input_tensors, &output_tensors);
        // Print message if error
        if (error) {
            printf_error(runtime_error_message());
        }

        // Clean up input tensors
        for (size_t i = 0; i < input_tensors.num_tensors; i++) {
            free(input_tensors.shapes[i]);
        }
        free(input_tensors.shapes);
        free(input_tensors.ranks);
        free(input_tensors.data);

        char *output_message;
        build_output_mpack(output_tensors.names, output_tensors.num_tensors,
                                      output_tensors.data, output_tensors.shapes,
                                      output_tensors.ranks, output_tensors.data_types,
                                      &output_message, &result_len);

        if (result_len > shm_capacity) {
            // Shm not big enough for result. Realloc ( module will have to detect that existing ID is not longer valid )
            shm_close(data_pointer);
            shm_id = shm_realloc(shm_key, shm_id, result_len);
            shm_write(shm_id, output_message, result_len);
        } else {
            // Memory still attached. Reuse
            // Write the length header
            uint32_t data_length = (uint32_t) result_len;
            memcpy(data_pointer, &data_length, sizeof(data_length));
            // Write payload data
            memcpy(input_message_buffer, output_message, result_len);
            // Detach data
            shm_close(data_pointer);
        }

        // Signal module that SHM is ready to be read
        pipe_send(module_pipe, 1);

        // result length
        printf("Notice: RUNTIME - Model completed succesfully, result_len %zu.\n", result_len);

        // free runtime core pointers
        if (runtime_inference_cleanup()) {
            print_error(runtime_error_message());
        }
    }

    pipe_close(engine_pipe);
    pipe_close(module_pipe);
    return NULL;
}

void daemon_init(void *data) {

    daemon_init_signals();
    if (pthread_create(&daemon_listen_thread, NULL, engine_start_listen_thread, data) != 0) {
        printf_error("Can't create daemon_listen_thread.");
        daemon_error_exit("Error: RUNTIME - Can't create daemon_listen_thread: %m\n");
    }
}

int main(int32_t argc, char *argv[]) {
    if (argc != 7) {
        printf("ERROR: Not enough parameters given to inference engine.\n");
        printf("Usage: %s <runtime library path> <model filepath> <engine pipe> <module pipe> <shm id> <shm key>\n",
               argv[0]);
        exit(EXIT_FAILURE);
    }
    char *runtime_library_path = argv[1];
    const char *model_filepath = argv[2];
    engine_pipe_name = argv[3];
    module_pipe_name = argv[4];
    shm_id = atoi(argv[5]);
    shm_key = atoi(argv[6]);

    printf("Notice: RUNTIME - Using runtime library path: %s\n", runtime_library_path);
    printf("Notice: RUNTIME Started with engine pipe: %s module pipe: %s SHM ID: %d and SHM key %d\n",
           engine_pipe_name, module_pipe_name, shm_id, shm_key);

    // Load runtime library
    handle = dlopen(runtime_library_path, RTLD_NOW);
    if (!handle) {
        printf_error("Unable to load runtime library.");
        printf_error(dlerror());
        exit(EXIT_FAILURE);
    }

    runtime_initialization = (runtime_initialization_t) dlsym(handle, "runtime_initialization");
    runtime_model_loading = (runtime_model_loading_t) dlsym(handle, "runtime_model_loading");
    runtime_inference_execution = (runtime_inference_execution_t) dlsym(handle, "runtime_inference_execution");
    runtime_inference_cleanup = (runtime_inference_cleanup_t) dlsym(handle, "runtime_inference_cleanup");
    runtime_destruction = (runtime_destruction_t) dlsym(handle, "runtime_destruction");
    runtime_error_message = (runtime_error_message_t) dlsym(handle, "runtime_error_message");
    runtime_version = (runtime_version_t) dlsym(handle, "runtime_version");
    runtime_name = (runtime_name_t) dlsym(handle, "runtime_name");

    if (!runtime_initialization || !runtime_inference_execution || !runtime_inference_cleanup || !runtime_destruction
            || !runtime_error_message || !runtime_version || !runtime_name) {
        printf_error("Unable to load runtime functions.");
        printf_error(dlerror());
        dlclose(handle);
        exit(EXIT_FAILURE);
    }

    // Log information about the runtime
    const char *name = runtime_name();
    const char *version = runtime_version();
    printf("Notice: RUNTIME - Runtime name: '%s', version: '%s'\n", name, version);

    // initialize runtime environment
    if (runtime_initialization()) {
        print_error(runtime_error_message());
        dlclose(handle);
        exit(EXIT_FAILURE);
    }

    // load model file
    if (runtime_model_loading(model_filepath)) {
        print_error(runtime_error_message());
        dlclose(handle);
        exit(EXIT_FAILURE);
    }

    printf_notice("Initializing ...");

    // Start inference engine thread. Will listen for inputs in a loop
    daemon_init(NULL);

    // Wait for daemon to terminate
    pthread_join(daemon_listen_thread, NULL);

    // This is the end of the inference engine, so we can free the model
    printf_notice("Finalizing ...");

    // finalize runtime environment
    if (runtime_destruction()) {
        print_error(runtime_error_message());
        dlclose(handle);
        exit(EXIT_FAILURE);
    }

    printf("Information: Inference engine exited succesfully.\n");

    return EXIT_SUCCESS;
}
