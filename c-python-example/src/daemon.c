/* Copyright 2023, Scailable, All rights reserved. */

/* Based on code of Koynov Stas: skojnov@yandex.ru | BSD 3-Clause License 2015 */

#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "daemon.h"

void daemon_error_exit(const char *format, ...) {
    va_list ap;

    if (format && *format) {
        va_start(ap, format);
        fprintf(stderr, "%s: ", "example");
        vfprintf(stderr, format, ap);
        va_end(ap);
    }

    _exit(EXIT_FAILURE);
}

void set_module_sig_handler(int sig, signal_handler_t handler) {
    if (signal(sig, handler) == SIG_ERR)
        daemon_error_exit("Can't set handler for signal: %d %m\n", sig);
}
