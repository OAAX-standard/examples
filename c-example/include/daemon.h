/*
 * daemon.h
 *
 *
 * version 1.1
 *
 * BSD 3-Clause License
 *
 * Copyright (c) 2015, Koynov Stas - skojnov@yandex.ru
 *
 */

#ifndef DAEMON_HEADER
#define DAEMON_HEADER

#include <stddef.h>  //for NULL

#define DAEMON_DEF_TO_STR_(text) #text
#define DAEMON_DEF_TO_STR(arg) DAEMON_DEF_TO_STR_(arg)


#define DAEMON_MAJOR_VERSION_STR  DAEMON_DEF_TO_STR(DAEMON_MAJOR_VERSION)
#define DAEMON_MINOR_VERSION_STR  DAEMON_DEF_TO_STR(DAEMON_MINOR_VERSION)
#define DAEMON_PATCH_VERSION_STR  DAEMON_DEF_TO_STR(DAEMON_PATCH_VERSION)

#define DAEMON_VERSION_STR DAEMON_MAJOR_VERSION_STR "." DAEMON_MINOR_VERSION_STR "." DAEMON_PATCH_VERSION_STR

void daemon_error_exit(const char *format, ...);

typedef void (*signal_handler_t)(int);

void set_module_sig_handler(int sig, signal_handler_t handler);

#endif //DAEMON_HEADER
