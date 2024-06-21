#ifndef TIMER
#define TIMER

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct Timer{
    struct Timer *root, *next;
    long start;
    long end;
    long elapsed_time;
} Timer;

// Start recording 
Timer *start_recording(Timer *timer);

// Return elapsed time in microseconds
long stop_recording(Timer *timer);

// Free the timers objects
void free_all_timers(Timer *root);

// Get elapsed time from start to finish
long get_total_elapsed_timer(Timer *timer);

#endif // TIMER