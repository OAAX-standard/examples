#include "timer.h"
#include <locale.h>

static long get_current_us(){
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_usec;
}

Timer *start_recording(Timer *timer){
    Timer *new_timer = (Timer *) malloc(sizeof(Timer));
    new_timer->start = get_current_us();
    new_timer->elapsed_time = 0;
    new_timer->next = NULL;
    if (timer != NULL){
        timer->next = new_timer;
        new_timer->root = timer->root;
    } else {
        new_timer->root = new_timer;
    }
    return new_timer;
}

long stop_recording(Timer *timer){
    timer->end = get_current_us();
    timer->elapsed_time = timer->end - timer->start;
    printf("Elapsed time: %li us - %li ms\n", timer->elapsed_time, timer->elapsed_time / 1000);
    return timer->elapsed_time;
}

long get_total_elapsed_timer(Timer *timer){
    if (timer == NULL) return 0;
    timer = timer->root;
    long sum = 0;
    while(timer != NULL){
        sum += timer->elapsed_time;
        timer = timer->next;
    }
    printf("Totat elpased time: %'li us - %li ms\n", sum, sum / 1000);
    return sum;
}

void free_all_timers(Timer *root){
    if(root == NULL) return;
    free_all_timers(root->next);
    free(root);
}