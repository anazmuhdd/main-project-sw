#ifndef YOLOV8_POST_PROCESS_H
#define YOLOV8_POST_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x;
    float y;
    float w;
    float h;
} rel_box_t;

typedef struct {
    rel_box_t rel_box;
    float score;
    int class_index;
} det_res_single_t;

typedef struct {
    int num;
    det_res_single_t results[64];
} det_res_t;

int yolov8_post_process(float* data, int width, int height, det_res_t* res);

#ifdef __cplusplus
}
#endif

#endif
