#ifndef YOLOV10_POST_PROCESS_H
#define YOLOV10_POST_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

int yolo26_post_process(const char *imagepath, float **output);

#ifdef __cplusplus
}
#endif

#endif
