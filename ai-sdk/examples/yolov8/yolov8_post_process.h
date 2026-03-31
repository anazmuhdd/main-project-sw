#ifndef __YOLOV8_POST_PROCESS_H__
#define __YOLOV8_POST_PROCESS_H__
#ifdef __cplusplus
extern "C" {
#endif

int yolov8_post_process(const char *imagepath, float **output);

#ifdef __cplusplus
}
#endif

#endif
