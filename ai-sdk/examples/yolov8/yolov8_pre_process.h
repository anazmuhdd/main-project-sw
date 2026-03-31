#ifndef __YOLOV8_PRE_PROCESS_H__
#define __YOLOV8_PRE_PROCESS_H__
#ifdef __cplusplus
extern "C" {
#endif

unsigned char *yolov8_pre_process(const char* imagepath, unsigned int *file_size);

#ifdef __cplusplus
}
#endif

#endif
