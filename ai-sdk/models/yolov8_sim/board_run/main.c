#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#include <awnn_lib.h>

#include "yolov8_pre_process.h"
#include "yolov8_post_process.h"

int main(int argc, char **argv) {
    if(argc < 3)
    {
        printf("Usage: %s <nbg> <input.jpg>\n", argv[0]);
        return -1;
    }
    const char* nbg = argv[1];
    const char* input = argv[2];

    printf("NPU Init...\n");
    awnn_init();
    
    printf("Create Network: %s\n", nbg);
    Awnn_Context_t *context = awnn_create(nbg);
    if (!context) {
        printf("Failed to create network context\n");
        return -1;
    }

    unsigned int file_size;
    printf("Preprocessing: %s\n", input);
    unsigned char* plant_data = yolov8_pre_process(input, &file_size);
    if (!plant_data) {
        printf("Pre-processing failed\n");
        return -1;
    }

    void *input_buffers[] = {plant_data};
    awnn_set_input_buffers(context, input_buffers);
    
    printf("NPU Run...\n");
    awnn_run(context);
    
    printf("Fetching Output...\n");
    float **results = awnn_get_output_buffers(context);
    
    printf("Postprocessing...\n");
    yolov8_post_process(input, results);

    free(plant_data);
    awnn_destroy(context);
    awnn_uninit();
    
    printf("Done. Result saved to result.png\n");
    return 0;
}
