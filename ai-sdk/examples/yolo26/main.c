#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#include <awnn_lib.h>

#include "yolo26_pre_process.h"
#include "yolo26_post_process.h"

int main(int argc, char **argv) {
    if(argc < 3)
    {
        printf("Usage: %s <nbg_model> <input_image>\n", argv[0]);
        return -1;
    }
    const char* nbg = argv[1];
    const char* input = argv[2];

    printf("Initializing NPU...\n");
    awnn_init();
    
    printf("Creating network from %s...\n", nbg);
    Awnn_Context_t *context = awnn_create(nbg);
    if (!context) {
        printf("Failed to create network context!\n");
        return -1;
    }

    printf("Preprocessing image %s...\n", input);
    unsigned int file_size;
    unsigned char* plant_data = yolo26_pre_process(input, &file_size);

    void *input_buffers[] = {plant_data};
    awnn_set_input_buffers(context, input_buffers);
    
    printf("Running inference...\n");
    awnn_run(context);
    
    printf("Getting output buffers...\n");
    float **results = awnn_get_output_buffers(context);
    
    printf("Post-processing results...\n");
    yolo26_post_process(input, results);

    free(plant_data);
    awnn_destroy(context);
    awnn_uninit();

    printf("Done!\n");
    return 0;
}
