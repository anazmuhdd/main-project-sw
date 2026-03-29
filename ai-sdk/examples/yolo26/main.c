#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "viplite_lib.h"
#include "yolo26_post_process.h"

// External pre-process function from yolo26_pre_process.cpp
unsigned char *yolo26_pre_process(const char* imagepath, unsigned int *file_size);

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model.nb> <image.jpg>\n", argv[0]);
        return -1;
    }

    char* model_path = argv[1];
    char* image_path = argv[2];
    struct timeval start, end;
    float time_use;

    printf("Initializing NPU...\n");
    vip_status_e status = vip_init();
    if (status != VIP_SUCCESS) {
        printf("vip_init failed: %d\n", status);
        return -1;
    }

    printf("Creating network from %s...\n", model_path);
    vip_network network = vip_create_network(model_path);
    if (network == NULL) {
        printf("vip_create_network failed\n");
        return -1;
    }

    // Set input
    unsigned int input_size = 0;
    unsigned char* input_data = yolo26_pre_process(image_path, &input_size);
    if (input_data == NULL) {
        printf("Pre-processing failed\n");
        return -1;
    }

    // Create Input Buffer
    vip_buffer_create_params_t input_params;
    memset(&input_params, 0, sizeof(input_params));
    input_params.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    vip_buffer input_buffer = vip_create_buffer(network, &input_params, sizeof(input_params));
    
    // Copy pre-processed data to NPU input buffer
    void* input_ptr = vip_map_buffer(input_buffer);
    memcpy(input_ptr, input_data, input_size);
    vip_unmap_buffer(input_buffer);
    free(input_data);

    // Create Output Buffer (1x84x8400 float, which is 84*8400*4 bytes)
    vip_buffer_create_params_t output_params;
    memset(&output_params, 0, sizeof(output_params));
    output_params.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    vip_buffer output_buffer = vip_create_buffer(network, &output_params, sizeof(output_params));

    status = vip_set_network_input(network, 0, input_buffer);
    status = vip_set_network_output(network, 0, output_buffer);

    printf("Running inference...\n");
    gettimeofday(&start, NULL);
    status = vip_run_network(network);
    gettimeofday(&end, NULL);
    time_use = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
    printf("NPU inference total: %.2f ms.\n", time_use);

    if (status != VIP_SUCCESS) {
        printf("vip_run_network failed: %d\n", status);
        return -1;
    }

    printf("Post-processing results...\n");
    void* output_ptr = vip_map_buffer(output_buffer);
    det_res_t results;
    memset(&results, 0, sizeof(det_res_t));
    
    // Call our NMS-based post-process
    yolo26_post_process((float*)output_ptr, 640, 640, &results);

    printf("Found %d detections\n", results.num);
    for (int i = 0; i < results.num; i++) {
        printf("  Det[%d]: class=%d, score=%.2f, box=[%.2f, %.2f, %.2f, %.2f]\n",
               i, results.results[i].class_index, results.results[i].score,
               results.results[i].rel_box.x * 640.0f, results.results[i].rel_box.y * 640.0f,
               results.results[i].rel_box.w * 640.0f, results.results[i].rel_box.h * 640.0f);
    }

    vip_unmap_buffer(output_buffer);
    vip_destroy_buffer(input_buffer);
    vip_destroy_buffer(output_buffer);
    vip_destroy_network(network);
    vip_destroy();
    return 0;
}
