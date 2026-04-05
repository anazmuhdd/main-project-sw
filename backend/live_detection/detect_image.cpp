#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <awnn_lib.h>

extern "C" {
    unsigned char *yolov5_pre_process_mat(const cv::Mat& frame, unsigned int *file_size);
    int yolov5_post_process_mat(cv::Mat& frame, float **output, float fps);
}

int main(int argc, char **argv) {
    if(argc < 3) {
        printf("Usage: %s <model.nb> <image_path> [output_path]\n", argv[0]);
        printf("Example: %s ./model/yolov5.nb ../../person.jpg output_person.jpg\n", argv[0]);
        return -1;
    }

    const char* nbg = argv[1];
    const char* image_path = argv[2];
    const char* output_path = (argc > 3) ? argv[3] : "output.jpg";

    // Initialize NPU
    awnn_init();
    Awnn_Context_t *context = awnn_create(nbg);
    if (!context) {
        fprintf(stderr, "Failed to create NPU context from %s\n", nbg);
        return -1;
    }

    // Load Image
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        fprintf(stderr, "Failed to load image %s\n", image_path);
        awnn_destroy(context);
        awnn_uninit();
        return -1;
    }

    printf("Detecting objects in %s...\n", image_path);

    // Pre-process
    unsigned int file_size;
    unsigned char* input_data = yolov5_pre_process_mat(frame, &file_size);

    // Inference
    auto infer_start = std::chrono::steady_clock::now();
    void *input_buffers[] = {input_data};
    awnn_set_input_buffers(context, input_buffers);
    awnn_run(context);
    float **results = awnn_get_output_buffers(context);
    auto infer_end = std::chrono::steady_clock::now();

    // Post-process & Draw
    yolov5_post_process_mat(frame, results, 0.0f); // 0 FPS for single image

    free(input_data);

    // Save Output
    if (cv::imwrite(output_path, frame)) {
        printf("Detection saved to %s\n", output_path);
    } else {
        fprintf(stderr, "Failed to save output image to %s\n", output_path);
    }

    std::chrono::duration<float> infer_elapsed = infer_end - infer_start;
    printf("Inference Latency: %.2f ms\n", infer_elapsed.count() * 1000.0f);

    // Cleanup
    awnn_destroy(context);
    awnn_uninit();

    return 0;
}
