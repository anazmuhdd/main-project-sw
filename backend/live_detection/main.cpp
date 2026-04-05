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
        printf("Usage: %s <model.nb> <video_device>\n", argv[0]);
        printf("Example: %s ./model/yolov5.nb /dev/video0\n", argv[0]);
        return -1;
    }

    const char* nbg = argv[1];
    const char* video_device = argv[2];

    // Initialize NPU
    awnn_init();
    Awnn_Context_t *context = awnn_create(nbg);
    if (!context) {
        fprintf(stderr, "Failed to create NPU context from %s\n", nbg);
        return -1;
    }

    // Open Camera
    cv::VideoCapture cap(video_device);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open video device %s\n", video_device);
        return -1;
    }

    // Set resolution (optional, 640x480 is standard)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    printf("Starting live detection... Press 'q' to exit.\n");

    cv::Mat frame;
    float fps = 0;
    auto start_time = std::chrono::steady_clock::now();
    int frame_count = 0;

    while (true) {
        auto frame_start = std::chrono::steady_clock::now();

        cap >> frame;
        if (frame.empty()) break;

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
        yolov5_post_process_mat(frame, results, fps);

        free(input_data);

        // Display
        cv::imshow("NPU YOLOv5 Live Detection", frame);

        auto frame_end = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = frame_end - frame_start;
        std::chrono::duration<float> infer_elapsed = infer_end - infer_start;

        frame_count++;
        if (frame_count >= 10) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<float> total_elapsed = now - start_time;
            fps = frame_count / total_elapsed.count();
            start_time = now;
            frame_count = 0;
            printf("Inference Latency: %.2f ms | Overall FPS: %.2f\n", infer_elapsed.count() * 1000.0f, fps);
        }

        if (cv::waitKey(1) == 'q') break;
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    awnn_destroy(context);
    awnn_uninit();

    return 0;
}
