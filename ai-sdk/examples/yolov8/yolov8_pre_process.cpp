#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>

using namespace std;

void get_input_data_v8(const char* image_file, unsigned char* input_data, int letterbox_rows, int letterbox_cols)
{
    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;

    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / img.rows) < (letterbox_cols * 1.0 / img.cols))
        scale_letterbox = letterbox_rows * 1.0 / img.rows;
    else
        scale_letterbox = letterbox_cols * 1.0 / img.cols;

    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    img.convertTo(img, CV_32FC3);
    
    // Create background (filling)
    cv::Mat img_new(letterbox_cols, letterbox_rows, CV_32FC3, cv::Scalar(0, 0, 0));
    int top = (letterbox_rows - resize_rows) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    
    // Copy resized image into the center
    cv::Rect roi(left, top, resize_cols, resize_rows);
    img.copyTo(img_new(roi));

    img_new.convertTo(img_new, CV_32FC3);
    float* img_data = (float*)img_new.data;

    /* nhwc to nchw */
    for (int h = 0; h < letterbox_rows; h++)
    {
        for (int w = 0; w < letterbox_cols; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * letterbox_cols * 3 + w * 3 + c;
                int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                // Since this model uses uint8 quantization for input, we pass the pixel values directly
                input_data[out_index] = (unsigned char)(img_data[in_index]);
            }
        }
    }
}

extern "C"{
unsigned char *yolov8_pre_process(const char* imagepath, unsigned int *file_size)
{
    int img_c = 3;

	// set default letterbox size
    int letterbox_rows = 640;
    int letterbox_cols = 640;
    int img_size = letterbox_rows * letterbox_cols * img_c;

    *file_size = img_size * sizeof(unsigned char);

    unsigned char *tensorData = (unsigned char *)malloc(1 * img_size * sizeof(unsigned char));

    get_input_data_v8(imagepath, tensorData, letterbox_rows, letterbox_cols);

    return tensorData;
}
}
