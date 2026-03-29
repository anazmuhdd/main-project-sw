#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include "yolo26_post_process.h"

using namespace std;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));
    }

    cv::imwrite("result.png", image);
    printf("Result saved to result.png\n");
}

extern "C"{
int yolo26_post_process(const char *imagepath, float **output)
{
    printf("yolo26_postprocess.cpp run. \n");

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    // YOLO26 output is [1, 300, 6]
    // output[0] points to the begining of these 1800 floats.
    float *data = output[0];
    const float prob_threshold = 0.25f;

    std::vector<Object> objects;
    
    // Rescale logic
    int letterbox_rows = 640;
    int letterbox_cols = 640;
    
    float scale_letterbox;
    if ((letterbox_rows * 1.0 / m.rows) < (letterbox_cols * 1.0 / m.cols))
        scale_letterbox = letterbox_rows * 1.0 / m.rows;
    else
        scale_letterbox = letterbox_cols * 1.0 / m.cols;

    int resize_cols = int(scale_letterbox * m.cols);
    int resize_rows = int(scale_letterbox * m.rows);
    int tmp_h = (letterbox_rows - resize_rows) / 2;
    int tmp_w = (letterbox_cols - resize_cols) / 2;
    float ratio_x = (float)m.cols / resize_cols;
    float ratio_y = (float)m.rows / resize_rows;

    for (int i = 0; i < 300; i++)
    {
        float *box = data + i * 6;
        float score = box[4];
        if (score < prob_threshold) continue;

        float x0 = box[0];
        float y0 = box[1];
        float x1 = box[2];
        float y1 = box[3];
        int label = (int)box[5];

        // Rescale to original image
        x0 = (x0 - tmp_w) * ratio_x;
        y0 = (y0 - tmp_h) * ratio_y;
        x1 = (x1 - tmp_w) * ratio_x;
        y1 = (y1 - tmp_h) * ratio_y;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = label;
        obj.prob = score;
        objects.push_back(obj);
    }

    fprintf(stderr, "detection num: %d\n", (int)objects.size());
    draw_objects(m, objects);

    return 0;
}
}
