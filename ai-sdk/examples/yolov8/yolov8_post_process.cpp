/*
 * Adapted from YOLOv5 example for YOLOv8 (1x84x8400 output)
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "yolov8_post_process.h"

using namespace std;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static int detect_yolov8(const cv::Mat& bgr, std::vector<Object>& objects, float **output)
{
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;
    const int class_num = 80;
    const int anchor_num = 8400; // 640/8=80, 640/16=40, 640/32=20. 80*80 + 40*40 + 20*20 = 8400

    float *data = output[0]; // (1, 84, 8400)
    std::vector<Object> proposals;

    for (int i = 0; i < anchor_num; i++)
    {
        float max_score = 0;
        int class_id = -1;

        // Find which class has the maximum score
        for (int c = 0; c < class_num; c++)
        {
            // Data is [84][8400], so data[ (4+c)*8400 + i ]
            float score = data[(4 + c) * anchor_num + i];
            if (score > max_score)
            {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score > prob_threshold)
        {
            float cx = data[0 * anchor_num + i];
            float cy = data[1 * anchor_num + i];
            float w  = data[2 * anchor_num + i];
            float h  = data[3 * anchor_num + i];

            float x0 = cx - w * 0.5f;
            float y0 = cy - h * 0.5f;

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = w;
            obj.rect.height = h;
            obj.label = class_id;
            obj.prob = max_score;
            proposals.push_back(obj);
        }
    }

    // Sort by prob
    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // Scaling back to original image size
    int letterbox_rows = 640;
    int letterbox_cols = 640;
    float scale_letterbox;
    if ((letterbox_rows * 1.0 / bgr.rows) < (letterbox_cols * 1.0 / bgr.cols))
        scale_letterbox = letterbox_rows * 1.0 / bgr.rows;
    else
        scale_letterbox = letterbox_cols * 1.0 / bgr.cols;

    int resize_cols = int(scale_letterbox * bgr.cols);
    int resize_rows = int(scale_letterbox * bgr.rows);
    int tmp_h = (letterbox_rows - resize_rows) / 2;
    int tmp_w = (letterbox_cols - resize_cols) / 2;

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        // Adjust for letterbox offset and scale
        float x0 = (objects[i].rect.x - tmp_w) / scale_letterbox;
        float y0 = (objects[i].rect.y - tmp_h) / scale_letterbox;
        float x1 = (objects[i].rect.x + objects[i].rect.width - tmp_w) / scale_letterbox;
        float y1 = (objects[i].rect.y + objects[i].rect.height - tmp_h) / scale_letterbox;

        x0 = std::max(std::min(x0, (float)(bgr.cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(bgr.rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(bgr.cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(bgr.rows - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    fprintf(stderr, "detection num: %d\n", count);
    return 0;
}

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

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0), 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols) x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));
    }

    cv::imwrite("result.png", image);
}

extern "C"{
int yolov8_post_process(const char *imagepath, float **output)
{
    printf("yolov8_postprocess.cpp run. \n");
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov8(m, objects, output);
    draw_objects(m, objects);

    return 0;
}
}
