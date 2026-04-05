#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include "yolov5_post_process.h"

using namespace std;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float desigmoid(float x)
{
    return static_cast<float>(-log(1.f / (x + 1e-7f) - 1.f)); // Avoid division by zero
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(faceobjects, left, j);
    if (i < right) qsort_descent_inplace(faceobjects, i, right);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = faceobjects[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / (union_area + 1e-7f) > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

static void generate_proposals(int stride, const float* feat, float prob_threshold, std::vector<Object>& objects,
                               int letterbox_cols, int letterbox_rows)
{
    static float anchors[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
    int anchor_num = 3;
    int feat_w = letterbox_cols / stride;
    int feat_h = letterbox_rows / stride;
    int cls_num = 80;
    int anchor_group = (stride == 8) ? 1 : (stride == 16) ? 2 : 3;

    float deprob_threshold = desigmoid(prob_threshold);
    int feat_size = feat_w * feat_h;
    int feat_size_cls_5 = feat_size * (cls_num + 5);

    for (int h = 0; h < feat_h; h++)
    {
        for (int w = 0; w < feat_w; w++)
        {
            for (int a = 0; a < anchor_num; a++)
            {
                int a_idx = a * feat_size_cls_5 + h * feat_w * (cls_num + 5) + w * (cls_num + 5);
                float box_score = feat[a_idx + 4];
                if (box_score < deprob_threshold) continue;

                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int s = 0; s < cls_num; s++)
                {
                    float score = feat[a_idx + 5 + s];
                    if (score > class_score)
                    {
                        class_index = s;
                        class_score = score;
                    }
                }

                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if (final_score >= prob_threshold)
                {
                    float dx = sigmoid(feat[a_idx + 0]);
                    float dy = sigmoid(feat[a_idx + 1]);
                    float dw = sigmoid(feat[a_idx + 2]);
                    float dh = sigmoid(feat[a_idx + 3]);
                    float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
                    float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
                    float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                    float pred_w = dw * dw * 4.0f * anchor_w;
                    float pred_h = dh * dh * 4.0f * anchor_h;
                    float x0 = pred_cx - pred_w * 0.5f;
                    float y0 = pred_cy - pred_h * 0.5f;
                    float x1 = pred_cx + pred_w * 0.5f;
                    float y1 = pred_cy + pred_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    objects.push_back(obj);
                }
            }
        }
    }
}

static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects, float **output)
{
    int size0 = 1*3*80*80*85;
    int size1 = 1*3*40*40*85;
    int size2 = 1*3*20*20*85;
    
    std::vector<Object> proposals;
    generate_proposals(32, output[2], 0.4f, proposals, 640, 640);
    generate_proposals(16, output[1], 0.4f, proposals, 640, 640);
    generate_proposals(8, output[0], 0.4f, proposals, 640, 640);

    if (proposals.empty()) return 0;
    
    qsort_descent_inplace(proposals, 0, proposals.size() - 1);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, 0.45f);

    int letterbox_rows = 640;
    int letterbox_cols = 640;
    float scale_letterbox = min(letterbox_rows * 1.0f / bgr.rows, letterbox_cols * 1.0f / bgr.cols);
    int resize_rows = int(scale_letterbox * bgr.rows);
    int resize_cols = int(scale_letterbox * bgr.cols);
    int tmp_h = (letterbox_rows - resize_rows) / 2;
    int tmp_w = (letterbox_cols - resize_cols) / 2;
    float ratio_x = (float)bgr.rows / resize_rows;
    float ratio_y = (float)bgr.cols / resize_cols;

    objects.resize(picked.size());
    for (size_t i = 0; i < picked.size(); i++)
    {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x - tmp_w) * ratio_y;
        float y0 = (objects[i].rect.y - tmp_h) * ratio_x;
        float x1 = (objects[i].rect.x + objects[i].rect.width - tmp_w) * ratio_y;
        float y1 = (objects[i].rect.y + objects[i].rect.height - tmp_h) * ratio_x;

        objects[i].rect.x = max(min(x0, (float)(bgr.cols - 1)), 0.f);
        objects[i].rect.y = max(min(y0, (float)(bgr.rows - 1)), 0.f);
        objects[i].rect.width = max(min(x1, (float)(bgr.cols - 1)), 0.f) - objects[i].rect.x;
        objects[i].rect.height = max(min(y1, (float)(bgr.rows - 1)), 0.f) - objects[i].rect.y;
    }
    return 0;
}

static void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects, float fps)
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

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        cv::rectangle(bgr, obj.rect, cv::Scalar(0, 255, 0), 2);
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        cv::putText(bgr, text, cv::Point(obj.rect.x, obj.rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    char fps_text[32];
    sprintf(fps_text, "FPS: %.2f", fps);
    cv::putText(bgr, fps_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
}

extern "C"{
int yolov5_post_process_mat(cv::Mat& frame, float **output, float fps)
{
    std::vector<Object> objects;
    detect_yolov5(frame, objects, output);
    draw_objects(frame, objects, fps);
    return 0;
}
}
