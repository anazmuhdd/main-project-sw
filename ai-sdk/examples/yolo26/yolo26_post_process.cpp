#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include "yolo26_post_process.h"

#define CLASS_NUM 80
#define ANCHOR_NUM 8400
#define CONF_THRESHOLD 0.25f
#define NMS_THRESHOLD 0.45f

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} Detection;

static float iou(Detection a, Detection b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter);
}

static void nms(std::vector<Detection>& dets, float threshold) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (removed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (removed[j]) continue;
            if (iou(dets[i], dets[j]) > threshold) {
                removed[j] = true;
            }
        }
    }

    std::vector<Detection> result;
    for (size_t i = 0; i < dets.size(); i++) {
        if (!removed[i]) result.push_back(dets[i]);
    }
    dets = result;
}

int yolo26_post_process(float* data, int width, int height, det_res_t* res) {
    // data shape is [1, 84, 8400]
    std::vector<Detection> detections;

    // The data is likely [Channels, Anchors]
    // 0~3: x_center, y_center, width, height
    // 4~83: class scores
    
    for (int i = 0; i < ANCHOR_NUM; i++) {
        float max_score = 0;
        int class_id = -1;
        
        // Find best class
        for (int c = 0; c < CLASS_NUM; c++) {
            float score = data[(4 + c) * ANCHOR_NUM + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score > CONF_THRESHOLD) {
            float cx = data[0 * ANCHOR_NUM + i];
            float cy = data[1 * ANCHOR_NUM + i];
            float w = data[2 * ANCHOR_NUM + i];
            float h = data[3 * ANCHOR_NUM + i];

            Detection det;
            det.x1 = cx - w / 2.0f;
            det.y1 = cy - h / 2.0f;
            det.x2 = cx + w / 2.0f;
            det.y2 = cy + h / 2.0f;
            det.score = max_score;
            det.class_id = class_id;
            detections.push_back(det);
        }
    }

    nms(detections, NMS_THRESHOLD);

    int det_num = std::min((int)detections.size(), 64);
    res->num = det_num;
    for (int i = 0; i < det_num; i++) {
        res->results[i].rel_box.x = detections[i].x1 / 640.0f;
        res->results[i].rel_box.y = detections[i].y1 / 640.0f;
        res->results[i].rel_box.w = (detections[i].x2 - detections[i].x1) / 640.0f;
        res->results[i].rel_box.h = (detections[i].y2 - detections[i].y1) / 640.0f;
        res->results[i].score = detections[i].score;
        res->results[i].class_index = detections[i].class_id;
    }

    printf("Post-processing: Found %d detections\n", det_num);
    return 0;
}
