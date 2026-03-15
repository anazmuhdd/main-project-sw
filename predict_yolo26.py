from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('yolo26n.pt')

# Run inference on the dog image
results = model('/home/radxa/test/ai-sdk/examples/yolov5/input_data/dog.jpg')

# Save the result
for r in results:
    r.save(filename='yolo26_cpu_result.jpg')
    print("Result saved to yolo26_cpu_result.jpg")
