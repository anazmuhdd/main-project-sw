# TARGET
TARGET = live_detection
TARGET_IMAGE = detect_image

# Paths
AI_SDK_DIR = /home/radxa/Project/main-project-sw/ai-sdk
PLATFORM = a733
NPU_VERSION = v2.0

# Dependencies
INCLUDE = -I$(AI_SDK_DIR)/examples/libawnn_viplite \
          -I$(AI_SDK_DIR)/examples/libawutils \
          -I$(AI_SDK_DIR) \
          -I/usr/include/opencv4 \
          -I$(AI_SDK_DIR)/viplite-tina/lib/aarch64-none-linux-gnu/$(NPU_VERSION)/inc

LIBS = -L/usr/lib/aarch64-linux-gnu \
       -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lpthread -ljpeg

# NPU Libraries
LIBS_VIP = -L$(AI_SDK_DIR)/viplite-tina/lib/aarch64-none-linux-gnu/$(NPU_VERSION)/ \
           -lNBGlinker -lVIPhal

# Source Files
SRCS_BASE = yolov5_post_process.cpp \
            yolov5_pre_process.cpp \
            $(AI_SDK_DIR)/examples/libawnn_viplite/awnn_lib.c \
            $(AI_SDK_DIR)/examples/libawnn_viplite/awnn_quantize.c \
            $(AI_SDK_DIR)/examples/libawutils/image_utils.c

SRCS_LIVE = main.cpp $(SRCS_BASE)
SRCS_IMAGE = detect_image.cpp $(SRCS_BASE)

# Compiler and Flags
CXX = g++
CFLAGS = -g -O2 -DDEBUG -D_DEBUG -DNPU_SW_VERSION=2

# Build Rules
all: $(TARGET) $(TARGET_IMAGE)

$(TARGET): $(SRCS_LIVE)
	$(CXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) $(LIBS_VIP) -lm

$(TARGET_IMAGE): $(SRCS_IMAGE)
	$(CXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) $(LIBS_VIP) -lm

clean:
	rm -rf $(TARGET) $(TARGET_IMAGE) *.o *~
