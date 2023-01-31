#!/bin/sh
python ./TensorRT-For-YOLO-Series/export.py -o yolov8n_custom.onnx -e yolov8n_custom.trt --end2end --v8
