# Generate ONNX file : 
python convertONNX.py 

# Generate TRT files (select one of 3 data types): 
python ./TensorRT-For-YOLO-Series/export.py -o yolov8n_custom.onnx -e yolov8n_custom_fp32.trt --end2end --v8 -p fp32\
python ./TensorRT-For-YOLO-Series/export.py -o yolov8n_custom.onnx -e yolov8n_custom_fp16.trt --end2end --v8 -p fp16\
python ./TensorRT-For-YOLO-Series/export.py -o yolov8n_custom.onnx -e yolov8n_custom_int8.trt --end2end --v8 -p int8 --calib_input img_dir

# Use docker images : 
gcr.io/boostcap-final/yolov8-tensorrt 

# Pull & run Docker :
docker pull gcr.io/boostcap-final/yolov8-tensorrt \
nvidia-docker run -it -p 8002:8002 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 gcr.io/boostcap-final/yolov8-tensorrt
