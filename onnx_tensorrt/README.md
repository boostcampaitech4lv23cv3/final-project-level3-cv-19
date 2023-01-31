Use docker images :
gcr.io/boostcap-final/yolov8-tensorrt

Pull & run Docker :
docker pull gcr.io/boostcap-final/yolov8-tensorrt \
nvidia-docker run -it -p 8002:8002 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 gcr.io/boostcap-final/yolov8-tensorrt

