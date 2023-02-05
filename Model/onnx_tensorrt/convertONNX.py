from ultralytics import YOLO

model = YOLO("../yolov8/yolov8n_custom.pt")
model.fuse()  
model.info(verbose=True)  # Print model information
model.export(format='onnx')  # TODO: 
