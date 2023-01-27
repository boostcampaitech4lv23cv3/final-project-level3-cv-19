from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.fuse()  
model.info(verbose=True)  # Print model information
model.export(format='onnx')  # TODO: 
