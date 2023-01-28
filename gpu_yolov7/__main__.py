if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gpu_yolov7.main:app", host="0.0.0.0", port=30001, workers=4, reload=True)