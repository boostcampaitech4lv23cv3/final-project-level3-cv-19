if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gpu_dummy.main:app", host="0.0.0.0", port=30002, workers=4, reload=True)