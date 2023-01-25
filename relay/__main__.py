if __name__ == "__main__":
    import uvicorn
    uvicorn.run("relay.main:application", host="0.0.0.0", port=8002, reload=True, workers=4)