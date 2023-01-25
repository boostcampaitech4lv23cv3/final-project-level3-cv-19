if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.relay:app", host="127.0.0.1", port=8002, reload=True)
    # uvicorn.run("app.main:application", host="127.0.0.1", port=8001, reload=True)
