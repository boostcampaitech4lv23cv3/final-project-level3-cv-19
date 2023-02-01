"""
This Codes Temporarily Used for Develop Convenient Purpose.
Not Used from 01Feb2023 because of Server Integration.
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:application", host="0.0.0.0", port=8001, reload=True, workers=4)
