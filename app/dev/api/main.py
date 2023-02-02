from fastapi import FastAPI
from app.api.router import general_pages_router


def include_router(application):
	application.include_router(general_pages_router)


def start_application():
	application = FastAPI()
	include_router(application)
	return application


app = start_application()
