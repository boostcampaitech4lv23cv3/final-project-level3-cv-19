from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")
general_pages_router = APIRouter()


@general_pages_router.get("/{user_id}")
async def home(user_id, request: Request):
    return templates.TemplateResponse(f"html/{user_id}/subs.html", {"request": request})
