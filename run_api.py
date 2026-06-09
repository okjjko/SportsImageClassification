"""FastAPI 启动入口

使用方法:
    python run_api.py
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api import router

app = FastAPI(title="Sports Image Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

PROJECT_ROOT = Path(__file__).resolve().parent

# 挂载示例图片静态目录
example_dir = PROJECT_ROOT / "archive" / "test"
if example_dir.exists():
    app.mount("/api/examples/static", StaticFiles(directory=str(example_dir)), name="examples")

# 挂载历史图片目录
history_dir = PROJECT_ROOT / "web" / "history_images"
history_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/files/history", StaticFiles(directory=str(history_dir)), name="history_images")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
