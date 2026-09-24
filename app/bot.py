from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from loguru import logger

from app.config import BASE_DIR, get_settings
from app.handlers.generate import make_router
from app.handlers.start import router as start_router


async def main() -> None:
    # CUDA_VISIBLE_DEVICES is intentionally read before any lazy model import.
    load_dotenv(BASE_DIR / ".env")
    settings = get_settings()
    settings.create_directories()
    logger.remove()
    logger.add(lambda message: print(message, end=""), level=settings.log_level)
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(start_router)
    dispatcher.include_router(make_router(settings))
    logger.info("Item Cards AI bot started")
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
