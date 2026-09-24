from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError
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
    session = AiohttpSession(proxy=settings.telegram_proxy_url) if settings.telegram_proxy_url else None
    bot = Bot(token=settings.telegram_bot_token, session=session)
    dispatcher = Dispatcher()
    dispatcher.include_router(start_router)
    dispatcher.include_router(make_router(settings))
    logger.info("Item Cards AI bot started")
    # fic_comp occasionally times out while opening Telegram's HTTPS endpoint.
    # Keep the worker alive and retry transient network failures instead of
    # requiring a person to SSH in and restart it.
    while True:
        try:
            await dispatcher.start_polling(bot, close_bot_session=False)
        except (TelegramNetworkError, asyncio.TimeoutError):
            logger.exception("Telegram connection failed; retrying in 5 seconds")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
