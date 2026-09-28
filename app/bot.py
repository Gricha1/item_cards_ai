from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError
from dotenv import load_dotenv
from loguru import logger

from app.config import BASE_DIR, get_settings
from app.dashboard import start_dashboard
from app.handlers.generate import make_router as make_generation_router
from app.handlers.start import make_router as make_start_router
from app.services.analytics import AnalyticsStore


async def main() -> None:
    # CUDA_VISIBLE_DEVICES is intentionally read before any lazy model import.
    load_dotenv(BASE_DIR / ".env")
    settings = get_settings()
    settings.create_directories()
    analytics = AnalyticsStore(settings.analytics_db_path)
    analytics.initialize()
    logger.remove()
    logger.add(lambda message: print(message, end=""), level=settings.log_level)
    session = AiohttpSession(proxy=settings.telegram_proxy_url) if settings.telegram_proxy_url else None
    bot = Bot(token=settings.telegram_bot_token, session=session)
    dispatcher = Dispatcher()
    dispatcher.include_router(make_start_router(analytics))
    dispatcher.include_router(make_generation_router(settings, analytics))
    dashboard = await start_dashboard(
        analytics, settings.dashboard_host, settings.dashboard_port, settings.dashboard_token
    )
    logger.info(f"Analytics dashboard started on http://{settings.dashboard_host}:{settings.dashboard_port}/")
    logger.info("Item Cards AI bot started")
    # Confirm this specific aiohttp/SOCKS session before entering long polling.
    # A short timeout prevents a worker that looks alive but never subscribes
    # to Telegram updates.
    while True:
        try:
            profile = await bot.get_me(request_timeout=15)
            bot._me = profile
            logger.info(f"Telegram connection ready: @{profile.username}")
            break
        except (TelegramNetworkError, asyncio.TimeoutError):
            logger.exception("Telegram connection failed; retrying in 5 seconds")
            await asyncio.sleep(5)
    # fic_comp occasionally times out while opening Telegram's HTTPS endpoint.
    # Keep the worker alive and retry transient network failures instead of
    # requiring a person to SSH in and restart it.
    try:
        while True:
            try:
                await dispatcher.start_polling(bot, close_bot_session=False)
                # A normal return means a shutdown was requested; do not start a
                # new polling loop after SIGTERM.
                break
            except (TelegramNetworkError, asyncio.TimeoutError):
                logger.exception("Telegram connection failed; retrying in 5 seconds")
                await asyncio.sleep(5)
    finally:
        await dashboard.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
