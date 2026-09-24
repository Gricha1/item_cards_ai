from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

router = Router()


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Привет! Пришлите одно фото одежды. Затем выберите пол модели и кадр — "
        "я создам 2 варианта изображения."
    )
