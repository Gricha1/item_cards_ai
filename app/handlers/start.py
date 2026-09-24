from pathlib import Path

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message

router = Router()
WELCOME_IMAGE = Path(__file__).resolve().parent.parent / "assets" / "welcome" / "polina_welcome.png"


def welcome_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✨ Создать модель", callback_data="welcome:create"),
        InlineKeyboardButton(text="ℹ️ Как это работает", callback_data="welcome:help"),
    ]])


@router.message(CommandStart())
async def start(message: Message) -> None:
    caption = "Привет, раб. Я госпожа Полина, и я могу тебе сгенерировать модель для карточек. Выбери опцию:"
    if WELCOME_IMAGE.exists():
        await message.answer_photo(
            FSInputFile(WELCOME_IMAGE), caption=caption, reply_markup=welcome_keyboard()
        )
        return
    await message.answer(caption, reply_markup=welcome_keyboard())


@router.callback_query(F.data == "welcome:create")
async def begin_generation(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.answer("Пришлите одно фото одежды — я создам 2 варианта изображения.")


@router.callback_query(F.data == "welcome:help")
async def show_help(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.answer(
        "Отправьте фото одежды, затем выберите пол модели и кадр. "
        "Генерация двух вариантов обычно занимает несколько минут."
    )
