from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

router = Router()


def welcome_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✨ Создать модель", callback_data="welcome:create"),
    ]])


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Привет! Я помогу создать изображения модели для карточки товара. Выберите опцию:",
        reply_markup=welcome_keyboard(),
    )


@router.callback_query(F.data == "welcome:create")
async def begin_generation(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.answer("Пришлите одно фото одежды — я создам 2 варианта изображения.")

