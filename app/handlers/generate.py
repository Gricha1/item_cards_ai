from __future__ import annotations

import asyncio
import secrets
from pathlib import Path

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message
from loguru import logger

from app.config import Settings
from app.services.pipeline import GenerationPipeline, TemplateNotFoundError
from app.utils.image_io import ImageValidationError, validate_and_normalize
from app.utils.temp_files import unique_path

router = Router()


class GenerationState(StatesGroup):
    choosing_gender = State()
    choosing_age = State()
    choosing_framing = State()
    ready = State()
    adding_details = State()


def gender_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Мужчина", callback_data="gender:male"),
        InlineKeyboardButton(text="Женщина", callback_data="gender:female"),
    ]])


def framing_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="В полный рост", callback_data="frame:full"),
        InlineKeyboardButton(text="По пояс", callback_data="frame:waist"),
    ]])


def age_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="18–25", callback_data="age:18-25"),
            InlineKeyboardButton(text="26–35", callback_data="age:26-35"),
        ],
        [
            InlineKeyboardButton(text="36–45", callback_data="age:36-45"),
            InlineKeyboardButton(text="46+", callback_data="age:46-plus"),
        ],
    ])


def result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Перегенерировать", callback_data="result:retry")],
        [InlineKeyboardButton(text="✍️ Добавить детали", callback_data="result:details")],
        [InlineKeyboardButton(text="🧥 Другая вещь", callback_data="result:new-item")],
    ])


def make_router(settings: Settings) -> Router:
    configured = Router()
    pipeline = GenerationPipeline(settings)

    async def generate_and_send(message: Message, state: FSMContext, data: dict) -> None:
        await message.edit_text("Генерация началась. Подготавливаю фото и модель…")
        try:
            loop = asyncio.get_running_loop()
            progress_queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()

            def report_progress(variant: str, current_step: int, total_steps: int) -> None:
                loop.call_soon_threadsafe(progress_queue.put_nowait, (variant, current_step, total_steps))

            generation_task = asyncio.create_task(
                asyncio.to_thread(
                    pipeline.generate,
                    Path(data["garment_path"]),
                    data["gender"],
                    data["framing"],
                    report_progress,
                    age_range=data["age_range"],
                    seed=secrets.randbelow(2_000_000_000) + 1,
                )
            )
            reported_bucket = -1
            active_variant = ""
            while not generation_task.done():
                try:
                    variant, current_step, total_steps = await asyncio.wait_for(progress_queue.get(), timeout=1)
                except TimeoutError:
                    continue
                if variant != active_variant:
                    active_variant = variant
                    reported_bucket = -1
                percent = round(current_step * 100 / total_steps)
                bucket = percent // 10
                if bucket > reported_bucket or percent == 100:
                    variant_number = "1" if variant == "first" else "2"
                    await message.edit_text(
                        f"Вариант {variant_number} из 2: {percent}% ({current_step}/{total_steps})."
                    )
                    reported_bucket = bucket

            variants = await generation_task
            for variant in variants:
                await message.answer_photo(FSInputFile(variant.path), caption=variant.label)
            await state.set_state(GenerationState.ready)
            await message.answer("Что сделать с результатом?", reply_markup=result_keyboard())
        except TemplateNotFoundError as error:
            await message.answer(str(error))
        except Exception:
            logger.exception("Generation failed")
            await message.answer("Не удалось создать изображения. Проверьте шаблоны, веса моделей и свободную VRAM.")

    @configured.message(F.photo)
    async def receive_photo(message: Message, state: FSMContext) -> None:
        photo = message.photo[-1]
        raw_path = unique_path(settings.temp_dir, ".jpg")
        normalized_path = unique_path(settings.input_dir)
        try:
            await message.bot.download(photo, destination=raw_path)
            validate_and_normalize(raw_path, normalized_path)
        except ImageValidationError as error:
            raw_path.unlink(missing_ok=True)
            normalized_path.unlink(missing_ok=True)
            await message.answer(str(error))
            return
        finally:
            raw_path.unlink(missing_ok=True)

        await state.clear()
        await state.update_data(garment_path=str(normalized_path))
        await state.set_state(GenerationState.choosing_gender)
        await message.answer("Выберите пол виртуальной модели:", reply_markup=gender_keyboard())

    @configured.callback_query(GenerationState.choosing_gender, F.data.startswith("gender:"))
    async def choose_gender(callback: CallbackQuery, state: FSMContext) -> None:
        gender = callback.data.split(":", 1)[1]
        await state.update_data(gender=gender)
        await state.set_state(GenerationState.choosing_age)
        await callback.message.edit_text("Выберите возраст модели (только 18+):", reply_markup=age_keyboard())
        await callback.answer()

    @configured.callback_query(GenerationState.choosing_age, F.data.startswith("age:"))
    async def choose_age(callback: CallbackQuery, state: FSMContext) -> None:
        age_range = callback.data.split(":", 1)[1]
        await state.update_data(age_range=age_range)
        await state.set_state(GenerationState.choosing_framing)
        await callback.message.edit_text("Выберите кадр:", reply_markup=framing_keyboard())
        await callback.answer()

    @configured.callback_query(GenerationState.choosing_framing, F.data.startswith("frame:"))
    async def choose_framing(callback: CallbackQuery, state: FSMContext) -> None:
        framing = callback.data.split(":", 1)[1]
        await state.update_data(framing=framing)
        data = await state.get_data()
        await callback.answer()
        await generate_and_send(callback.message, state, data)

    @configured.callback_query(GenerationState.ready, F.data == "result:retry")
    async def retry_generation(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer("Запускаю новую вариацию")
        await generate_and_send(callback.message, state, await state.get_data())

    @configured.callback_query(GenerationState.ready, F.data == "result:new-item")
    async def start_new_item(callback: CallbackQuery, state: FSMContext) -> None:
        await state.clear()
        await callback.answer()
        await callback.message.edit_text("Пришлите фото другой вещи как изображение.")

    @configured.callback_query(GenerationState.ready, F.data == "result:details")
    async def request_details(callback: CallbackQuery, state: FSMContext) -> None:
        await state.set_state(GenerationState.adding_details)
        await callback.answer()
        await callback.message.edit_text("Опишите, что добавить или изменить в текущей фотографии.")

    @configured.message(GenerationState.adding_details, F.text)
    async def save_details(message: Message, state: FSMContext) -> None:
        await state.update_data(edit_instruction=message.text)
        await state.set_state(GenerationState.ready)
        await message.answer(
            "Описание сохранено. Точная дорисовка по тексту будет доступна после подключения Qwen на GPU 24 GB; "
            "FASHN на fic_comp умеет только замену одежды, поэтому не буду имитировать применение текста."
        )
        await message.answer("Что сделать с результатом?", reply_markup=result_keyboard())

    @configured.message(F.document)
    async def receive_document(message: Message) -> None:
        await message.answer("Отправьте фото как изображение, а не как файл: тогда Telegram передаст корректный preview.")

    return configured
