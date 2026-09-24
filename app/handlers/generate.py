from __future__ import annotations

import asyncio
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
    choosing_framing = State()


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


def make_router(settings: Settings) -> Router:
    configured = Router()
    pipeline = GenerationPipeline(settings)

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
        await state.set_state(GenerationState.choosing_framing)
        await callback.message.edit_text("Выберите кадр:", reply_markup=framing_keyboard())
        await callback.answer()

    @configured.callback_query(GenerationState.choosing_framing, F.data.startswith("frame:"))
    async def choose_framing(callback: CallbackQuery, state: FSMContext) -> None:
        framing = callback.data.split(":", 1)[1]
        data = await state.get_data()
        await callback.answer()
        await callback.message.edit_text("Генерация началась. Подготавливаю фото и модель…")
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
                    framing,
                    report_progress,
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
                    await callback.message.edit_text(
                        f"Вариант {variant_number} из 2: {percent}% ({current_step}/{total_steps})."
                    )
                    reported_bucket = bucket

            variants = await generation_task
            for variant in variants:
                await callback.message.answer_photo(FSInputFile(variant.path), caption=variant.label)
        except TemplateNotFoundError as error:
            await callback.message.answer(str(error))
        except Exception:
            logger.exception("Generation failed")
            await callback.message.answer("Не удалось создать изображения. Проверьте шаблоны, веса моделей и свободную VRAM.")
        finally:
            await state.clear()

    @configured.message(F.document)
    async def receive_document(message: Message) -> None:
        await message.answer("Отправьте фото как изображение, а не как файл: тогда Telegram передаст корректный preview.")

    return configured
