# Item Cards AI

Telegram MVP для создания двух вариантов товарной фотографии одежды на виртуальной модели. Бот принимает фото изделия, пол и кадр, затем запускает локальный FASHN VTON v1.5 и второй backend.

## Модели и ограничения

* **Вариант 1 — FASHN VTON v1.5**: локальный virtual try-on с person-template и фото одежды.
* **Вариант 2 — Qwen-Image-Edit**: локальный image edit, если включён и GPU имеет не менее 20 ГБ VRAM.
* **Fallback — FLUX.1-schnell**: запускается с CPU offload, если Qwen недоступен. Это только слабый image-to-image fallback и он не гарантирует переноса точных деталей одежды.

FASHN VTON v1.5, Qwen-Image-Edit и FLUX.1-schnell опубликованы с Apache-2.0. «Бесплатно» означает отсутствие платы за API: нужны собственные GPU, интернет и место для весов. Проверьте лицензии и правила маркетплейсов перед коммерческим запуском.

## Развёртывание на fic_comp

```bash
git clone https://github.com/Gricha1/item_cards_ai.git ~/item_cards_ai
cd ~/item_cards_ai
bash scripts/setup_server.sh
cp .env.example .env
chmod 600 .env
nano .env  # добавьте TELEGRAM_BOT_TOKEN только на сервере
bash scripts/run_bot.sh
```

`setup_server.sh` проверяет GPU, создаёт virtualenv, ставит зависимости и скачивает FASHN-веса. Qwen не скачивается, пока не включён явно. Для фонового процесса используйте:

```bash
tmux new -s itemcards 'bash scripts/run_bot.sh'
```

## Как пользоваться

1. Откройте бота и отправьте `/start`.
2. Пришлите одно фото одежды (JPG/PNG как Telegram photo).
3. Нажмите «Мужчина» или «Женщина», затем «В полный рост» или «По пояс».
4. Дождитесь двух изображений с подписями backend’ов.

Фото и результаты хранятся в `data/`, но исключены из Git. Шаблоны людей кладутся в `app/assets/templates/`; для реальных каталожных результатов замените их на собственные изображения моделей с подтверждёнными правами использования.

## Конфигурация

Шаблон находится в [.env.example](.env.example). Никогда не добавляйте `.env`, токен Telegram или ключи Hugging Face в Git. Для Qwen требуются существенно большие VRAM и дисковое пространство: его модельный репозиторий составляет около 57,7 ГБ.

## Проверка

```bash
pytest -q
python scripts/detect_gpu.py
```

Тесты не скачивают веса и не отправляют сообщения в Telegram.
