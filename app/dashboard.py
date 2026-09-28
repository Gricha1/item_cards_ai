from __future__ import annotations

from html import escape

from aiohttp import web

from app.services.analytics import AnalyticsStore


EVENT_LABELS = {
    "start": "Открыл бота (/start)",
    "create_clicked": "Нажал «Создать модель»",
    "photo_received": "Прислал фото вещи",
    "garment_type_selected": "Выбрал тип фото вещи",
    "gender_selected": "Выбрал пол модели",
    "age_selected": "Выбрал возраст",
    "framing_selected": "Выбрал кадр",
    "generation_started": "Запустил генерацию",
    "generation_completed": "Получил изображения",
    "generation_failed": "Генерация завершилась ошибкой",
    "retry_clicked": "Запустил перегенерацию",
    "details_requested": "Открыл добавление деталей",
    "new_item_clicked": "Начал другую вещь",
}


def _label(event_type: str) -> str:
    return EVENT_LABELS.get(event_type, event_type)


def _cell(value: object) -> str:
    return escape("" if value is None else str(value))


def _user_name(row) -> str:
    username = f"@{row['username']}" if row["username"] else "—"
    return f"{_cell(row['display_name'])}<br><small>{_cell(username)}</small>"


async def dashboard_page(request: web.Request) -> web.Response:
    token = request.app["token"]
    if token and request.query.get("token") != token:
        raise web.HTTPUnauthorized(text="Dashboard token is required.")

    snapshot = request.app["store"].snapshot()
    totals = snapshot["totals"]
    users_rows = "".join(
        "<tr>"
        f"<td>{_cell(row['telegram_id'])}</td>"
        f"<td>{_user_name(row)}</td>"
        f"<td>{_cell(row['first_seen'])}</td>"
        f"<td>{_cell(row['last_seen'])}</td>"
        f"<td>{_cell(row['starts_count'])}</td>"
        f"<td>{_cell(row['generations_count'])}</td>"
        f"<td>{_cell(_label(row['last_action']))}</td>"
        "</tr>"
        for row in snapshot["users"]
    ) or "<tr><td colspan='7'>Пока нет действий пользователей.</td></tr>"
    events_rows = "".join(
        "<tr>"
        f"<td>{_cell(row['created_at'])}</td>"
        f"<td>{_cell(row['telegram_id'])}</td>"
        f"<td>{_cell(_label(row['event_type']))}</td>"
        f"<td>{_cell(row['detail'] or '—')}</td>"
        "</tr>"
        for row in snapshot["events"]
    ) or "<tr><td colspan='4'>Пока нет событий.</td></tr>"
    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta http-equiv="refresh" content="20">
<title>ItemCards AI — пользователи</title>
<style>
body {{ font: 15px/1.4 system-ui,sans-serif; color:#17212b; margin:32px; background:#f7f9fc }}
h1 {{ margin-bottom:6px }} .hint {{ color:#667085; margin-top:0 }}
.cards {{ display:flex; gap:16px; margin:22px 0 }} .card {{ background:#fff; border-radius:12px; padding:16px 22px; min-width:160px; box-shadow:0 1px 3px #0001 }}
.number {{ font-size:28px; font-weight:700 }} table {{ width:100%; border-collapse:collapse; background:#fff; margin:12px 0 30px; box-shadow:0 1px 3px #0001 }}
th,td {{ padding:10px 12px; border-bottom:1px solid #e8edf3; text-align:left; vertical-align:top }} th {{ background:#edf4ff; white-space:nowrap }} small {{ color:#667085 }}
</style></head><body>
<h1>ItemCards AI — активность</h1><p class="hint">Обновляется каждые 20 секунд. Сообщения и изображения не сохраняются.</p>
<div class="cards"><div class="card">Пользователи<div class="number">{totals['users']}</div></div><div class="card">Успешные генерации<div class="number">{totals['generations']}</div></div></div>
<h2>Пользователи</h2><table><thead><tr><th>Telegram ID</th><th>Пользователь</th><th>Первый вход</th><th>Последняя активность</th><th>/start</th><th>Генерации</th><th>Последнее действие</th></tr></thead><tbody>{users_rows}</tbody></table>
<h2>Последние 100 действий</h2><table><thead><tr><th>Время (UTC)</th><th>Telegram ID</th><th>Действие</th><th>Детали</th></tr></thead><tbody>{events_rows}</tbody></table>
</body></html>"""
    return web.Response(text=html, content_type="text/html")


async def start_dashboard(store: AnalyticsStore, host: str, port: int, token: str | None) -> web.AppRunner:
    app = web.Application()
    app["store"] = store
    app["token"] = token
    app.router.add_get("/", dashboard_page)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    return runner
