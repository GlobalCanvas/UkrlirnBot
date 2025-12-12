#!/usr/bin/env python3
# bot.py - UkrLirn Monitor Bot (Fixed comparison + no canvaspix)

import sys
import logging
import asyncio
import json
import os
import io
from io import BytesIO
from datetime import datetime

# --- ПЕРЕВІРКА БІБЛІОТЕК ---
try:
    import numpy as np
    import PIL.Image
    import aiohttp
    from telegram import Update
    from telegram.ext import (
        ApplicationBuilder, CommandHandler, MessageHandler,
        ConversationHandler, ContextTypes, filters
    )
except ImportError as e:
    print("="*60)
    print("❌ ПОМИЛКА: Не встановлено бібліотеки!")
    print(f"Не знайдено: {e.name}")
    print("Виконай команду:")
    print("pip install python-telegram-bot aiohttp Pillow numpy")
    print("="*60)
    sys.exit(1)

# --- ЛОГУВАННЯ ---
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ======================================================================
# ⚙️ КОНФІГУРАЦІЯ
# ======================================================================

# ТОКЕН БОТА
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8133267244:AAFPj7GcUhgUPUiuAxM9afwQFoSsB5hEtUc")

# АВТОРИЗАЦІЯ (КРИТИЧНО ДЛЯ ФІКСА 401!)
AUTH_COOKIE = os.environ.get("AUTH_COOKIE", "s%3AS2qBqqlzYPCWST-OalOz6svoEoTYQIi9.%2BL0JZVKMRNrHr9eQ8WAuf4D9MdthKJP3pHCrqliUmZs")

API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Cookie': f'cpn.session={AUTH_COOKIE}; plang=ru',
    'Accept': 'application/json'
}

COLOR_TOLERANCE = 20
STATE_FILE = "state.json"
FACTION_API_URL = "https://pixmap.fun/api/faction/my-faction"
STRICT_TILE_SIZE = 256  # Фіксований розмір тайлів

# ======================================================================
# 🌐 САЙТИ (ВИДАЛЕНО CANVASPIX!)
# ======================================================================

SITES = {
    "pixmap": {
        "url": "https://pixmap.fun",
        "tile_url": "https://pixmap.fun/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://pixmap.fun/api/me",
        "api_faction": "https://pixmap.fun/api/faction/my-faction"
    },
    "pixelya": {
        "url": "https://pixelya.fun",
        "tile_url": "https://pixelya.fun/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://pixelya.fun/api/me",
        "api_faction": "https://pixelya.fun/api/faction/my-faction"
    },
    "globepixel": {
        "url": "https://globepixel.net",
        "tile_url": "https://globepixel.net/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://globepixel.net/api/me",
        "api_faction": "https://globepixel.net/api/faction/my-faction"
    }
}

CURRENT_SITE = "pixmap"

def set_site(site_name: str) -> bool:
    global CURRENT_SITE, FACTION_API_URL
    if site_name.lower() in SITES:
        CURRENT_SITE = site_name.lower()
        FACTION_API_URL = SITES[CURRENT_SITE]["api_faction"]
        return True
    return False

def get_current_site():
    return SITES[CURRENT_SITE]

# ======================================================================
# 🎨 ДВИЖОК (256px тайли) - FIXED COMPARISON
# ======================================================================

async def fetch_api_me():
    site = get_current_site()
    url = site["api_me"]
    
    async with aiohttp.ClientSession() as session:
        for _ in range(3):
            try:
                async with session.get(url, headers=API_HEADERS, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    logger.warning(f"API me: {resp.status}")
            except Exception as e:
                logger.warning(f"Помилка API: {e}")
                await asyncio.sleep(2)
    return None


async def fetch_tile(session, url, offx, offy, image, needed=False, debug_save=False, tx=0, ty=0):
    """Завантажує тайл і приводить до 256x256"""
    for _ in range(3):
        try:
            async with session.get(url, headers=API_HEADERS, timeout=15) as resp:
                if resp.status == 404:
                    if needed:
                        empty = PIL.Image.new('RGBA', (STRICT_TILE_SIZE, STRICT_TILE_SIZE), (0, 0, 0, 0))
                        image.paste(empty, (offx, offy))
                    if debug_save:
                        logger.info(f"  ❌ 404: tile_{tx}_{ty}")
                    return True
                
                if resp.status == 200:
                    data = await resp.read()
                    if data:
                        tile = PIL.Image.open(io.BytesIO(data)).convert('RGBA')
                        
                        # Приводимо до 256x256
                        if tile.size != (STRICT_TILE_SIZE, STRICT_TILE_SIZE):
                            tile = tile.resize((STRICT_TILE_SIZE, STRICT_TILE_SIZE), PIL.Image.NEAREST)
                        
                        # Debug save
                        if debug_save:
                            os.makedirs("debug/tiles", exist_ok=True)
                            tile.save(f"debug/tiles/tile_{tx}_{ty}.png")
                            logger.info(f"  ✅ Збережено: tile_{tx}_{ty}.png")
                        
                        image.paste(tile, (offx, offy), tile)
                        return True
        except Exception as e:
            logger.warning(f"Помилка тайлу: {e}")
            await asyncio.sleep(0.5)
    return False


async def get_canvas_area(canvas_id, x, y, width, height, canvas_size=32768, debug_save=False):
    """
    Завантажує область з канвасу (як в historyDownload.py)
    
    Відмінності від старого методу:
    - Використовує правильний offset: -canvas_size/2
    - Працює з 256px тайлами
    - Враховує що координати можуть бути негативними
    """
    site = get_current_site()
    offset = int(-canvas_size / 2)
    tile_size = STRICT_TILE_SIZE
    
    # Обчислюємо діапазон тайлів (як в historyDownload.py)
    xc = (x - offset) // tile_size
    wc = (x + width - offset) // tile_size
    yc = (y - offset) // tile_size
    hc = (y + height - offset) // tile_size
    
    logger.info(f"📐 Область: x={x}, y={y}, розмір={width}x{height}")
    logger.info(f"🗺️ Тайли (256px): X[{xc}..{wc}], Y[{yc}..{hc}]")
    logger.info(f"📦 Всього: {(wc - xc + 1) * (hc - yc + 1)} шт")
    
    result = PIL.Image.new('RGBA', (width, height), (0, 0, 0, 0))
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        for iy in range(yc, hc + 1):
            for ix in range(xc, wc + 1):
                url = site["tile_url"].format(canvas_id=canvas_id, zoom=7, tx=ix, ty=iy)
                
                # Розраховуємо offset для paste (як в historyDownload.py)
                offx = ix * tile_size + offset - x
                offy = iy * tile_size + offset - y
                
                if debug_save:
                    logger.info(f"Тайл [{ix},{iy}]: offx={offx}, offy={offy}")
                
                tasks.append(fetch_tile(session, url, offx, offy, result, needed=True, debug_save=debug_save, tx=ix, ty=iy))
        
        results = await asyncio.gather(*tasks)
        loaded = sum(1 for r in results if r)
        logger.info(f"✅ Завантажено: {loaded}/{len(tasks)}")
    
    if debug_save:
        os.makedirs("debug", exist_ok=True)
        result.save("debug/board_full.png")
        logger.info("💾 Склеєна доска: debug/board_full.png")
    
    return result


def compare_with_template(template, board, tolerance=20):
    """
    FIXED: Правильне порівняння піксель-за-пікселем
    
    Логіка (як порадив VKLShadow):
    1. Порівнюємо кожен піксель шаблону з дошкою
    2. Враховуємо тільки непрозорі пікселі шаблону (alpha > 10)
    3. Перевіряємо чи піксель на дошці співпадає з шаблоном (в межах tolerance)
    """
    tw, th = template.size
    
    # Обрізаємо board до розміру template
    if board.size != (tw, th):
        board = board.crop((0, 0, tw, th))
    
    # Конвертуємо в numpy для швидкого порівняння
    t_array = np.array(template, dtype=np.float32)
    b_array = np.array(board, dtype=np.float32)
    
    # Маска непрозорих пікселів шаблону
    template_mask = t_array[..., 3] > 10
    total_pixels = int(template_mask.sum())
    
    if total_pixels == 0:
        logger.warning("⚠️ Шаблон порожній (всі пікселі прозорі)!")
        return {"total": 0, "placed": 0, "remaining": 0, "percent": 100.0}
    
    # Порівнюємо RGB канали (ігноруємо alpha)
    # Обчислюємо евклідову відстань між кольорами
    diff = np.sqrt(np.sum((b_array[..., :3] - t_array[..., :3]) ** 2, axis=-1))
    
    # Перевіряємо чи піксель в межах tolerance
    color_match = diff <= tolerance
    
    # Перевіряємо чи піксель на дошці непрозорий
    board_mask = b_array[..., 3] > 10
    
    # Правильно розміщені = непрозорі в шаблоні + непрозорі на дошці + колір співпадає
    placed_pixels = int((template_mask & board_mask & color_match).sum())
    remaining_pixels = total_pixels - placed_pixels
    percent = (placed_pixels / total_pixels * 100.0) if total_pixels > 0 else 100.0
    
    logger.info(f"✅ Правильно: {placed_pixels}/{total_pixels} ({percent:.2f}%)")
    logger.info(f"❌ Залишилось: {remaining_pixels}")
    
    return {
        "total": total_pixels,
        "placed": placed_pixels,
        "remaining": remaining_pixels,
        "percent": percent
    }


def create_overlay(template, board, tolerance=20, output_path=None):
    """
    Створює overlay зображення:
    - Червоним позначає неправильні/відсутні пікселі
    - Залишає правильні пікселі як є
    """
    if not output_path:
        return None
        
    tw, th = template.size
    if board.size != (tw, th):
        board = board.crop((0, 0, tw, th))
    
    t_array = np.array(template, dtype=np.float32)
    b_array = np.array(board, dtype=np.float32)
    
    template_mask = t_array[..., 3] > 10
    board_mask = b_array[..., 3] > 10
    diff = np.sqrt(np.sum((b_array[..., :3] - t_array[..., :3]) ** 2, axis=-1))
    color_match = diff <= tolerance
    
    # Створюємо output (копія board)
    output = b_array.copy().astype(np.uint8)
    
    # Неправильні = (є в шаблоні) і (колір не співпадає АБО відсутній на дошці)
    bad_pixels = template_mask & (~color_match | ~board_mask)
    output[bad_pixels] = [255, 0, 0, 255]  # Червоний
    
    PIL.Image.fromarray(output.astype(np.uint8)).save(output_path)
    logger.info(f"💾 Overlay збережено: {output_path}")
    return output_path


async def process_lirn_template(template_path, x, y, canvas_id=0, tolerance=20, overlay_path=None, debug_mode=False):
    """Головна функція обробки шаблону"""
    template = PIL.Image.open(template_path).convert("RGBA")
    width, height = template.size
    logger.info(f"📐 Шаблон: {width}x{height} px")
    
    # Отримуємо розмір канвасу з API
    api_me = await fetch_api_me()
    canvas_size = 32768  # За замовчуванням
    
    if api_me and 'canvases' in api_me:
        canvas_info = api_me['canvases'].get(str(canvas_id))
        if canvas_info:
            canvas_size = canvas_info.get('size', 32768)
            logger.info(f"📏 Розмір канвасу: {canvas_size}x{canvas_size}")
    
    # Завантажуємо область з дошки
    board = await get_canvas_area(canvas_id, x, y, width, height, canvas_size, debug_save=debug_mode)
    
    # Порівнюємо
    result = compare_with_template(template, board, tolerance)
    
    # Створюємо overlay
    if overlay_path:
        create_overlay(template, board, tolerance, overlay_path)
    
    return result

# ======================================================================
# 🤖 БОТ
# ======================================================================

UPLOAD_TEMPLATE_WAITING = 1
LIRN_TEMPLATE = {"file": "templates/lirn.png"}
state = {
    "user_links": {},
    "medals": {},
    "lirn_coords": [0, 0],
    "current_site": "pixmap"
}

def load_state():
    global state
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state.update(json.load(f))
                set_site(state.get("current_site", "pixmap"))
                logger.info(f"📂 Стан завантажено. Сайт: {state['current_site']}")
        except Exception as e:
            logger.error(f"Помилка завантаження state.json: {e}")

def save_state():
    try:
        state["current_site"] = CURRENT_SITE
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Помилка збереження state.json: {e}")


# --- КОМАНДИ ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎨 **UkrLirn Monitor Bot** (FIXED)\n\n"
        "**Шаблон:**\n"
        "• `/upload_template` — завантажити PNG\n"
        "• `/set_coords X Y` — встановити координати\n"
        "• `/check` — прогрес малювання\n"
        "• `/test_check` — тест з дебагом\n\n"
        "**Гравці:**\n"
        "• `/connect <нік>` — прив'язати профіль\n"
        "• `/profile [нік]` — профіль гравця\n"
        "• `/list` — список фракції\n\n"
        "**Медалі (адмін):**\n"
        "• `/madd <назва> <1-10>` — додати (у відповідь)\n"
        "• `/mdel <номер>` — видалити (у відповідь)\n\n"
        "**Налаштування:**\n"
        "• `/site_change <сайт>` — змінити (pixmap/pixelya/globepixel)\n"
        "• `/status` — поточні налаштування",
        parse_mode="Markdown"
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показує поточні налаштування"""
    x, y = state.get("lirn_coords", [0, 0])
    template_exists = os.path.exists(LIRN_TEMPLATE["file"])
    
    template_info = "✅ Завантажено" if template_exists else "❌ Відсутній"
    coords_info = f"✅ ({x}, {y})" if [x, y] != [0, 0] else "❌ Не встановлено"
    
    await update.message.reply_text(
        f"⚙️ **Поточні налаштування:**\n\n"
        f"🌐 Сайт: `{CURRENT_SITE}`\n"
        f"📐 Шаблон: {template_info}\n"
        f"📍 Координати: {coords_info}\n"
        f"🎨 Толеранс: `{COLOR_TOLERANCE}`",
        parse_mode="Markdown"
    )


async def set_coords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) != 2:
        await update.message.reply_text(
            "⚠️ **Формат:** `/set_coords X Y`\n\n"
            "**Приклад:** `/set_coords 4031 -11628`\n\n"
            "_💡 Підказка: використовуй R на сайті для копіювання координат_",
            parse_mode="Markdown"
        )
        return
    
    try:
        x, y = int(context.args[0]), int(context.args[1])
        state["lirn_coords"] = [x, y]
        save_state()
        await update.message.reply_text(
            f"✅ **Координати встановлено!**\n\n"
            f"📍 X: `{x}`\n"
            f"📍 Y: `{y}`",
            parse_mode="Markdown"
        )
    except ValueError:
        await update.message.reply_text("❌ Координати мають бути числами!")


async def upload_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📤 **Завантаження шаблону**\n\n"
        "Надішли PNG файл шаблону.\n\n"
        "_💡 Для скасування: /cancel_"
    )
    return UPLOAD_TEMPLATE_WAITING


async def upload_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith('.png'):
        await update.message.reply_text("❌ Потрібен PNG файл!")
        return ConversationHandler.END
    
    file = await doc.get_file()
    os.makedirs("templates", exist_ok=True)
    await file.download_to_drive(LIRN_TEMPLATE["file"])
    
    img = PIL.Image.open(LIRN_TEMPLATE["file"])
    await update.message.reply_text(
        f"✅ **Шаблон завантажено!**\n\n"
        f"📐 Розмір: `{img.size[0]}x{img.size[1]}` px\n\n"
        f"Тепер встанови координати: `/set_coords X Y`",
        parse_mode="Markdown"
    )
    return ConversationHandler.END


async def cancel_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Завантаження скасовано.")
    return ConversationHandler.END


async def check_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Перевіряє прогрес малювання"""
    if not os.path.exists(LIRN_TEMPLATE["file"]):
        await update.message.reply_text(
            "❌ **Шаблон відсутній!**\n\n"
            "Завантаж його: `/upload_template`",
            parse_mode="Markdown"
        )
        return
    
    x, y = state.get("lirn_coords", [0, 0])
    if [x, y] == [0, 0]:
        await update.message.reply_text(
            "❌ **Координати не встановлено!**\n\n"
            "Встанови їх: `/set_coords X Y`",
            parse_mode="Markdown"
        )
        return
    
    status_msg = await update.message.reply_text(
        f"⏳ **Перевіряю прогрес...**\n\n"
        f"📍 Координати: ({x}, {y})\n"
        f"🌐 Сайт: {CURRENT_SITE}"
    )
    
    try:
        os.makedirs("progress", exist_ok=True)
        res = await process_lirn_template(
            LIRN_TEMPLATE["file"], x, y,
            tolerance=COLOR_TOLERANCE,
            overlay_path="progress/overlay.png"
        )
        
        caption = (
            f"📊 **Прогрес UkrLirn**\n\n"
            f"🎯 Всього пікселів: `{res['total']:,}`\n"
            f"✅ Правильно: `{res['placed']:,}`\n"
            f"❌ Залишилось: `{res['remaining']:,}`\n\n"
            f"📈 **Готовність: {res['percent']:.2f}%**\n\n"
            f"🌐 Сайт: {CURRENT_SITE}\n"
            f"📍 Координати: ({x}, {y})"
        )
        
        with open("progress/overlay.png", "rb") as f:
            await update.message.reply_document(
                document=f,
                caption=caption,
                parse_mode="Markdown",
                filename="progress.png"
            )
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"Помилка check: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ **Помилка:** `{str(e)}`", parse_mode="Markdown")


async def test_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ТЕСТОВИЙ режим з збереженням тайлів"""
    if not os.path.exists(LIRN_TEMPLATE["file"]):
        await update.message.reply_text("❌ Спочатку завантаж шаблон: `/upload_template`", parse_mode="Markdown")
        return
    
    x, y = state.get("lirn_coords", [0, 0])
    status_msg = await update.message.reply_text(
        f"🔍 **ТЕСТОВИЙ РЕЖИМ**\n\n"
        f"Завантажую тайли та зберігаю в `debug/tiles/`...\n"
        f"📍 Координати: ({x}, {y})",
        parse_mode="Markdown"
    )
    
    try:
        res = await process_lirn_template(
            LIRN_TEMPLATE["file"], x, y,
            tolerance=COLOR_TOLERANCE,
            overlay_path="debug/test_overlay.png",
            debug_mode=True
        )
        
        msg = (
            f"🔍 **Тестові результати**\n\n"
            f"📂 Тайли: `debug/tiles/`\n"
            f"📄 Склеєна доска: `debug/board_full.png`\n\n"
            f"📊 **Прогрес:**\n"
            f"🎯 Всього: `{res['total']:,}` px\n"
            f"✅ Готово: `{res['placed']:,}` px\n"
            f"❌ Залишилось: `{res['remaining']:,}` px\n"
            f"📈 **{res['percent']:.2f}%**"
        )
        
        # Відправляємо склеєну доску
        if os.path.exists("debug/board_full.png"):
            with open("debug/board_full.png", "rb") as f:
                await update.message.reply_document(document=f, caption="📄 Склеєна доска")
        
        # Відправляємо overlay
        if os.path.exists("debug/test_overlay.png"):
            with open("debug/test_overlay.png", "rb") as f:
                await update.message.reply_document(document=f, caption=msg, parse_mode="Markdown")
        
        await status_msg.delete()
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ **Помилка:** `{str(e)}`", parse_mode="Markdown")


async def connect_player(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "⚠️ **Формат:** `/connect <нік>`\n\n"
            "**Приклад:** `/connect Puwo`",
            parse_mode="Markdown"
        )
        return
    
    nick = " ".join(context.args)
    state["user_links"][str(update.effective_user.id)] = nick
    save_state()
    await update.message.reply_text(
        f"✅ **Профіль прив'язано!**\n\n"
        f"👤 Нік: **{nick}**\n\n"
        f"Тепер можеш використовувати `/profile`",
        parse_mode="Markdown"
    )


async def get_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    
    if context.args:
        nick = " ".join(context.args)
        target_id = None
        for uid, n in state["user_links"].items():
            if n.lower() == nick.lower():
                target_id = uid
                break
    else:
        nick = state["user_links"].get(user_id)
        target_id = user_id
        
        if not nick:
            await update.message.reply_text(
                "⚠️ **Прив'яжи профіль!**\n\n"
                "Використай: `/connect <нік>`",
                parse_mode="Markdown"
            )
            return
    
    msg = await update.message.reply_text("🔍 Завантажую дані фракції...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(FACTION_API_URL, headers=API_HEADERS, timeout=10) as resp:
                if resp.status == 401:
                    await msg.edit_text("❌ **Помилка 401:** Оновіть AUTH_COOKIE!")
                    return
                if resp.status != 200:
                    await msg.edit_text(f"❌ **Помилка API:** {resp.status}")
                    return
                data = await resp.json()
    except Exception as e:
        await msg.edit_text(f"❌ **Помилка:** `{e}`", parse_mode="Markdown")
        return

    faction = data.get("faction", data)
    found = None
    for member in faction.get("members", []):
        if member.get("User", {}).get("name", "").lower() == nick.lower():
            found = member
            break
    
    if not found:
        await msg.edit_text(f"❌ Гравець **{nick}** не знайдений у фракції.", parse_mode="Markdown")
        return
    
    u = found["User"]
    pixels = u.get("totalPixels", 0)
    status = "✅ Активний" if found.get("isActive") else "💤 Неактивний"
    
    joined = ""
    if found.get("joinedAt"):
        try:
            dt = datetime.fromisoformat(found["joinedAt"].replace("Z", "+00:00"))
            joined = dt.strftime("%d.%m.%Y")
        except:
            pass
    
    medals_text = ""
    if target_id and target_id in state["medals"]:
        medals_text = "\n\n🏅 **Медалі:**\n"
        for i, m in enumerate(state["medals"][target_id], 1):
            stars = "⭐" * m["weight"]
            medals_text += f"{i}. {m['name']} {stars}\n"
    
    txt = (
        f"👤 **{u['name']}**\n\n"
        f"📌 Пікселів: `{pixels:,}`\n"
        f"🎯 Статус: {status}"
    )
    if joined:
        txt += f"\n📅 У фракції з: {joined}"
    txt += medals_text
    
    await msg.edit_text(txt, parse_mode="Markdown")


async def list_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Список учасників фракції"""
    msg = await update.message.reply_text("⏳ Завантажую список фракції...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(FACTION_API_URL, headers=API_HEADERS, timeout=10) as resp:
                if resp.status != 200:
                    await msg.edit_text(f"❌ **Помилка:** {resp.status}")
                    return
                data = await resp.json()
    except Exception as e:
        await msg.edit_text(f"❌ **Помилка:** `{e}`", parse_mode="Markdown")
        return
    
    faction = data.get("faction", data)
    members = faction.get("members", [])
    
    if not members:
        await msg.edit_text("📭 Список учасників порожній.")
        return
    
    sorted_members = sorted(
        members,
        key=lambda m: m.get("User", {}).get("totalPixels", 0),
        reverse=True
    )
    
    name = faction.get("name", "?")
    tag = faction.get("tag", "")
    total = faction.get("totalPixels", 0)
    
    txt = f"🏰 **{name}** [{tag}]\n📊 Всього: `{total:,}` px\n👥 Учасників: {len(members)}\n\n"
    
    for i, m in enumerate(sorted_members[:30], 1):
        u = m.get("User", {})
        n = u.get("name", "?")
        p = u.get("totalPixels", 0)
        s = "✅" if m.get("isActive") else "💤"
        txt += f"{i}. {s} **{n}** — `{p:,}` px\n"
    
    if len(members) > 30:
        txt += f"\n_...та ще {len(members) - 30} учасників_"
    
    await msg.edit_text(txt, parse_mode="Markdown")


async def add_medal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Додати медаль: /madd <назва> <вага 1-10> у відповідь"""
    if not update.message.reply_to_message:
        await update.message.reply_text(
            "⚠️ **Використання:**\n\n"
            "Відповідай на повідомлення гравця командою:\n"
            "`/madd <назва> <вага 1-10>`\n\n"
            "**Приклад:** `/madd Найкращий_художник 10`",
            parse_mode="Markdown"
        )
        return
    
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "⚠️ **Формат:** `/madd <назва> <вага>`\n\n"
            "**Приклад:** `/madd Найкращий_художник 10`",
            parse_mode="Markdown"
        )
        return
    
    try:
        weight = int(context.args[-1])
        if weight < 1 or weight > 10:
            raise ValueError
        name = " ".join(context.args[:-1])
    except ValueError:
        await update.message.reply_text("❌ Вага має бути від 1 до 10!")
        return
    
    target_id = str(update.message.reply_to_message.from_user.id)
    
    if target_id not in state["medals"]:
        state["medals"][target_id] = []
    
    state["medals"][target_id].append({
        "name": name,
        "weight": weight,
        "date": datetime.now().strftime("%Y-%m-%d")
    })
    save_state()
    
    stars = "⭐" * weight
    await update.message.reply_text(
        f"✅ **Медаль додано!**\n\n"
        f"🏅 **{name}**\n"
        f"{stars}",
        parse_mode="Markdown"
    )


async def delete_medal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Видалити медаль: /mdel <номер> у відповідь"""
    if not update.message.reply_to_message:
        await update.message.reply_text(
            "⚠️ **Використання:**\n\n"
            "Відповідай на повідомлення гравця командою:\n"
            "`/mdel <номер>`\n\n"
            "**Приклад:** `/mdel 1`",
            parse_mode="Markdown"
        )
        return
    
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(
            "⚠️ **Формат:** `/mdel <номер>`\n\n"
            "**Приклад:** `/mdel 1`",
            parse_mode="Markdown"
        )
        return
    
    try:
        index = int(context.args[0]) - 1
    except ValueError:
        await update.message.reply_text("❌ Номер має бути числом!")
        return
    
    target_id = str(update.message.reply_to_message.from_user.id)
    
    if target_id not in state["medals"] or not state["medals"][target_id]:
        await update.message.reply_text("❌ У цього гравця немає медалей!")
        return
    
    if index < 0 or index >= len(state["medals"][target_id]):
        await update.message.reply_text("❌ Медалі з таким номером не існує!")
        return
    
    removed = state["medals"][target_id].pop(index)
    save_state()
    
    await update.message.reply_text(
        f"✅ **Медаль видалено!**\n\n"
        f"🏅 {removed['name']}",
        parse_mode="Markdown"
    )


async def change_site(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Змінити сайт: /site_change <назва>"""
    if not context.args:
        sites_list = "\n".join([f"• `{name}`" for name in SITES.keys()])
        await update.message.reply_text(
            f"⚠️ **Використання:** `/site_change <назва>`\n\n"
            f"**Доступні сайти:**\n{sites_list}\n\n"
            f"🌐 **Поточний:** `{CURRENT_SITE}`",
            parse_mode="Markdown"
        )
        return
    
    site_name = context.args[0].lower()
    
    if set_site(site_name):
        state["current_site"] = site_name
        save_state()
        await update.message.reply_text(
            f"✅ **Сайт змінено!**\n\n"
            f"🌐 Новий сайт: **{site_name}**\n"
            f"🔗 URL: {SITES[site_name]['url']}",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"❌ **Невідомий сайт:** `{site_name}`\n\n"
            f"Доступні: {', '.join(SITES.keys())}",
            parse_mode="Markdown"
        )


# ======================================================================
# 🚀 ЗАПУСК БОТА
# ======================================================================

def main():
    """Головна функція запуску бота"""
    
    # Перевірка токену
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN не встановлено!")
        print("\n" + "="*60)
        print("❌ ПОМИЛКА: BOT_TOKEN не встановлено!")
        print("="*60)
        print("\nВстанови токен одним з способів:")
        print("1. Через змінну середовища:")
        print("   export BOT_TOKEN='твій_токен'")
        print("\n2. Або відредагуй bot.py і вставте токен в:")
        print("   BOT_TOKEN = 'твій_токен'")
        print("="*60 + "\n")
        sys.exit(1)
    
    # Завантажуємо стан
    load_state()
    
    # Створюємо application
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # ConversationHandler для завантаження шаблону
    upload_conv = ConversationHandler(
        entry_points=[CommandHandler("upload_template", upload_start)],
        states={
            UPLOAD_TEMPLATE_WAITING: [MessageHandler(filters.Document.ALL, upload_file)]
        },
        fallbacks=[CommandHandler("cancel", cancel_upload)]
    )
    
    # Реєструємо всі команди
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("set_coords", set_coords))
    app.add_handler(CommandHandler("check", check_progress))
    app.add_handler(CommandHandler("test_check", test_check))
    app.add_handler(CommandHandler("connect", connect_player))
    app.add_handler(CommandHandler("profile", get_profile))
    app.add_handler(CommandHandler("list", list_members))
    app.add_handler(CommandHandler("madd", add_medal))
    app.add_handler(CommandHandler("mdel", delete_medal))
    app.add_handler(CommandHandler("site_change", change_site))
    app.add_handler(upload_conv)
    
    # Виводимо інфо про запуск
    logger.info("=" * 60)
    logger.info("🤖 UkrLirn Monitor Bot запущено!")
    logger.info("=" * 60)
    logger.info(f"📐 Режим: 256px тайли (STRICT)")
    logger.info(f"🌐 Поточний сайт: {CURRENT_SITE}")
    logger.info(f"🔐 Авторизація: {'✅ Активна' if AUTH_COOKIE else '❌ Відсутня'}")
    logger.info(f"🎨 Толеранс кольору: {COLOR_TOLERANCE}")
    logger.info(f"📂 Файл стану: {STATE_FILE}")
    logger.info("=" * 60)
    logger.info("Бот готовий до роботи! Натисни Ctrl+C для зупинки.")
    logger.info("=" * 60)
    
    # Запускаємо бота
    try:
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("⛔ Зупинка бота...")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
