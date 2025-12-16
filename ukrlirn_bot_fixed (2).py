#!/usr/bin/env python3
# bot.py - UkrLirn Monitor Bot (Fixed + Fast + /get restored)

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

BOT_TOKEN = os.environ.get("BOT_TOKEN", "8133267244:AAEimjL3_gSTWiYV7bglcyrqGA2woQykDZo")
AUTH_COOKIE = os.environ.get("AUTH_COOKIE", "s%3AS2qBqqlzYPCWST-OalOz6svoEoTYQIi9.%2BL0JZVKMRNrHr9eQ8WAuf4D9MdthKJP3pHCrqliUmZs")

API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Cookie': f'cpn.session={AUTH_COOKIE}; plang=ru',
    'Accept': 'application/json'
}

COLOR_TOLERANCE = 20
STATE_FILE = "state.json"
STRICT_TILE_SIZE = 256
FACTION_ID = 530  # UkrLirn | Pxl

# ======================================================================
# 🌐 САЙТИ
# ======================================================================

SITES = {
    "pixmap": {
        "url": "https://pixmap.fun",
        "tile_url": "https://pixmap.fun/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://pixmap.fun/api/me",
        "api_faction_list": "https://pixmap.fun/api/faction/list"
    },
    "pixelya": {
        "url": "https://pixelya.fun",
        "tile_url": "https://pixelya.fun/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://pixelya.fun/api/me",
        "api_faction_list": "https://pixelya.fun/api/faction/list"
    },
    "globepixel": {
        "url": "https://globepixel.net",
        "tile_url": "https://globepixel.net/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://globepixel.net/api/me",
        "api_faction_list": "https://globepixel.net/api/faction/list"
    }
}

CURRENT_SITE = "pixmap"

def set_site(site_name: str) -> bool:
    global CURRENT_SITE
    if site_name.lower() in SITES:
        CURRENT_SITE = site_name.lower()
        return True
    return False

def get_current_site():
    return SITES[CURRENT_SITE]

# ======================================================================
# 🎨 ДВИЖОК (ШВИДКИЙ!)
# ======================================================================

async def fetch_api_me():
    """Отримує інфо про канваси"""
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


async def fetch_faction_data():
    """Отримує дані фракції через /api/faction/list"""
    site = get_current_site()
    url = site["api_faction_list"]
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=API_HEADERS, timeout=10) as resp:
                if resp.status == 200:
                    factions = await resp.json()
                    # Шукаємо нашу фракцію по ID
                    for faction in factions:
                        if faction.get("id") == FACTION_ID:
                            return faction
                    logger.warning(f"Фракцію {FACTION_ID} не знайдено")
                else:
                    logger.warning(f"API faction/list: {resp.status}")
        except Exception as e:
            logger.error(f"Помилка faction API: {e}")
    return None


async def fetch_tile(session, url, offx, offy, image, needed=False, debug_save=False, tx=0, ty=0):
    """Завантажує тайл"""
    for attempt in range(2):  # Тільки 2 спроби для швидкості
        try:
            async with session.get(url, headers=API_HEADERS, timeout=10) as resp:
                if resp.status == 404:
                    if needed:
                        empty = PIL.Image.new('RGBA', (STRICT_TILE_SIZE, STRICT_TILE_SIZE), (0, 0, 0, 0))
                        image.paste(empty, (offx, offy))
                    return True
                
                if resp.status == 200:
                    data = await resp.read()
                    if data:
                        tile = PIL.Image.open(io.BytesIO(data)).convert('RGBA')
                        
                        if tile.size != (STRICT_TILE_SIZE, STRICT_TILE_SIZE):
                            tile = tile.resize((STRICT_TILE_SIZE, STRICT_TILE_SIZE), PIL.Image.NEAREST)
                        
                        if debug_save:
                            os.makedirs("debug/tiles", exist_ok=True)
                            tile.save(f"debug/tiles/tile_{tx}_{ty}.png")
                        
                        image.paste(tile, (offx, offy), tile)
                        return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout тайлу [{tx},{ty}]")
        except Exception as e:
            logger.warning(f"Помилка [{tx},{ty}]: {e}")
        
        if attempt < 1:
            await asyncio.sleep(0.3)
    
    # Порожній тайл якщо не вдалось
    if needed:
        empty = PIL.Image.new('RGBA', (STRICT_TILE_SIZE, STRICT_TILE_SIZE), (0, 0, 0, 0))
        image.paste(empty, (offx, offy))
    return False


async def get_canvas_area(canvas_id, x, y, width, height, canvas_size=32768, debug_save=False):
    """Завантажує область з канвасу"""
    site = get_current_site()
    offset = int(-canvas_size / 2)
    tile_size = STRICT_TILE_SIZE
    
    xc = (x - offset) // tile_size
    wc = (x + width - offset) // tile_size
    yc = (y - offset) // tile_size
    hc = (y + height - offset) // tile_size
    
    logger.info(f"📐 Область: x={x}, y={y}, {width}x{height}")
    logger.info(f"🗺️ Тайли: X[{xc}..{wc}], Y[{yc}..{hc}]")
    
    result = PIL.Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Завантажуємо тайли паралельно
    async with aiohttp.ClientSession() as session:
        tasks = []
        for iy in range(yc, hc + 1):
            for ix in range(xc, wc + 1):
                url = site["tile_url"].format(canvas_id=canvas_id, zoom=7, tx=ix, ty=iy)
                offx = ix * tile_size + offset - x
                offy = iy * tile_size + offset - y
                tasks.append(fetch_tile(session, url, offx, offy, result, needed=True, debug_save=debug_save, tx=ix, ty=iy))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        loaded = sum(1 for r in results if r and not isinstance(r, Exception))
        logger.info(f"✅ Завантажено: {loaded}/{len(tasks)}")
    
    if debug_save:
        os.makedirs("debug", exist_ok=True)
        result.save("debug/board_full.png")
        logger.info("💾 Доска: debug/board_full.png")
    
    return result


def compare_with_template(template, board, tolerance=20):
    """Порівнює шаблон з дошкою (ШВИДКО!)"""
    tw, th = template.size
    
    if board.size != (tw, th):
        board = board.crop((0, 0, min(tw, board.size[0]), min(th, board.size[1])))
        if board.size != (tw, th):
            # Якщо board менший - доповнюємо прозорим
            temp = PIL.Image.new('RGBA', (tw, th), (0, 0, 0, 0))
            temp.paste(board, (0, 0))
            board = temp
    
    t_array = np.array(template, dtype=np.uint8)
    b_array = np.array(board, dtype=np.uint8)
    
    # Маска непрозорих пікселів шаблону
    template_mask = t_array[..., 3] > 10
    total_pixels = int(template_mask.sum())
    
    if total_pixels == 0:
        return {"total": 0, "placed": 0, "remaining": 0, "percent": 100.0}
    
    # Швидке порівняння кольорів
    diff = np.abs(b_array[..., :3].astype(np.int16) - t_array[..., :3].astype(np.int16))
    color_distance = np.sqrt((diff ** 2).sum(axis=-1))
    color_match = color_distance <= tolerance
    
    board_mask = b_array[..., 3] > 10
    
    placed_pixels = int((template_mask & board_mask & color_match).sum())
    remaining_pixels = total_pixels - placed_pixels
    percent = (placed_pixels / total_pixels * 100.0) if total_pixels > 0 else 100.0
    
    logger.info(f"✅ {placed_pixels}/{total_pixels} ({percent:.1f}%)")
    
    return {
        "total": total_pixels,
        "placed": placed_pixels,
        "remaining": remaining_pixels,
        "percent": percent
    }


def create_overlay(template, board, tolerance=20, output_path=None):
    """Створює overlay (червоні = неправильні)"""
    if not output_path:
        return None
        
    tw, th = template.size
    if board.size != (tw, th):
        board = board.crop((0, 0, min(tw, board.size[0]), min(th, board.size[1])))
        if board.size != (tw, th):
            temp = PIL.Image.new('RGBA', (tw, th), (0, 0, 0, 0))
            temp.paste(board, (0, 0))
            board = temp
    
    t_array = np.array(template, dtype=np.uint8)
    b_array = np.array(board, dtype=np.uint8)
    
    template_mask = t_array[..., 3] > 10
    board_mask = b_array[..., 3] > 10
    
    diff = np.abs(b_array[..., :3].astype(np.int16) - t_array[..., :3].astype(np.int16))
    color_distance = np.sqrt((diff ** 2).sum(axis=-1))
    color_match = color_distance <= tolerance
    
    output = b_array.copy()
    bad_pixels = template_mask & (~color_match | ~board_mask)
    output[bad_pixels] = [255, 0, 0, 255]
    
    # Конвертуємо в RGB для Telegram
    output_img = PIL.Image.fromarray(output, mode='RGBA').convert('RGB')
    output_img.save(output_path, 'PNG')
    logger.info(f"💾 Overlay: {output_path}")
    return output_path


async def process_lirn_template(template_path, x, y, canvas_id=0, tolerance=20, overlay_path=None, debug_mode=False):
    """Головна функція обробки"""
    template = PIL.Image.open(template_path).convert("RGBA")
    width, height = template.size
    logger.info(f"📐 Шаблон: {width}x{height} px")
    
    api_me = await fetch_api_me()
    canvas_size = 32768
    
    if api_me and 'canvases' in api_me:
        canvas_info = api_me['canvases'].get(str(canvas_id))
        if canvas_info:
            canvas_size = canvas_info.get('size', 32768)
    
    board = await get_canvas_area(canvas_id, x, y, width, height, canvas_size, debug_save=debug_mode)
    result = compare_with_template(template, board, tolerance)
    
    if overlay_path:
        create_overlay(template, board, tolerance, overlay_path)
    
    return result

# ======================================================================
# 🤖 БОТ
# ======================================================================

UPLOAD_TEMPLATE_WAITING = 1
UPLOAD_VERSION_WAITING = 2
UPLOAD_COORDS_WAITING = 3
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
        except Exception as e:
            logger.error(f"Помилка завантаження: {e}")

def save_state():
    try:
        state["current_site"] = CURRENT_SITE
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Помилка збереження: {e}")


# --- КОМАНДИ ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎨 **UkrLirn Monitor Bot**\n\n"
        "**Шаблон:**\n"
        "• `/upload_template` — завантажити (версія + координати)\n"
        "• `/get` — скачати поточний шаблон\n"
        "• `/check` — перевірити прогрес\n\n"
        "**Гравці:**\n"
        "• `/connect <нік>` — прив'язати\n"
        "• `/profile [нік]` — профіль\n"
        "• `/list` — список фракції\n\n"
        "**Медалі:**\n"
        "• `/madd <назва> <1-10>` (у відповідь)\n"
        "• `/mdel <номер>` (у відповідь)\n\n"
        "**Інше:**\n"
        "• `/site_change <сайт>`\n"
        "• `/status` — налаштування",
        parse_mode="Markdown"
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    x, y = state.get("lirn_coords", [0, 0])
    template_exists = os.path.exists(LIRN_TEMPLATE["file"])
    
    await update.message.reply_text(
        f"⚙️ **Налаштування:**\n\n"
        f"🌐 Сайт: `{CURRENT_SITE}`\n"
        f"📐 Шаблон: {'✅' if template_exists else '❌'}\n"
        f"📍 Координати: {f'({x}, {y})' if [x,y] != [0,0] else '❌'}\n"
        f"🎨 Толеранс: {COLOR_TOLERANCE}\n"
        f"🏰 Фракція ID: {FACTION_ID}",
        parse_mode="Markdown"
    )


async def set_coords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) != 2:
        await update.message.reply_text("⚠️ `/set_coords X Y`\n\nПриклад: `/set_coords 4031 -11628`", parse_mode="Markdown")
        return
    
    try:
        x, y = int(context.args[0]), int(context.args[1])
        state["lirn_coords"] = [x, y]
        save_state()
        await update.message.reply_text(f"✅ Координати: `{x}, {y}`", parse_mode="Markdown")
    except ValueError:
        await update.message.reply_text("❌ Координати мають бути числами!")


async def upload_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📤 **Завантаження шаблону**\n\n"
        "**Крок 1/3:** Надішли PNG файл шаблону\n\n"
        "_Для скасування: /cancel_"
    )
    return UPLOAD_TEMPLATE_WAITING


async def upload_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith('.png'):
        await update.message.reply_text("❌ Потрібен PNG файл!")
        return ConversationHandler.END
    
    # Зберігаємо файл тимчасово
    file = await doc.get_file()
    os.makedirs("templates", exist_ok=True)
    temp_path = "templates/temp_upload.png"
    await file.download_to_drive(temp_path)
    
    # Зберігаємо шлях у context
    context.user_data['temp_template'] = temp_path
    
    img = PIL.Image.open(temp_path)
    await update.message.reply_text(
        f"✅ Файл отримано: `{img.size[0]}x{img.size[1]}` px\n\n"
        f"**Крок 2/3:** Введи назву версії\n"
        f"_Приклад: `v1.0` або `0_0`_",
        parse_mode="Markdown"
    )
    return UPLOAD_VERSION_WAITING


async def upload_version(update: Update, context: ContextTypes.DEFAULT_TYPE):
    version = update.message.text.strip()
    
    if not version or len(version) > 50:
        await update.message.reply_text("❌ Назва версії має бути 1-50 символів!")
        return UPLOAD_VERSION_WAITING
    
    context.user_data['template_version'] = version
    
    await update.message.reply_text(
        f"✅ Версія: `{version}`\n\n"
        f"**Крок 3/3:** Введи координати\n"
        f"_Формат: `X Y` (наприклад: `4031 -11628`)_",
        parse_mode="Markdown"
    )
    return UPLOAD_COORDS_WAITING


async def upload_coords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        parts = update.message.text.strip().split()
        if len(parts) != 2:
            raise ValueError
        x, y = int(parts[0]), int(parts[1])
    except ValueError:
        await update.message.reply_text(
            "❌ Неправильний формат!\n\n"
            "_Введи координати: `X Y`_\n"
            "_Приклад: `4031 -11628`_",
            parse_mode="Markdown"
        )
        return UPLOAD_COORDS_WAITING
    
    # Переміщуємо тимчасовий файл в фінальний
    temp_path = context.user_data.get('temp_template')
    version = context.user_data.get('template_version')
    
    if temp_path and os.path.exists(temp_path):
        os.rename(temp_path, LIRN_TEMPLATE["file"])
        
        # Зберігаємо координати
        state["lirn_coords"] = [x, y]
        save_state()
        
        img = PIL.Image.open(LIRN_TEMPLATE["file"])
        
        await update.message.reply_text(
            f"✅ **Шаблон збережено!**\n\n"
            f"📐 Розмір: `{img.size[0]}x{img.size[1]}` px\n"
            f"📝 Версія: `{version}`\n"
            f"📍 Координати: `{x}, {y}`\n\n"
            f"Перевір прогрес: `/check`",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text("❌ Помилка збереження файлу!")
    
    return ConversationHandler.END


async def cancel_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Скасовано")
    return ConversationHandler.END


async def check_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Повна перевірка з overlay"""
    if not os.path.exists(LIRN_TEMPLATE["file"]):
        await update.message.reply_text("❌ Завантаж шаблон: `/upload_template`", parse_mode="Markdown")
        return
    
    x, y = state.get("lirn_coords", [0, 0])
    if [x, y] == [0, 0]:
        await update.message.reply_text("❌ Встанови координати: `/set_coords X Y`", parse_mode="Markdown")
        return
    
    status_msg = await update.message.reply_text(f"⏳ Перевіряю ({x}, {y})...")
    
    try:
        os.makedirs("progress", exist_ok=True)
        
        # З таймаутом 60 секунд
        res = await asyncio.wait_for(
            process_lirn_template(
                LIRN_TEMPLATE["file"], x, y,
                tolerance=COLOR_TOLERANCE,
                overlay_path="progress/overlay.png"
            ),
            timeout=60.0
        )
        
        caption = (
            f"📊 **Прогрес UkrLirn**\n\n"
            f"🎯 Всього: `{res['total']:,}` px\n"
            f"✅ Готово: `{res['placed']:,}` px\n"
            f"❌ Залишилось: `{res['remaining']:,}` px\n\n"
            f"📈 **{res['percent']:.1f}%**\n\n"
            f"📍 ({x}, {y}) • {CURRENT_SITE}"
        )
        
        # Відправляємо overlay як ДОКУМЕНТ (PNG файл)
        with open("progress/overlay.png", "rb") as f:
            await update.message.reply_document(
                document=f,
                caption=caption,
                parse_mode="Markdown",
                filename="progress.png"
            )
        
        await status_msg.delete()
        
    except asyncio.TimeoutError:
        await status_msg.edit_text("❌ **Таймаут!** Занадто велика область або повільний інтернет.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Помилка check: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Помилка: `{str(e)}`", parse_mode="Markdown")


ови координати: `/set_coords X Y`", parse_mode="Markdown")
        return
    
    status_msg = await update.message.reply_text(f"⏳ Швидка перевірка...")
    
    try:
        # БЕЗ overlay для швидкості
        res = await asyncio.wait_for(
            process_lirn_template(
                LIRN_TEMPLATE["file"], x, y,
                tolerance=COLOR_TOLERANCE,
                overlay_path=None  # БЕЗ overlay!
            ),
            timeout=30.0
        )
        
        txt = (
            f"📊 **Прогрес UkrLirn**\n\n"
            f"🎯 Всього: `{res['total']:,}` px\n"
            f"✅ Готово: `{res['placed']:,}` px\n"
            f"❌ Залишилось: `{res['remaining']:,}` px\n\n"
            f"📈 **{res['percent']:.1f}%**\n\n"
            f"_Для overlay використай /check_"
        )
        
        await status_msg.edit_text(txt, parse_mode="Markdown")
        
    except asyncio.TimeoutError:
        await status_msg.edit_text("❌ Таймаут!", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Помилка get: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ `{str(e)}`", parse_mode="Markdown")


async def connect_player(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ `/connect <нік>`\n\nПриклад: `/connect Puwe`", parse_mode="Markdown")
        return
    
    nick = " ".join(context.args)
    state["user_links"][str(update.effective_user.id)] = nick
    save_state()
    await update.message.reply_text(f"✅ Прив'язано: **{nick}**", parse_mode="Markdown")


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
            await update.message.reply_text("⚠️ `/connect <нік>`", parse_mode="Markdown")
            return
    
    msg = await update.message.reply_text("🔍 Завантажую...")
    
    try:
        faction_data = await fetch_faction_data()
        if not faction_data:
            await msg.edit_text("❌ Не вдалось завантажити дані фракції")
            return
        
        found = None
        for member in faction_data.get("members", []):
            if member.get("name", "").lower() == nick.lower():
                found = member
                break
        
        if not found:
            await msg.edit_text(f"❌ **{nick}** не знайдений", parse_mode="Markdown")
            return
        
        pixels = found.get("totalPixels", 0)
        daily = found.get("dailyPixels", 0)
        role = found.get("role", "member")
        
        medals_text = ""
        if target_id and target_id in state["medals"]:
            medals_text = "\n\n🏅 **Медалі:**\n"
            for i, m in enumerate(state["medals"][target_id], 1):
                stars = "⭐" * m["weight"]
                medals_text += f"{i}. {m['name']} {stars}\n"
        
        txt = (
            f"👤 **{found['name']}**\n\n"
            f"📌 Всього: `{pixels:,}` px\n"
            f"📅 Сьогодні: `{daily:,}` px\n"
            f"👑 Роль: {role}"
        )
        txt += medals_text
        
        await msg.edit_text(txt, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Помилка profile: {e}", exc_info=True)
        await msg.edit_text(f"❌ `{str(e)}`", parse_mode="Markdown")


async def list_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Завантажую...")
    
    try:
        faction_data = await fetch_faction_data()
        if not faction_data:
            await msg.edit_text("❌ Не вдалось завантажити")
            return
        
        members = faction_data.get("members", [])
        if not members:
            await msg.edit_text("📭 Список порожній")
            return
        
        sorted_members = sorted(members, key=lambda m: m.get("totalPixels", 0), reverse=True)
        
        name = faction_data.get("name", "?")
        total = faction_data.get("totalPixels", 0)
        
        txt = f"🏰 **{name}**\n📊 Всього: `{total:,}` px\n👥 Учасників: {len(members)}\n\n"
        
        for i, m in enumerate(sorted_members[:20], 1):
            n = m.get("name", "?")
            p = m.get("totalPixels", 0)
            role = m.get("role", "")
            crown = "👑" if role == "owner" else ""
            txt += f"{i}. {crown}**{n}** — `{p:,}` px\n"
        
        if len(members) > 20:
            txt += f"\n_...ще {len(members) - 20}_"
        
        await msg.edit_text(txt, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Помилка list: {e}", exc_info=True)
        await msg.edit_text(f"❌ `{str(e)}`", parse_mode="Markdown")


async def add_medal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Відповідай на повідомлення!\n`/madd <назва> <1-10>`", parse_mode="Markdown")
        return
    
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("⚠️ `/madd <назва> <вага>`\n\nПриклад: `/madd Художник 10`", parse_mode="Markdown")
        return
    
    try:
        weight = int(context.args[-1])
        if weight < 1 or weight > 10:
            raise ValueError
        name = " ".join(context.args[:-1])
    except ValueError:
        await update.message.reply_text("❌ Вага 1-10!")
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
    await update.message.reply_text(f"✅ Медаль додано!\n\n🏅 **{name}** {stars}", parse_mode="Markdown")


async def delete_medal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Відповідай на повідомлення!\n`/mdel <номер>`", parse_mode="Markdown")
        return
    
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("⚠️ `/mdel <номер>`", parse_mode="Markdown")
        return
    
    try:
        index = int(context.args[0]) - 1
    except ValueError:
        await update.message.reply_text("❌ Номер має бути числом!")
        return
    
    target_id = str(update.message.reply_to_message.from_user.id)
    
    if target_id not in state["medals"] or not state["medals"][target_id]:
        await update.message.reply_text("❌ Немає медалей!")
        return
    
    if index < 0 or index >= len(state["medals"][target_id]):
        await update.message.reply_text("❌ Медалі з таким номером не існує!")
        return
    
    removed = state["medals"][target_id].pop(index)
    save_state()
    
    await update.message.reply_text(f"✅ Видалено: 🏅 {removed['name']}", parse_mode="Markdown")


async def change_site(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        sites_list = "\n".join([f"• `{name}`" for name in SITES.keys()])
        await update.message.reply_text(
            f"⚠️ `/site_change <сайт>`\n\n**Доступні:**\n{sites_list}\n\n🌐 Поточний: `{CURRENT_SITE}`",
            parse_mode="Markdown"
        )
        return
    
    site_name = context.args[0].lower()
    
    if set_site(site_name):
        state["current_site"] = site_name
        save_state()
        await update.message.reply_text(
            f"✅ Сайт змінено!\n\n🌐 **{site_name}**\n🔗 {SITES[site_name]['url']}",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(f"❌ Невідомий сайт: `{site_name}`", parse_mode="Markdown")


# ======================================================================
# 🚀 ЗАПУСК
# ======================================================================

def main():
    """Головна функція"""
    
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN не встановлено!")
        print("\n" + "="*60)
        print("❌ Встанови BOT_TOKEN!")
        print("="*60)
        print("\nСпосіб 1: export BOT_TOKEN='твій_токен'")
        print("Спосіб 2: Відредагуй bot.py\n")
        print("="*60 + "\n")
        sys.exit(1)
    
    load_state()
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # ConversationHandler для завантаження (3 кроки!)
    upload_conv = ConversationHandler(
        entry_points=[CommandHandler("upload_template", upload_start)],
        states={
            UPLOAD_TEMPLATE_WAITING: [MessageHandler(filters.Document.ALL, upload_file)],
            UPLOAD_VERSION_WAITING: [MessageHandler(filters.TEXT & ~filters.COMMAND, upload_version)],
            UPLOAD_COORDS_WAITING: [MessageHandler(filters.TEXT & ~filters.COMMAND, upload_coords)]
        },
        fallbacks=[CommandHandler("cancel", cancel_upload)]
    )
    
    # Реєструємо команди
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("set_coords", set_coords))
    app.add_handler(CommandHandler("check", check_progress))
    app.add_handler(CommandHandler("get", get_template))  # СКАЧАТИ ШАБЛОН!
    app.add_handler(CommandHandler("connect", connect_player))
    app.add_handler(CommandHandler("profile", get_profile))
    app.add_handler(CommandHandler("list", list_members))
    app.add_handler(CommandHandler("madd", add_medal))
    app.add_handler(CommandHandler("mdel", delete_medal))
    app.add_handler(CommandHandler("site_change", change_site))
    app.add_handler(upload_conv)
    
    logger.info("=" * 60)
    logger.info("🤖 UkrLirn Monitor Bot запущено!")
    logger.info("=" * 60)
    logger.info(f"📐 Режим: 256px тайли")
    logger.info(f"🌐 Сайт: {CURRENT_SITE}")
    logger.info(f"🏰 Фракція ID: {FACTION_ID}")
    logger.info(f"🔐 Auth: {'✅' if AUTH_COOKIE else '❌'}")
    logger.info("=" * 60)
    logger.info("✅ Готовий! Ctrl+C для зупинки.")
    logger.info("=" * 60)
    
    try:
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        logger.info("\n⛔ Зупинка...")
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()