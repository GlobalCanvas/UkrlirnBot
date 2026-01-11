#!/usr/bin/env python3
# bot.py - UkrLirn Monitor Bot (Full Features)

import sys
import os
import io
import json
import asyncio
import logging
import math
from datetime import datetime
import numpy as np
import PIL.Image
import aiohttp
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters
)

# --- ЛОГУВАННЯ ---
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ======================================================================
# ⚙️ КОНФІГУРАЦІЯ
# ======================================================================

BOT_TOKEN = os.environ.get("BOT_TOKEN", "8133267244:AAFPj7GcUhgUPUiuAxM9afwQFoSsB5hEtUc")

# COOKIES
COOKIE_PIXELYA = "s%3AMiTkf-27ZLGgiV8Xt13qhsg6tGliAuLx.p64kN9RRkoAMMecownmgM1SiJmO67d4CuNR4nD3k1AM"
COOKIE_PIXMAP = "s%3AS2qBqqlzYPCWST-OalOz6svoEoTYQIi9.%2BL0JZVKMRNrHr9eQ8WAuf4D9MdthKJP3pHCrqliUmZs"

CHUNK_SIZE = 256
MAX_CONCURRENT = 20
STATE_FILE = "state.json"
TEMPLATE_FILE = "template.png"
FACTION_ID = 359  # UkrLirn на pixelya

# Параметри сайтів
SITES = {
    "pixelya": {
        "url": "https://pixelya.fun",
        "chunk_url": "https://pixelya.fun/chunks/5/{x}/{y}.bmp",
        "api_me": "https://pixelya.fun/api/me",
        "api_faction": "https://pixelya.fun/api/faction/list",
        "canvas_size": 65536,
        "cookie": f"pixelya.session={COOKIE_PIXELYA}"
    },
    "pixmap": {
        "url": "https://pixmap.fun",
        "chunk_url": "https://pixmap.fun/chunks/5/{x}/{y}.bmp",
        "api_me": "https://pixmap.fun/api/me",
        "api_faction": "https://pixmap.fun/api/faction/list",
        "canvas_size": 65536,
        "cookie": f"cpn.session={COOKIE_PIXMAP}"
    }
}

# Стан
state = {
    "site": "pixelya",
    "coords": [0, 0],
    "colors": {},
    "user_links": {},  # {telegram_id: nickname}
    "medals": {}  # {telegram_id: [{name, weight, date}]}
}

# ======================================================================
# 🎨 ДВИЖОК
# ======================================================================

def load_state():
    global state
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                loaded = json.load(f)
                state.update(loaded)
            logger.info(f"✅ Стан завантажено: {state['site']}")
        except Exception as e:
            logger.error(f"Помилка завантаження: {e}")

def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Помилка збереження: {e}")


async def fetch_canvas_colors(site_name):
    """Отримує палітру кольорів з API"""
    site = SITES[site_name]
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Cookie': site['cookie']
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(site['api_me'], headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    canvas = data['canvases'].get('5')
                    if canvas and 'colors' in canvas:
                        colors = {}
                        for i, color in enumerate(canvas['colors']):
                            if len(color) == 3:
                                colors[i] = tuple(color) + (255,)
                            else:
                                colors[i] = tuple(color)
                        logger.info(f"✅ Завантажено {len(colors)} кольорів з canvas 5")
                        return colors
    except Exception as e:
        logger.error(f"Помилка API: {e}")
    
    return {i: (i*10, i*10, i*10, 255) for i in range(32)}


async def fetch_faction_data(site_name):
    """Отримує дані фракції"""
    site = SITES[site_name]
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Cookie': site['cookie']
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(site['api_faction'], headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    factions = await resp.json()
                    for faction in factions:
                        if faction.get("id") == FACTION_ID:
                            return faction
    except Exception as e:
        logger.error(f"Помилка faction API: {e}")
    return None


async def fetch_chunk(session, url, headers, cx, cy, colors, sem):
    """Читання чанку"""
    async with sem:
        try:
            async with session.get(url, headers=headers, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    
                    if len(data) == 0:
                        return PIL.Image.new('RGBA', (CHUNK_SIZE, CHUNK_SIZE), (0, 0, 0, 0))
                    
                    img = PIL.Image.new('RGBA', (CHUNK_SIZE, CHUNK_SIZE), (0, 0, 0, 0))
                    pixels = img.load()
                    
                    for i, byte in enumerate(data[:CHUNK_SIZE*CHUNK_SIZE]):
                        x = i % CHUNK_SIZE
                        y = i // CHUNK_SIZE
                        color_index = byte & 0x7F
                        pixels[x, y] = colors.get(color_index, (0, 0, 0, 255))
                    
                    return img
                
                return PIL.Image.new('RGBA', (CHUNK_SIZE, CHUNK_SIZE), (0, 0, 0, 0))
                    
        except:
            return PIL.Image.new('RGBA', (CHUNK_SIZE, CHUNK_SIZE), (0, 0, 0, 0))


async def get_map_area(site_name, x, y, w, h, progress_msg=None):
    """Завантажує область карти з прогрес-баром"""
    site = SITES[site_name]
    canvas_size = site["canvas_size"]
    
    if site_name not in state.get("colors", {}):
        if progress_msg:
            await progress_msg.edit_text("⏳ Завантажую палітру кольорів...")
        colors = await fetch_canvas_colors(site_name)
        if "colors" not in state:
            state["colors"] = {}
        state["colors"][site_name] = colors
    else:
        colors = state["colors"][site_name]
    
    canvasoffset = math.sqrt(canvas_size)
    offset = int(-canvasoffset * canvasoffset / 2)
    
    cx_start = (x - offset) // CHUNK_SIZE
    cx_end = (x + w - offset) // CHUNK_SIZE
    cy_start = (y - offset) // CHUNK_SIZE
    cy_end = (y + h - offset) // CHUNK_SIZE
    
    total_chunks = (cx_end - cx_start + 1) * (cy_end - cy_start + 1)
    logger.info(f"🗺️ Чанки: X[{cx_start}..{cx_end}], Y[{cy_start}..{cy_end}], всього: {total_chunks}")
    
    canvas = PIL.Image.new('RGBA', (w, h), (0, 0, 0, 0))
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Cookie': site['cookie'],
        'Accept': 'image/bmp,*/*'
    }
    
    # Лічильник завантажених чанків
    loaded_chunks = [0]  # Використовуємо list щоб змінювати в async функції
    last_update = [0]  # Час останнього оновлення
    
    async def update_progress():
        """Оновлює прогрес-бар"""
        if not progress_msg:
            return
        
        current_time = asyncio.get_event_loop().time()
        # Оновлюємо не частіше ніж раз на 2 секунди
        if current_time - last_update[0] < 2 and loaded_chunks[0] < total_chunks:
            return
        
        last_update[0] = current_time
        percent = (loaded_chunks[0] / total_chunks * 100)
        filled = int(percent / 5)
        bar = "🟩" * filled + "⬜" * (20 - filled)
        
        try:
            await progress_msg.edit_text(
                f"⏳ **Завантаження чанків**\n\n"
                f"{bar}\n\n"
                f"📦 {loaded_chunks[0]}/{total_chunks} ({percent:.1f}%)",
                parse_mode="Markdown"
            )
        except:
            pass  # Ігноруємо помилки (занадто часте оновлення)
    
    async def fetch_with_progress(session, url, headers, cx, cy, colors, sem, canvas, px, py):
        """Завантажує чанк і оновлює прогрес"""
        img = await fetch_chunk(session, url, headers, cx, cy, colors, sem)
        try:
            canvas.paste(img, (px, py), img)
        except Exception as e:
            logger.error(f"Помилка paste [{cx},{cy}]: {e}")
        
        loaded_chunks[0] += 1
        await update_progress()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for cy in range(cy_start, cy_end + 1):
            for cx in range(cx_start, cx_end + 1):
                url = site["chunk_url"].format(x=cx, y=cy)
                px = (cx * CHUNK_SIZE + offset) - x
                py = (cy * CHUNK_SIZE + offset) - y
                tasks.append(fetch_with_progress(session, url, headers, cx, cy, colors, sem, canvas, px, py))
        
        await asyncio.gather(*tasks)
    
    # Фінальне оновлення
    if progress_msg:
        await progress_msg.edit_text("✅ Завантаження завершено!")
    
    return canvas


async def process_chunk(session, url, headers, cx, cy, colors, sem, canvas, px, py):
    img = await fetch_chunk(session, url, headers, cx, cy, colors, sem)
    try:
        canvas.paste(img, (px, py), img)
    except Exception as e:
        logger.error(f"Помилка paste [{cx},{cy}]: {e}")


def compare_with_template(board, progress_msg=None):
    """Порівнює з шаблоном (ОПТИМІЗОВАНО!)"""
    if not os.path.exists(TEMPLATE_FILE):
        return None, None
    
    logger.info("📊 Початок порівняння...")
    
    tmpl = PIL.Image.open(TEMPLATE_FILE).convert("RGBA")
    tw, th = tmpl.size
    
    if board.size != (tw, th):
        board = board.crop((0, 0, min(tw, board.size[0]), min(th, board.size[1])))
        if board.size != (tw, th):
            temp = PIL.Image.new('RGBA', (tw, th), (0, 0, 0, 0))
            temp.paste(board, (0, 0))
            board = temp
    
    logger.info("📊 Конвертую в numpy...")
    t_arr = np.array(tmpl, dtype=np.uint8)
    b_arr = np.array(board, dtype=np.uint8)
    
    logger.info("📊 Обчислюю маску шаблону...")
    template_mask = t_arr[:, :, 3] > 10
    total = int(np.sum(template_mask))
    
    if total == 0:
        return {"percent": 100, "errors": 0, "total": 0, "correct": 0}, tmpl
    
    logger.info(f"📊 Всього пікселів: {total:,}")
    logger.info("📊 Обчислюю різницю кольорів...")
    
    # ОПТИМІЗАЦІЯ: обчислюємо тільки де є маска
    diff_r = np.abs(t_arr[:,:,0].astype(np.int16) - b_arr[:,:,0].astype(np.int16))
    diff_g = np.abs(t_arr[:,:,1].astype(np.int16) - b_arr[:,:,1].astype(np.int16))
    diff_b = np.abs(t_arr[:,:,2].astype(np.int16) - b_arr[:,:,2].astype(np.int16))
    
    logger.info("📊 Перевіряю співпадіння...")
    color_match = (diff_r <= 20) & (diff_g <= 20) & (diff_b <= 20)
    board_mask = b_arr[:, :, 3] > 10
    
    logger.info("📊 Обчислюю правильні та помилки...")
    correct_mask = template_mask & board_mask & color_match
    errors_mask = template_mask & (~board_mask | ~color_match)
    
    correct_count = int(np.sum(correct_mask))
    err_count = int(np.sum(errors_mask))
    percent = (correct_count / total * 100) if total > 0 else 0
    
    logger.info(f"📊 Результат: {correct_count:,}/{total:,} ({percent:.1f}%)")
    logger.info("📊 Створюю overlay...")
    
    # ОПТИМІЗАЦІЯ: створюємо overlay швидше
    overlay = t_arr.copy()
    overlay[errors_mask] = [255, 0, 0, 255]
    
    result = {
        "percent": percent,
        "errors": err_count,
        "correct": correct_count,
        "total": total
    }
    
    logger.info("📊 Завершено!")
    
    return result, PIL.Image.fromarray(overlay)

# ======================================================================
# 🤖 БОТ
# ======================================================================

UPLOAD_WAITING = 1

async def start_cmd(u: Update, c):
    await u.message.reply_text(
        "🎨 **UkrLirn Monitor Bot**\n\n"
        "**Шаблон:**\n"
        "• `/upload` — завантажити\n"
        "• `/get` — скачати шаблон\n"
        "• `/set_coords X Y` — координати\n"
        "• `/check` — прогрес\n\n"
        "**Гравці:**\n"
        "• `/connect <нік>` — прив'язати\n"
        "• `/profile [нік]` — профіль\n\n"
        "**Медалі:**\n"
        "• `/madd <назва> <1-10>` (у відповідь)\n"
        "• `/mdel <номер>` (у відповідь)\n\n"
        "**Інше:**\n"
        "• `/site <назва>` — сайт\n"
        "• `/status` — налаштування",
        parse_mode="Markdown"
    )


async def status_cmd(u: Update, c):
    x, y = state.get("coords", [0, 0])
    has_template = os.path.exists(TEMPLATE_FILE)
    
    await u.message.reply_text(
        f"⚙️ **Налаштування:**\n\n"
        f"🌐 Сайт: `{state['site']}`\n"
        f"📐 Шаблон: {'✅' if has_template else '❌'}\n"
        f"📍 Координати: `{x}_{y}`\n"
        f"🏰 Фракція ID: {FACTION_ID}\n"
        f"⚡ Метод: CHUNKS (BMP)",
        parse_mode="Markdown"
    )


async def get_template_cmd(u: Update, c):
    """Отримати шаблон"""
    if not os.path.exists(TEMPLATE_FILE):
        return await u.message.reply_text("❌ Шаблон відсутній!")
    
    x, y = state.get("coords", [0, 0])
    coords_str = f"{x}_{y}"
    
    img = PIL.Image.open(TEMPLATE_FILE)
    caption = (
        f"📐 **Шаблон UkrLirn**\n\n"
        f"📍 Координати: `{coords_str}`\n"
        f"📏 Розмір: `{img.size[0]}x{img.size[1]}` px\n"
        f"🌐 Сайт: {state['site']}"
    )
    
    with open(TEMPLATE_FILE, "rb") as f:
        await u.message.reply_document(
            document=f,
            caption=caption,
            parse_mode="Markdown",
            filename="ukrlirn_template.png"
        )


async def check_cmd(u: Update, c):
    """Перевірка прогресу"""
    if not os.path.exists(TEMPLATE_FILE):
        return await u.message.reply_text("❌ Завантаж шаблон: `/upload`", parse_mode="Markdown")
    
    msg = await u.message.reply_text("⏳ Підготовка...")
    
    try:
        with PIL.Image.open(TEMPLATE_FILE) as tmpl:
            w, h = tmpl.size
        
        x, y = state["coords"]
        logger.info(f"🎬 Перевірка: {state['site']}, ({x},{y}), {w}x{h}")
        
        # Завантажуємо з прогрес-баром
        board = await get_map_area(state["site"], x, y, w, h, progress_msg=msg)
        
        await msg.edit_text("⏳ Порівнюю з шаблоном...")
        
        # Запускаємо порівняння в executor щоб не блокувати
        loop = asyncio.get_event_loop()
        result, overlay = await loop.run_in_executor(None, compare_with_template, board, msg)
        
        if result:
            await msg.edit_text("⏳ Створюю звіт...")
            
            bio = io.BytesIO()
            overlay.save(bio, 'PNG')
            bio.seek(0)
            
            coords_str = f"{x}_{y}"
            caption = (
                f"📊 **Прогрес UkrLirn**\n\n"
                f"🌐 Сайт: {state['site']}\n"
                f"📍 Координати: `{coords_str}`\n"
                f"🎯 Всього: `{result['total']:,}` px\n"
                f"✅ Правильно: `{result['correct']:,}` px\n"
                f"❌ Помилок: `{result['errors']:,}` px\n\n"
                f"📈 **Готовність: {result['percent']:.2f}%**"
            )
            
            await u.message.reply_document(
                document=bio,
                caption=caption,
                parse_mode="Markdown",
                filename="progress.png"
            )
            await msg.delete()
        else:
            await msg.edit_text("❌ Помилка порівняння")
            
    except Exception as e:
        logger.error(f"Помилка check: {e}", exc_info=True)
        await msg.edit_text(f"❌ Помилка: `{str(e)}`", parse_mode="Markdown")


async def debug_cmd(u: Update, c):
    """Debug вигляд"""
    if not os.path.exists(TEMPLATE_FILE):
        return await u.message.reply_text("❌ Потрібен шаблон")
    
    msg = await u.message.reply_text("👁️ Підготовка...")
    
    try:
        with PIL.Image.open(TEMPLATE_FILE) as tmpl:
            w, h = tmpl.size
        
        x, y = state["coords"]
        board = await get_map_area(state["site"], x, y, w, h, progress_msg=msg)
        
        await msg.edit_text("⏳ Створюю зображення...")
        
        bio = io.BytesIO()
        board.save(bio, 'PNG')
        bio.seek(0)
        
        await u.message.reply_photo(bio, caption=f"🗺️ Debug\n{state['site']} ({x}_{y})")
        await msg.delete()
        
    except Exception as e:
        logger.error(f"Debug error: {e}", exc_info=True)
        await msg.edit_text(f"❌ Помилка: {e}")


async def set_coords_cmd(u: Update, c):
    """Встановити координати"""
    try:
        if len(c.args) != 2:
            raise ValueError
        x, y = int(c.args[0]), int(c.args[1])
        state["coords"] = [x, y]
        save_state()
        await u.message.reply_text(f"✅ Координати: `{x}_{y}`", parse_mode="Markdown")
    except:
        await u.message.reply_text("⚠️ Формат: `/set_coords X Y`", parse_mode="Markdown")


async def set_site_cmd(u: Update, c):
    """Змінити сайт"""
    if c.args and c.args[0] in SITES:
        state["site"] = c.args[0]
        save_state()
        await u.message.reply_text(f"✅ Сайт: **{state['site']}**", parse_mode="Markdown")
    else:
        sites = ", ".join(SITES.keys())
        await u.message.reply_text(f"⚠️ Доступні: {sites}")


async def connect_cmd(u: Update, c):
    """Прив'язати профіль"""
    if not c.args:
        return await u.message.reply_text("⚠️ `/connect <нік>`", parse_mode="Markdown")
    
    nick = " ".join(c.args)
    user_id = str(u.effective_user.id)
    state["user_links"][user_id] = nick
    save_state()
    await u.message.reply_text(f"✅ Прив'язано: **{nick}**", parse_mode="Markdown")


async def profile_cmd(u: Update, c):
    """Профіль гравця"""
    user_id = str(u.effective_user.id)
    
    if c.args:
        nick = " ".join(c.args)
        target_id = None
        for uid, n in state["user_links"].items():
            if n.lower() == nick.lower():
                target_id = uid
                break
    else:
        nick = state["user_links"].get(user_id)
        target_id = user_id
        
        if not nick:
            return await u.message.reply_text("⚠️ `/connect <нік>`", parse_mode="Markdown")
    
    msg = await u.message.reply_text("🔍 Завантажую...")
    
    try:
        faction_data = await fetch_faction_data(state["site"])
        if not faction_data:
            return await msg.edit_text("❌ Не вдалось завантажити дані фракції")
        
        found = None
        for member in faction_data.get("members", []):
            if member.get("name", "").lower() == nick.lower():
                found = member
                break
        
        if not found:
            return await msg.edit_text(f"❌ **{nick}** не знайдений", parse_mode="Markdown")
        
        pixels = found.get("totalPixels", 0)
        daily = found.get("dailyPixels", 0)
        role = found.get("role", "member")
        
        medals_text = ""
        if target_id and target_id in state.get("medals", {}):
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


async def add_medal_cmd(u: Update, c):
    """Додати медаль"""
    if not u.message.reply_to_message:
        return await u.message.reply_text(
            "⚠️ Відповідай на повідомлення!\n`/madd <назва> <1-10>`",
            parse_mode="Markdown"
        )
    
    if not c.args or len(c.args) < 2:
        return await u.message.reply_text(
            "⚠️ `/madd <назва> <вага>`\n\nПриклад: `/madd Художник 10`",
            parse_mode="Markdown"
        )
    
    try:
        weight = int(c.args[-1])
        if weight < 1 or weight > 10:
            raise ValueError
        name = " ".join(c.args[:-1])
    except ValueError:
        return await u.message.reply_text("❌ Вага 1-10!")
    
    target_id = str(u.message.reply_to_message.from_user.id)
    
    if "medals" not in state:
        state["medals"] = {}
    if target_id not in state["medals"]:
        state["medals"][target_id] = []
    
    state["medals"][target_id].append({
        "name": name,
        "weight": weight,
        "date": datetime.now().strftime("%Y-%m-%d")
    })
    save_state()
    
    stars = "⭐" * weight
    await u.message.reply_text(
        f"✅ Медаль додано!\n\n🏅 **{name}** {stars}",
        parse_mode="Markdown"
    )


async def del_medal_cmd(u: Update, c):
    """Видалити медаль"""
    if not u.message.reply_to_message:
        return await u.message.reply_text(
            "⚠️ Відповідай на повідомлення!\n`/mdel <номер>`",
            parse_mode="Markdown"
        )
    
    if not c.args or len(c.args) != 1:
        return await u.message.reply_text("⚠️ `/mdel <номер>`", parse_mode="Markdown")
    
    try:
        index = int(c.args[0]) - 1
    except ValueError:
        return await u.message.reply_text("❌ Номер має бути числом!")
    
    target_id = str(u.message.reply_to_message.from_user.id)
    
    if target_id not in state.get("medals", {}) or not state["medals"][target_id]:
        return await u.message.reply_text("❌ Немає медалей!")
    
    if index < 0 or index >= len(state["medals"][target_id]):
        return await u.message.reply_text("❌ Медалі з таким номером не існує!")
    
    removed = state["medals"][target_id].pop(index)
    save_state()
    
    await u.message.reply_text(
        f"✅ Видалено: 🏅 {removed['name']}",
        parse_mode="Markdown"
    )


async def upload_start(u: Update, c):
    await u.message.reply_text("📤 Надішли PNG файл шаблону:")
    return UPLOAD_WAITING


async def upload_file(u: Update, c):
    doc = u.message.document
    if not doc or not doc.file_name.lower().endswith('.png'):
        await u.message.reply_text("❌ Потрібен PNG!")
        return ConversationHandler.END
    
    file = await doc.get_file()
    await file.download_to_drive(TEMPLATE_FILE)
    
    img = PIL.Image.open(TEMPLATE_FILE)
    await u.message.reply_text(
        f"✅ Шаблон завантажено: `{img.size[0]}x{img.size[1]}` px",
        parse_mode="Markdown"
    )
    return ConversationHandler.END


async def cancel_upload(u: Update, c):
    await u.message.reply_text("❌ Скасовано")
    return ConversationHandler.END


# ======================================================================
# 🚀 ЗАПУСК
# ======================================================================

def main():
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN не встановлено!")
        sys.exit(1)
    
    load_state()
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    upload_conv = ConversationHandler(
        entry_points=[CommandHandler("upload", upload_start)],
        states={
            UPLOAD_WAITING: [MessageHandler(filters.Document.ALL, upload_file)]
        },
        fallbacks=[CommandHandler("cancel", cancel_upload)]
    )
    
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("check", check_cmd))
    app.add_handler(CommandHandler("debug", debug_cmd))
    app.add_handler(CommandHandler("get", get_template_cmd))
    app.add_handler(CommandHandler("set_coords", set_coords_cmd))
    app.add_handler(CommandHandler("site", set_site_cmd))
    app.add_handler(CommandHandler("connect", connect_cmd))
    app.add_handler(CommandHandler("profile", profile_cmd))
    app.add_handler(CommandHandler("madd", add_medal_cmd))
    app.add_handler(CommandHandler("mdel", del_medal_cmd))
    app.add_handler(upload_conv)
    
    logger.info("=" * 60)
    logger.info("🤖 UkrLirn Monitor Bot запущено!")
    logger.info("=" * 60)
    logger.info(f"⚡ Метод: CHUNKS (BMP RAW)")
    logger.info(f"🌐 Сайт: {state['site']}")
    logger.info(f"🏰 Фракція ID: {FACTION_ID}")
    logger.info("=" * 60)
    
    app.run_polling()


if __name__ == "__main__":
    main()
