#!/usr/bin/env python3
# bot.py - UkrLirn Monitor Bot (FIXED: Coords + Cookies)

import sys
import logging
import asyncio
import json
import os
import io
import math
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

BOT_TOKEN = os.environ.get("BOT_TOKEN", "8133267244:AAFPj7GcUhgUPUiuAxM9afwQFoSsB5hEtUc")

# --- 👇 СЮДА ВСТАВЛЯЙ РАЗНЫЕ КУКИ 👇 ---
# Куки для PIXMAP (обычно длинный)
COOKIE_PIXMAP = os.environ.get("COOKIE_PIXMAP", "s%3AS2qBqqlzYPCWST-OalOz6svoEoTYQIi9.%2BL0JZVKMRNrHr9eQ8WAuf4D9MdthKJP3pHCrqliUmZs")

# Куки для PIXELYA (вставь сюда свой куки от pixelya)
COOKIE_PIXELYA = os.environ.get("COOKIE_PIXELYA", "pixelya.session=s%3AMiTKf-27ZLGgiV8Xt13qhsg6tGliAuLx.p64kN9RRkoAMMecownmgM1SiJmO67d")

COLOR_TOLERANCE = 10
STATE_FILE = "state.json"
CHUNK_SIZE = 256
MAX_CONCURRENT_CHUNKS = 25

# НАЛАШТУВАННЯ САЙТІВ
SITES = {
    "pixmap": {
        "url": "https://pixmap.fun",
        "chunks_url": "https://pixmap.fun/chunks/{canvas_id}/{ix}/{iy}.bmp",
        "api_me": "https://pixmap.fun/api/me",
        "default_size": 32768,    # Розмір 32k
        "cookie": COOKIE_PIXMAP
    },
    "pixelya": {
        "url": "https://pixelya.fun",
        "chunks_url": "https://pixelya.fun/chunks/{canvas_id}/{ix}/{iy}.bmp",
        "api_me": "https://pixelya.fun/api/me",
        "default_size": 65536,    # Розмір 65k (Важливо!)
        "cookie": COOKIE_PIXELYA
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

def get_headers():
    """Генерує заголовки з правильним куки для поточного сайту"""
    site_config = get_current_site()
    cookie_val = site_config.get("cookie", "")
    
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Cookie': f'cpn.session={cookie_val}; plang=ru',
        'Accept': 'application/json, image/bmp, */*'
    }

# ======================================================================
# 🎨 ДВИЖОК
# ======================================================================

class CanvasManager:
    def __init__(self, x, y, width, height, canvas_size):
        self.req_x = x
        self.req_y = y
        self.width = width
        self.height = height
        self.canvas_size = canvas_size
        
        # Центр координат залежить від розміру полотна!
        # Pixmap (32k) -> -16384
        # Pixelya (65k) -> -32768
        self.offset = -(canvas_size // 2) 
        
        self.image = PIL.Image.new('RGBA', (width, height), (0, 0, 0, 0))

    def paste_chunk(self, chunk_img, ix, iy):
        chunk_x_global = ix * CHUNK_SIZE + self.offset
        chunk_y_global = iy * CHUNK_SIZE + self.offset
        
        paste_x = chunk_x_global - self.req_x
        paste_y = chunk_y_global - self.req_y
        
        self.image.paste(chunk_img, (paste_x, paste_y))

async def fetch_api_me():
    site = get_current_site()
    async with aiohttp.ClientSession() as session:
        try:
            # Використовуємо get_headers() замість статики
            async with session.get(site["api_me"], headers=get_headers(), timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error(f"API Error ({CURRENT_SITE}): {resp.status}")
        except Exception as e:
            logger.error(f"Connection Error: {e}")
    return None

async def fetch_chunk(session, url, ix, iy, sem):
    async with sem:
        for attempt in range(3):
            try:
                # Використовуємо get_headers()
                async with session.get(url, headers=get_headers(), timeout=10) as resp:
                    if resp.status == 404:
                        return None
                    elif resp.status == 200:
                        data = await resp.read()
                        try:
                            img = PIL.Image.open(io.BytesIO(data)).convert("RGBA")
                            return img
                        except:
                            return None
                    else:
                        pass
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return None

async def download_area(x, y, width, height, canvas_id=0):
    site_config = get_current_site()
    
    # 1. Визначаємо розмір (спочатку дефолтний)
    canvas_size = site_config["default_size"]
    
    # 2. Пробуємо отримати точний розмір з API
    api_data = await fetch_api_me()
    if api_data and 'canvases' in api_data:
        c_info = api_data['canvases'].get(str(canvas_id))
        if c_info:
            canvas_size = c_info.get('size', canvas_size)
    
    logger.info(f"🌍 Site: {CURRENT_SITE.upper()} | Size: {canvas_size}")
    
    manager = CanvasManager(x, y, width, height, canvas_size)
    offset = manager.offset
    
    start_cx = (x - offset) // CHUNK_SIZE
    end_cx = (x + width - 1 - offset) // CHUNK_SIZE
    start_cy = (y - offset) // CHUNK_SIZE
    end_cy = (y + height - 1 - offset) // CHUNK_SIZE
    
    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
    
    async with aiohttp.ClientSession() as session:
        for cy in range(start_cy, end_cy + 1):
            for cx in range(start_cx, end_cx + 1):
                url = site_config["chunks_url"].format(canvas_id=canvas_id, ix=cx, iy=cy)
                task = asyncio.create_task(fetch_chunk_worker(session, url, cx, cy, sem, manager))
                tasks.append(task)
        
        if tasks:
            logger.info(f"🚀 Завантаження {len(tasks)} чанків...")
            await asyncio.gather(*tasks)
    
    return manager.image

async def fetch_chunk_worker(session, url, cx, cy, sem, manager):
    img = await fetch_chunk(session, url, cx, cy, sem)
    if img:
        manager.paste_chunk(img, cx, cy)

def compare_images(template_path, board_img, tolerance=10):
    template = PIL.Image.open(template_path).convert("RGBA")
    if board_img.size != template.size:
        board_img = board_img.crop((0, 0, template.size[0], template.size[1]))
    
    t_arr = np.array(template)
    b_arr = np.array(board_img)
    
    t_mask = t_arr[:, :, 3] > 10
    total_pixels = np.sum(t_mask)
    
    if total_pixels == 0:
        return {"total": 0, "placed": 0, "remaining": 0, "percent": 100.0}, board_img

    diff = np.abs(t_arr[:, :, :3].astype(int) - b_arr[:, :, :3].astype(int))
    dist = np.linalg.norm(diff, axis=2)
    
    match_mask = dist <= tolerance
    correct_pixels = np.logical_and(t_mask, match_mask)
    placed_count = np.sum(correct_pixels)
    
    error_mask = np.logical_and(t_mask, ~match_mask)
    
    overlay_arr = np.array(board_img)
    overlay_arr[error_mask] = [255, 0, 0, 255] # Червоний колір помилок
    
    final_overlay = PIL.Image.fromarray(overlay_arr)
    
    return {
        "total": int(total_pixels),
        "placed": int(placed_count),
        "remaining": int(total_pixels - placed_count),
        "percent": (placed_count / total_pixels) * 100.0
    }, final_overlay

# ======================================================================
# 🤖 БОТ
# ======================================================================

LIRN_TEMPLATE = {"file": "templates/lirn.png"}
state = {
    "lirn_coords": [0, 0],
    "current_site": "pixmap"
}

def load_state():
    global state
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                state.update(saved)
                # Відновлюємо сайт при запуску
                set_site(state.get("current_site", "pixmap"))
        except Exception:
            pass

def save_state():
    state["current_site"] = CURRENT_SITE
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎨 **UkrLirn Fix v3.0**\n\n"
        "Команди:\n"
        "/upload_template - завантажити шаблон\n"
        "/set_coords X Y - координати\n"
        "/check - перевірити прогрес\n"
        "/site pixmap (або pixelya) - змінити сайт\n"
        "/status - поточні налаштування",
        parse_mode="Markdown"
    )

async def change_site_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"Поточний сайт: {CURRENT_SITE}\nПриклад: `/site pixelya`", parse_mode="Markdown")
        return
    
    new_site = context.args[0].lower()
    if set_site(new_site):
        save_state()
        size = get_current_site()["default_size"]
        await update.message.reply_text(f"✅ Сайт змінено на: **{new_site.upper()}**\n📏 Базовий розмір: {size}x{size}", parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ Невідомий сайт. Доступні: pixmap, pixelya")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    site_conf = get_current_site()
    has_cookie = "✅ Є" if site_conf["cookie"] and "ТВОЙ_КУКИ" not in site_conf["cookie"] else "❌ Немає/Дефолтний"
    
    await update.message.reply_text(
        f"⚙️ **Status**\n"
        f"🌍 Site: {CURRENT_SITE.upper()}\n"
        f"📏 Size: {site_conf['default_size']}\n"
        f"🍪 Cookie: {has_cookie}\n"
        f"📍 Coords: {state['lirn_coords']}",
        parse_mode="Markdown"
    )

async def check_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(LIRN_TEMPLATE["file"]):
        await update.message.reply_text("❌ Немає шаблону! /upload_template")
        return

    x, y = state.get("lirn_coords", [0, 0])
    status_msg = await update.message.reply_text(f"⏳ Качаю з {CURRENT_SITE.upper()}...")

    try:
        tmpl = PIL.Image.open(LIRN_TEMPLATE["file"])
        w, h = tmpl.size
        
        board_img = await download_area(x, y, w, h)
        stats, overlay = compare_images(LIRN_TEMPLATE["file"], board_img, tolerance=COLOR_TOLERANCE)
        
        os.makedirs("progress", exist_ok=True)
        overlay.save("progress/overlay.png")
        
        caption = (
            f"📊 **Звіт {CURRENT_SITE.upper()}**\n"
            f"📍 {x}, {y}\n"
            f"✅ {stats['percent']:.2f}%\n"
            f"🟥 Помилок: {stats['remaining']}"
        )
        
        with open("progress/overlay.png", "rb") as f:
            await update.message.reply_document(f, caption=caption, parse_mode="Markdown")
        await status_msg.delete()

    except Exception as e:
        logger.error(f"Check failed: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Помилка: {e}")

async def set_coords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        x, y = int(context.args[0]), int(context.args[1])
        state["lirn_coords"] = [x, y]
        save_state()
        await update.message.reply_text(f"✅ Координати: {x}, {y}")
    except:
        await update.message.reply_text("Приклад: /set_coords 100 -200")

# --- UPLOAD ---
UPLOAD = 1
async def upload_start(update: Update, context):
    await update.message.reply_text("📤 Кидай PNG:")
    return UPLOAD

async def upload_file(update: Update, context):
    doc = update.message.document
    if not doc.file_name.lower().endswith('.png'):
        await update.message.reply_text("Тільки PNG!")
        return ConversationHandler.END
    os.makedirs("templates", exist_ok=True)
    f = await doc.get_file()
    await f.download_to_drive(LIRN_TEMPLATE["file"])
    await update.message.reply_text("✅ Прийнято!")
    return ConversationHandler.END

async def cancel(update: Update, context):
    await update.message.reply_text("❌")
    return ConversationHandler.END

def main():
    if not BOT_TOKEN:
        print("Встанови BOT_TOKEN")
        return
    load_state()
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("check", check_progress))
    app.add_handler(CommandHandler("set_coords", set_coords))
    app.add_handler(CommandHandler("site", change_site_cmd)) # Нова команда
    app.add_handler(CommandHandler("status", status_cmd))

    conv = ConversationHandler(
        entry_points=[CommandHandler("upload_template", upload_start)],
        states={UPLOAD: [MessageHandler(filters.Document.ALL, upload_file)]},
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    app.add_handler(conv)

    print(f"🚀 Bot started on {CURRENT_SITE}...")
    app.run_polling()

if __name__ == "__main__":
    main()
