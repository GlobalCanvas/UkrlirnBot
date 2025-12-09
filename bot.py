#!/usr/bin/env python3
# bot_merged.py - UkrLirn Monitor Bot (Strict 256x256 Version)

import sys
import logging
import asyncio
import json
import os
import io
from io import BytesIO
from datetime import datetime

# --- ПРОВЕРКА БИБЛИОТЕК ---
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
    print("="*40)
    print("❌ ОШИБКА: Не установлены библиотеки!")
    print(f"Не найдено: {e.name}")
    print("Выполните команду:")
    print("pip install python-telegram-bot aiohttp Pillow numpy")
    print("="*40)
    sys.exit(1)

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ======================================================================
# ⚙️ КОНФИГУРАЦИЯ
# ======================================================================

# 👇 ВСТАВЬ СВОЙ ТОКЕН НИЖЕ
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8133267244:AAEimjL3_gSTWiYV7bglcyrqGA2woQykDZo")

COLOR_TOLERANCE = 20
STATE_FILE = "state.json"
FACTION_API_URL = "https://canvaspix.fun/api/faction/133"

# 🔥 СТРОГО 256 ПИКСЕЛЕЙ 🔥
STRICT_TILE_SIZE = 256

# ======================================================================
# 🧩 ДВИЖОК (LIRN ENGINE - 256px MODE)
# ======================================================================

SITES = {
    "pixmap": {
        "url": "https://pixmap.fun",
        "tile_url": "https://pixmap.fun/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://pixmap.fun/api/me"
    },
    "canvaspix": {
        "url": "https://canvaspix.fun",
        "tile_url": "https://canvaspix.fun/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://canvaspix.fun/api/me"
    },
    "pixelya": {
        "url": "https://pixelya.fun",
        "tile_url": "https://pixelya.fun/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://pixelya.fun/api/me"
    },
    "globepixel": {
        "url": "https://globepixel.net",
        "tile_url": "https://globepixel.net/tiles/{canvas_id}/{zoom}/{tx}/{ty}.webp",
        "api_me": "https://globepixel.net/api/me"
    }
}

USER_AGENT = "UkrLirn Monitor Bot 1.0 (Strict 256)"
CURRENT_SITE = "canvaspix"

def set_site(site_name: str) -> bool:
    global CURRENT_SITE
    if site_name.lower() in SITES:
        CURRENT_SITE = site_name.lower()
        return True
    return False

def get_current_site():
    return SITES[CURRENT_SITE]

async def fetch_api_me():
    site = get_current_site()
    url = site["api_me"]
    headers = {'User-Agent': USER_AGENT}
    
    async with aiohttp.ClientSession() as session:
        for _ in range(3):
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.json()
            except Exception as e:
                logger.warning(f"Ошибка загрузки API: {e}")
                await asyncio.sleep(2)
    return None

async def fetch_tile(session, url, offx, offy, image, needed=False):
    """
    Загружает тайл и СТРОГО приводит его к 256x256
    """
    headers = {'User-Agent': USER_AGENT}
    for _ in range(3):
        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 404:
                    if needed:
                        # Пустой прозрачный квадрат 256x256
                        empty = PIL.Image.new('RGBA', (STRICT_TILE_SIZE, STRICT_TILE_SIZE), (0, 0, 0, 0))
                        image.paste(empty, (offx, offy))
                    return True
                
                if resp.status == 200:
                    data = await resp.read()
                    if data:
                        tile = PIL.Image.open(io.BytesIO(data)).convert('RGBA')
                        
                        # 🔥 ПРИНУДИТЕЛЬНЫЙ РЕСАЙЗ В 256x256 🔥
                        # Если сервер прислал 1024, мы жмем его в 256, чтобы координаты не улетели
                        if tile.size != (STRICT_TILE_SIZE, STRICT_TILE_SIZE):
                            tile = tile.resize((STRICT_TILE_SIZE, STRICT_TILE_SIZE), PIL.Image.NEAREST)
                        
                        # Вставляем с учетом прозрачности самого тайла
                        image.paste(tile, (offx, offy), tile)
                        return True
        except:
            await asyncio.sleep(0.5)
    return False

async def get_canvas_area(canvas_id, x, y, width, height, canvas_size=32768):
    """
    Загрузка области с шагом сетки 256 пикселей
    """
    site = get_current_site()
    offset = int(-canvas_size / 2)
    
    # Сетка СТРОГО 256 (как в historyDownload.py)
    tile_size = STRICT_TILE_SIZE
    
    xc = (x - offset) // tile_size
    wc = (x + width - offset) // tile_size
    yc = (y - offset) // tile_size
    hc = (y + height - offset) // tile_size
    
    logger.info(f"Загрузка (256px mode): тайлы X[{xc}..{wc}], Y[{yc}..{hc}]")
    
    result = PIL.Image.new('RGBA', (width, height), (0, 0, 0, 0))
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        for iy in range(yc, hc + 1):
            for ix in range(xc, wc + 1):
                # Формируем URL
                # Zoom 7 обычно стандарт, но сервер может вернуть 1024.
                # Функция fetch_tile сама ужмет это в 256.
                url = site["tile_url"].format(canvas_id=canvas_id, zoom=7, tx=ix, ty=iy)
                
                # Расчет места вставки
                offx = ix * tile_size + offset - x
                offy = iy * tile_size + offset - y
                
                tasks.append(fetch_tile(session, url, offx, offy, result, needed=True))
        
        await asyncio.gather(*tasks)
    
    return result

def compare_with_template(template, board, tolerance=20):
    """Сравнение с учетом прозрачности"""
    tw, th = template.size
    if board.size != (tw, th):
        board = board.crop((0, 0, tw, th))
    
    t = np.array(template, dtype=np.float32)
    b = np.array(board, dtype=np.float32)
    
    # Маска шаблона (что должно быть нарисовано)
    mask_template = t[..., 3] > 10
    total = int(mask_template.sum())
    
    if total == 0:
        return {"total": 0, "placed": 0, "remaining": 0, "percent": 100.0}

    # Маска доски (что реально есть на доске)
    mask_board = b[..., 3] > 10
    
    # Разница цветов RGB
    diff = np.sqrt(np.sum((b[..., :3] - t[..., :3]) ** 2, axis=-1))
    within_color = diff <= tolerance
    
    # Логика:
    # 1. Пиксель есть в шаблоне
    # 2. Пиксель на доске НЕ прозрачный (mask_board)
    # 3. Цвет совпадает (within_color)
    placed = int((mask_template & mask_board & within_color).sum())
    
    remaining = total - placed
    percent = (placed / total * 100.0) if total > 0 else 100.0
    
    return {"total": total, "placed": placed, "remaining": remaining, "percent": percent}

def create_overlay(template, board, tolerance=20, output_path=None):
    if not output_path: return None
    
    tw, th = template.size
    if board.size != (tw, th):
        board = board.crop((0, 0, tw, th))
    
    t = np.array(template, dtype=np.float32)
    b = np.array(board, dtype=np.float32)
    
    mask_template = t[..., 3] > 10
    mask_board = b[..., 3] > 10
    diff = np.sqrt(np.sum((b[..., :3] - t[..., :3]) ** 2, axis=-1))
    within_color = diff <= tolerance
    
    output = b.copy().astype(np.uint8)
    
    # Красим ошибки красным
    # Ошибка = (есть в шаблоне) И ( (нет на доске) ИЛИ (цвет не тот) )
    bad = (mask_template & (~within_color | ~mask_board))
    output[bad] = [255, 0, 0, 255]
    
    PIL.Image.fromarray(output.astype(np.uint8)).save(output_path)
    return output_path

async def process_lirn_template(template_path, x, y, canvas_id=0, tolerance=20, overlay_path=None):
    template = PIL.Image.open(template_path).convert("RGBA")
    width, height = template.size
    
    api_me = await fetch_api_me()
    canvas_size = 32768
    
    if api_me and 'canvases' in api_me:
        info = api_me['canvases'].get(str(canvas_id))
        if info:
            canvas_size = info.get('size', 32768)
            # Игнорируем tileSize из API, используем строго 256
            
    board = await get_canvas_area(canvas_id, x, y, width, height, canvas_size)
    result = compare_with_template(template, board, tolerance)
    
    if overlay_path:
        create_overlay(template, board, tolerance, overlay_path)
        
    return result

# ======================================================================
# 🤖 БОТ
# ======================================================================

UPLOAD_TEMPLATE_WAITING = 1
LIRN_TEMPLATE = {"file": "templates/lirn.png"}
state = {"user_links": {}, "medals": {}, "lirn_coords": [0, 0], "current_site": "canvaspix"}

def load_state():
    global state
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state.update(json.load(f))
                set_site(state.get("current_site", "canvaspix"))
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")

def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения состояния: {e}")

# --- КОМАНДЫ ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 **Бот работает (Режим 256px)!**\n\n"
        "1. `/upload_template` - загрузить PNG (админ)\n"
        "2. `/set_coords X Y` - задать координаты (админ)\n"
        "3. `/check` - проверить прогрес\n"
        "4. `/connect Nickname` - привязать ник\n"
        "5. `/profile` - статистика",
        parse_mode="Markdown"
    )

async def set_coords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) != 2:
        await update.message.reply_text("⚠️ Пример: `/set_coords 100 -200`", parse_mode="Markdown")
        return
    try:
        x, y = int(context.args[0]), int(context.args[1])
        state["lirn_coords"] = [x, y]
        save_state()
        await update.message.reply_text(f"✅ Координаты: {x}, {y}")
    except ValueError:
        await update.message.reply_text("❌ Координати должны быть числами!")

async def upload_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Пришли мне PNG файл шаблона.")
    return UPLOAD_TEMPLATE_WAITING

async def upload_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith('.png'):
        await update.message.reply_text("❌ Это не PNG!")
        return ConversationHandler.END
    
    file = await doc.get_file()
    os.makedirs("templates", exist_ok=True)
    await file.download_to_drive(LIRN_TEMPLATE["file"])
    
    await update.message.reply_text("✅ Шаблон сохранен! Теперь задай координаты через /set_coords")
    return ConversationHandler.END

async def cancel_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отмена.")
    return ConversationHandler.END

async def check_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(LIRN_TEMPLATE["file"]):
        await update.message.reply_text("❌ Сначала загрузи шаблон: /upload_template")
        return

    x, y = state.get("lirn_coords", [0, 0])
    status_msg = await update.message.reply_text(f"⏳ Проверяю ({x}, {y})...")
    
    try:
        os.makedirs("progress", exist_ok=True)
        res = await process_lirn_template(
            LIRN_TEMPLATE["file"], x, y, 
            tolerance=COLOR_TOLERANCE, 
            overlay_path="progress/overlay.png"
        )
        
        caption = (
            f"📊 **Прогресс (256px)**\n"
            f"Всего: `{res['total']}` px\n"
            f"✅ Готово: `{res['placed']}`\n"
            f"❌ Осталось: `{res['remaining']}`\n"
            f"📈 **{res['percent']:.2f}%**"
        )
        
        with open("progress/overlay.png", "rb") as f:
            await update.message.reply_document(document=f, caption=caption, parse_mode="Markdown")
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"Check error: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")

async def connect_player(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: return
    nick = " ".join(context.args)
    state["user_links"][str(update.effective_user.id)] = nick
    save_state()
    await update.message.reply_text(f"✅ Привязан ник: **{nick}**", parse_mode="Markdown")

async def get_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    nick = state["user_links"].get(user_id)
    if not nick:
        await update.message.reply_text("⚠️ Сначала привяжи ник: `/connect Nickname`", parse_mode="Markdown")
        return
    
    msg = await update.message.reply_text("🔍 Ищу данные...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(FACTION_API_URL, headers={'User-Agent': USER_AGENT}, timeout=10) as resp:
                if resp.status != 200:
                    await msg.edit_text(f"❌ Ошибка API: {resp.status}")
                    return
                data = await resp.json()
    except Exception as e:
        await msg.edit_text(f"❌ Ошибка соединения: {e}")
        return

    faction = data.get("faction", data)
    found = None
    for member in faction.get("members", []):
        if member.get("User", {}).get("name", "").lower() == nick.lower():
            found = member
            break
            
    if found:
        u = found["User"]
        txt = (
            f"👤 **{u['name']}**\n"
            f"🎨 Пикселей: `{u['totalPixels']}`\n"
            f"📅 Вступил: {found.get('joinedAt', 'Unknown').split('T')[0]}"
        )
        await msg.edit_text(txt, parse_mode="Markdown")
    else:
        await msg.edit_text(f"❌ Игрок **{nick}** не найден во фракции.", parse_mode="Markdown")

if __name__ == "__main__":
    if "ВСТАВЬ_СЮДА" in BOT_TOKEN:
        print("❌ ОШИБКА: Вы не вставили токен бота в код!")
        sys.exit(1)

    load_state()
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("set_coords", set_coords))
    app.add_handler(CommandHandler("check", check_progress))
    app.add_handler(CommandHandler("connect", connect_player))
    app.add_handler(CommandHandler("profile", get_profile))
    
    conv = ConversationHandler(
        entry_points=[CommandHandler("upload_template", upload_start)],
        states={UPLOAD_TEMPLATE_WAITING: [MessageHandler(filters.Document.ALL, upload_file)]},
        fallbacks=[CommandHandler("cancel", cancel_upload)]
    )
    app.add_handler(conv)

    print("✅ Бот запущен (Режим 256px)! Нажмите Ctrl+C для выхода.")
    app.run_polling()
