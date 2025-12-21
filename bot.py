#!/usr/bin/env python3
# bot.py - UkrLirn Monitor Bot (ПРАВИЛЬНИЙ МЕТОД CHUNKS!)

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
AUTH_COOKIE = os.environ.get("AUTH_COOKIE", "s%3AS2qBqqlzYPCWST-OalOz6svoEoTYQIi9.%2BL0JZVKMRNrHr9eQ8WAuf4D9MdthKJP3pHCrqliUmZs")

API_HEADERS = {
    'User-Agent': 'UkrLirn Monitor Bot 1.0',
    'Cookie': f'cpn.session={AUTH_COOKIE}; plang=ru',
    'Accept': 'application/json'
}

COLOR_TOLERANCE = 20
STATE_FILE = "state.json"
CHUNK_SIZE = 256  # Розмір чанку

# ФРАКЦІЇ ДЛЯ КОЖНОГО САЙТУ
FACTION_IDS = {
    "pixmap": 530,    # UkrLirn | Pxl
    "pixelya": 359    # UkrLirn на pixelya
}

# ======================================================================
# 🌐 САЙТИ (ПРАВИЛЬНІ URL!)
# ======================================================================

SITES = {
    "pixmap": {
        "url": "https://pixmap.fun",
        "chunks_url": "https://pixmap.fun/chunks/{canvas_id}/{ix}/{iy}.bmp",
        "api_me": "https://pixmap.fun/api/me",
        "api_faction_list": "https://pixmap.fun/api/faction/list"
    },
    "pixelya": {
        "url": "https://pixelya.fun",
        "chunks_url": "https://pixelya.fun/chunks/{canvas_id}/{ix}/{iy}.bmp",
        "api_me": "https://pixelya.fun/api/me",
        "api_faction_list": "https://pixelya.fun/api/faction/list"
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
# 🎨 ДВИЖОК (CHUNKS METHOD - ПРАВИЛЬНО!)
# ======================================================================

class Matrix:
    """Матриця для зберігання пікселів (як в areaDownload.py)"""
    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.width = None
        self.height = None
        self.matrix = {}
        self.colors = {}

    def add_coords(self, x, y, w, h):
        if self.start_x is None or self.start_x > x:
            self.start_x = x
        if self.start_y is None or self.start_y > y:
            self.start_y = y
        
        end_x_a = x + w
        end_y_a = y + h
        
        if self.width is None or self.height is None:
            self.width = w
            self.height = h
        else:
            end_x_b = self.start_x + self.width
            end_y_b = self.start_y + self.height
            self.width = max(end_x_b, end_x_a) - self.start_x
            self.height = max(end_y_b, end_y_a) - self.start_y

    def set_pixel(self, x, y, color_index, color_rgb):
        if x >= self.start_x and x < (self.start_x + self.width) and y >= self.start_y and y < (self.start_y + self.height):
            if x not in self.matrix:
                self.matrix[x] = {}
            self.matrix[x][y] = color_rgb
            if color_index not in self.colors:
                self.colors[color_index] = color_rgb

    def create_image(self):
        """Створює PIL Image з матриці"""
        img = PIL.Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        pxls = img.load()
        
        for x in range(self.width):
            for y in range(self.height):
                try:
                    color = self.matrix[x + self.start_x][y + self.start_y]
                    pxls[x, y] = color
                except (IndexError, KeyError, AttributeError):
                    pass
        
        return img


async def fetch_api_me():
    """Отримує інфо про канваси"""
    site = get_current_site()
    url = site["api_me"]
    
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
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
    """Отримує дані фракції"""
    site = get_current_site()
    url = site["api_faction_list"]
    
    # Отримуємо ID фракції для поточного сайту
    faction_id = FACTION_IDS.get(CURRENT_SITE)
    if not faction_id:
        logger.error(f"Немає ID фракції для сайту {CURRENT_SITE}")
        return None
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=API_HEADERS, timeout=10) as resp:
                if resp.status == 200:
                    factions = await resp.json()
                    for faction in factions:
                        if faction.get("id") == faction_id:
                            logger.info(f"✅ Знайдено фракцію: {faction.get('name')} (ID: {faction_id})")
                            return faction
                    logger.warning(f"Фракцію {faction_id} не знайдено на {CURRENT_SITE}")
                else:
                    logger.warning(f"API faction/list: {resp.status}")
        except Exception as e:
            logger.error(f"Помилка faction API: {e}")
    return None


async def fetch_chunk(session, canvas_id, canvas_colors, canvasoffset, ix, iy, target_matrix):
    """
    ПРАВИЛЬНИЙ метод завантаження чанку (з areaDownload.py)
    Використовує /chunks/ замість /tiles/
    """
    site = get_current_site()
    url = site["chunks_url"].format(canvas_id=canvas_id, ix=ix, iy=iy)
    
    logger.info(f"🔄 Завантажую chunk [{ix},{iy}]: {url}")
    
    for attempt in range(2):
        try:
            async with session.get(url, headers=API_HEADERS, timeout=15) as resp:
                logger.info(f"📡 Chunk [{ix},{iy}] status: {resp.status}")
                data = await resp.read()
                logger.info(f"📦 Chunk [{ix},{iy}] size: {len(data)} bytes")
                
                # Offset як в areaDownload.py
                offset = int(-canvasoffset * canvasoffset / 2)
                off_x = ix * CHUNK_SIZE + offset
                off_y = iy * CHUNK_SIZE + offset
                
                # Якщо чанк порожній (404 або 0 bytes)
                if resp.status == 404 or len(data) == 0:
                    logger.info(f"⚪ Chunk [{ix},{iy}] порожній, заповнюю дефолтним кольором")
                    # Заповнюємо дефолтним кольором (індекс 0)
                    clr = canvas_colors[0] if 0 in canvas_colors else (0, 0, 0, 0)
                    for i in range(CHUNK_SIZE * CHUNK_SIZE):
                        tx = off_x + i % CHUNK_SIZE
                        ty = off_y + i // CHUNK_SIZE
                        target_matrix.set_pixel(tx, ty, 0, clr)
                else:
                    # Читаємо BMP дані
                    logger.info(f"✍️ Chunk [{ix},{iy}] парсинг {len(data)} байтів...")
                    i = 0
                    for b in data:
                        tx = off_x + i % CHUNK_SIZE
                        ty = off_y + i // CHUNK_SIZE
                        color_index = b & 0x7F  # Маскуємо старший біт
                        color_rgb = canvas_colors.get(color_index, (0, 0, 0, 255))
                        target_matrix.set_pixel(tx, ty, color_index, color_rgb)
                        i += 1
                    logger.info(f"✅ Chunk [{ix},{iy}] оброблено {i} пікселів")
                
                return True
                
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Timeout chunk [{ix},{iy}], спроба {attempt+1}/2")
        except Exception as e:
            logger.error(f"❌ Помилка chunk [{ix},{iy}]: {e}", exc_info=True)
        
        if attempt < 1:
            await asyncio.sleep(0.3)
    
    logger.error(f"💀 Failed chunk [{ix},{iy}] після 2 спроб")
    return False


async def get_canvas_area(canvas_id, canvas_info, x, y, width, height):
    """
    ПРАВИЛЬНИЙ метод завантаження області (з areaDownload.py)
    """
    logger.info(f"🎬 Початок завантаження області...")
    logger.info(f"📍 Координати: x={x}, y={y}, width={width}, height={height}")
    
    # Створюємо матрицю
    target_matrix = Matrix()
    target_matrix.add_coords(x, y, width, height)
    
    # Отримуємо кольори канвасу
    canvas_colors = {}
    colors_list = canvas_info.get('colors', [])
    logger.info(f"🎨 Кольорів у палітрі: {len(colors_list)}")
    
    for i, color in enumerate(colors_list):
        if len(color) == 3:
            canvas_colors[i] = tuple(color) + (255,)  # RGB -> RGBA
        else:
            canvas_colors[i] = tuple(color)
    
    # ПРАВИЛЬНИЙ offset (як в areaDownload.py)
    canvas_size = canvas_info.get('size', 32768)
    canvasoffset = math.pow(canvas_size, 0.5)  # sqrt(size)
    offset = int(-canvasoffset * canvasoffset / 2)  # -(size/2)
    
    logger.info(f"📏 Canvas size: {canvas_size}")
    logger.info(f"🔢 Canvas offset: {canvasoffset}")
    logger.info(f"🔢 Offset: {offset}")
    
    # Обчислюємо діапазон чанків
    xc = (x - offset) // CHUNK_SIZE
    wc = (x + width - offset) // CHUNK_SIZE
    yc = (y - offset) // CHUNK_SIZE
    hc = (y + height - offset) // CHUNK_SIZE
    
    logger.info(f"📐 Область: x={x}, y={y}, {width}x{height}")
    logger.info(f"🗺️ Чанки: X[{xc}..{wc}], Y[{yc}..{hc}]")
    logger.info(f"📦 Всього чанків: {(wc - xc + 1) * (hc - yc + 1)}")
    
    # Завантажуємо чанки паралельно
    tasks = []
    async with aiohttp.ClientSession() as session:
        for iy in range(yc, hc + 1):
            for ix in range(xc, wc + 1):
                tasks.append(fetch_chunk(session, canvas_id, canvas_colors, canvasoffset, ix, iy, target_matrix))
        
        logger.info(f"🚀 Запускаю завантаження {len(tasks)} чанків...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        loaded = sum(1 for r in results if r and not isinstance(r, Exception))
        logger.info(f"✅ Завантажено: {loaded}/{len(tasks)}")
    
    logger.info(f"🎬 Завершення завантаження області")
    return target_matrix


def compare_with_template(template, board_img, tolerance=20):
    """Порівнює шаблон з дошкою"""
    tw, th = template.size
    
    if board_img.size != (tw, th):
        board_img = board_img.crop((0, 0, min(tw, board_img.size[0]), min(th, board_img.size[1])))
        if board_img.size != (tw, th):
            temp = PIL.Image.new('RGBA', (tw, th), (0, 0, 0, 0))
            temp.paste(board_img, (0, 0))
            board_img = temp
    
    t_array = np.array(template, dtype=np.uint8)
    b_array = np.array(board_img, dtype=np.uint8)
    
    template_mask = t_array[..., 3] > 10
    total_pixels = int(template_mask.sum())
    
    if total_pixels == 0:
        return {"total": 0, "placed": 0, "remaining": 0, "percent": 100.0}
    
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


def create_overlay(template, board_img, tolerance=20, output_path=None):
    """Створює overlay (червоні = неправильні)"""
    if not output_path:
        return None
        
    tw, th = template.size
    if board_img.size != (tw, th):
        board_img = board_img.crop((0, 0, min(tw, board_img.size[0]), min(th, board_img.size[1])))
        if board_img.size != (tw, th):
            temp = PIL.Image.new('RGBA', (tw, th), (0, 0, 0, 0))
            temp.paste(board_img, (0, 0))
            board_img = temp
    
    t_array = np.array(template, dtype=np.uint8)
    b_array = np.array(board_img, dtype=np.uint8)
    
    template_mask = t_array[..., 3] > 10
    board_mask = b_array[..., 3] > 10
    
    diff = np.abs(b_array[..., :3].astype(np.int16) - t_array[..., :3].astype(np.int16))
    color_distance = np.sqrt((diff ** 2).sum(axis=-1))
    color_match = color_distance <= tolerance
    
    output = b_array.copy()
    bad_pixels = template_mask & (~color_match | ~board_mask)
    output[bad_pixels] = [255, 0, 0, 255]
    
    output_img = PIL.Image.fromarray(output, mode='RGBA').convert('RGB')
    output_img.save(output_path, 'PNG')
    logger.info(f"💾 Overlay: {output_path}")
    return output_path


async def process_lirn_template(template_path, x, y, canvas_id=0, tolerance=20, overlay_path=None):
    """Головна функція обробки (CHUNKS METHOD!)"""
    template = PIL.Image.open(template_path).convert("RGBA")
    width, height = template.size
    logger.info(f"📐 Шаблон: {width}x{height} px")
    
    # Отримуємо інфо про канвас
    api_me = await fetch_api_me()
    if not api_me or 'canvases' not in api_me:
        raise Exception("Не вдалось отримати інфо про канваси")
    
    canvas_info = api_me['canvases'].get(str(canvas_id))
    if not canvas_info:
        raise Exception(f"Канвас {canvas_id} не знайдено")
    
    canvas_size = canvas_info.get('size', 32768)
    canvas_max = canvas_size // 2
    canvas_min = -canvas_max
    
    logger.info(f"📏 Канвас: {canvas_info.get('title', '?')} ({canvas_size}x{canvas_size})")
    logger.info(f"📊 Межі канвасу: X[{canvas_min}..{canvas_max-1}], Y[{canvas_min}..{canvas_max-1}]")
    
    # ПЕРЕВІРКА КООРДИНАТ
    if x < canvas_min or y < canvas_min:
        raise Exception(f"❌ Координати ({x}, {y}) за межами канвасу! Мінімум: ({canvas_min}, {canvas_min})")
    
    if x + width > canvas_max or y + height > canvas_max:
        raise Exception(f"❌ Область виходить за межі! Максимум: ({canvas_max-1}, {canvas_max-1})")
    
    logger.info(f"✅ Координати в межах канвасу")
    
    # Завантажуємо область через CHUNKS
    matrix = await get_canvas_area(canvas_id, canvas_info, x, y, width, height)
    
    # Створюємо зображення з матриці
    logger.info(f"🖼️ Створюю зображення з матриці...")
    board_img = matrix.create_image()
    logger.info(f"✅ Зображення створено: {board_img.size}")
    
    # Порівнюємо
    logger.info(f"🔍 Порівнюю з шаблоном...")
    result = compare_with_template(template, board_img, tolerance)
    
    # Створюємо overlay
    if overlay_path:
        logger.info(f"🎨 Створюю overlay...")
        create_overlay(template, board_img, tolerance, overlay_path)
    
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
        "🎨 **UkrLirn Monitor Bot** (CHUNKS METHOD!)\n\n"
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
    faction_id = FACTION_IDS.get(CURRENT_SITE, "?")
    
    await update.message.reply_text(
        f"⚙️ **Налаштування:**\n\n"
        f"🌐 Сайт: `{CURRENT_SITE}`\n"
        f"📐 Шаблон: {'✅' if template_exists else '❌'}\n"
        f"📍 Координати: {f'({x}, {y})' if [x,y] != [0,0] else '❌'}\n"
        f"🎨 Толеранс: {COLOR_TOLERANCE}\n"
        f"🏰 Фракція ID: {faction_id}\n"
        f"⚡ Метод: CHUNKS (BMP)",
        parse_mode="Markdown"
    )


async def set_coords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) != 2:
        await update.message.reply_text(
            "⚠️ `/set_coords X Y`\n\n"
            "Приклад: `/set_coords 4031 -11628`\n\n"
            "📊 Межі канвасу (65536x65536):\n"
            "• X: від `-32768` до `32767`\n"
            "• Y: від `-32768` до `32767`",
            parse_mode="Markdown"
        )
        return
    
    try:
        x, y = int(context.args[0]), int(context.args[1])
        
        # ПЕРЕВІРКА КООРДИНАТ (65536x65536)
        MAX_COORD = 32768
        MIN_COORD = -32768
        
        if x < MIN_COORD or x >= MAX_COORD or y < MIN_COORD or y >= MAX_COORD:
            await update.message.reply_text(
                f"❌ Координати за межами канвасу!\n\n"
                f"📊 Допустимі межі (65536x65536):\n"
                f"• X: від `{MIN_COORD}` до `{MAX_COORD-1}`\n"
                f"• Y: від `{MIN_COORD}` до `{MAX_COORD-1}`",
                parse_mode="Markdown"
            )
            return
        
        state["lirn_coords"] = [x, y]
        save_state()
        await update.message.reply_text(f"✅ Координати: `{x}, {y}`", parse_mode="Markdown")
    except ValueError:
        await update.message.reply_text("❌ Координати мають бути числами!")


async def upload_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📤 **Завантаження шаблону**\n\n"
        "**Крок 1/3:** Надішли PNG файл шаблону\n\n"
        "Для скасування: /cancel"
    )
    return UPLOAD_TEMPLATE_WAITING


async def upload_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith('.png'):
        await update.message.reply_text("❌ Потрібен PNG файл!")
        return ConversationHandler.END
    
    file = await doc.get_file()
    os.makedirs("templates", exist_ok=True)
    temp_path = "templates/temp_upload.png"
    await file.download_to_drive(temp_path)
    
    context.user_data['temp_template'] = temp_path
    
    img = PIL.Image.open(temp_path)
    await update.message.reply_text(
        f"✅ Файл отримано: `{img.size[0]}x{img.size[1]}` px\n\n"
        f"**Крок 2/3:** Введи назву версії\n"
        f"_Приклад:_ `v1.0` _або_ `0\\_0`",
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
        f"Формат: `X Y` наприклад `4031 -11628`",
        parse_mode="Markdown"
    )
    return UPLOAD_COORDS_WAITING


async def upload_coords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        parts = update.message.text.strip().split()
        if len(parts) != 2:
            raise ValueError
        x, y = int(parts[0]), int(parts[1])
        
        # ПЕРЕВІРКА КООРДИНАТ (межі 65536x65536)
        MAX_COORD = 32768
        MIN_COORD = -32768
        
        if x < MIN_COORD or x >= MAX_COORD or y < MIN_COORD or y >= MAX_COORD:
            await update.message.reply_text(
                f"❌ Координати за межами канвасу!\n\n"
                f"📊 Допустимі межі (65536x65536):\n"
                f"• X: від `{MIN_COORD}` до `{MAX_COORD-1}`\n"
                f"• Y: від `{MIN_COORD}` до `{MAX_COORD-1}`\n\n"
                f"Ти ввів: `{x}, {y}`",
                parse_mode="Markdown"
            )
            return UPLOAD_COORDS_WAITING
            
    except ValueError:
        await update.message.reply_text(
            "❌ Неправильний формат!\n\n"
            "Введи координати: `X Y`\n"
            "Приклад: `4031 -11628`",
            parse_mode="Markdown"
        )
        return UPLOAD_COORDS_WAITING
    
    temp_path = context.user_data.get('temp_template')
    version = context.user_data.get('template_version')
    
    if temp_path and os.path.exists(temp_path):
        os.rename(temp_path, LIRN_TEMPLATE["file"])
        
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
    await update.message.reply_text("❌ Завантаження скасовано")
    return ConversationHandler.END


async def get_template(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отримати поточний шаблон"""
    if not os.path.exists(LIRN_TEMPLATE["file"]):
        await update.message.reply_text(
            "❌ Шаблон відсутній!\n\nЗавантаж його: `/upload_template`",
            parse_mode="Markdown"
        )
        return
    
    x, y = state.get("lirn_coords", [0, 0])
    
    caption = (
        f"📐 **Поточний шаблон**\n\n"
        f"📍 Координати: `{x}, {y}`\n"
        f"🌐 Сайт: {CURRENT_SITE}"
    )
    
    with open(LIRN_TEMPLATE["file"], "rb") as f:
        await update.message.reply_document(
            document=f,
            caption=caption,
            parse_mode="Markdown",
            filename="lirn_template.png"
        )


async def check_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Перевірка прогресу (CHUNKS METHOD!)"""
    if not os.path.exists(LIRN_TEMPLATE["file"]):
        await update.message.reply_text(
            "❌ Завантаж шаблон: `/upload_template`",
            parse_mode="Markdown"
        )
        return
    
    x, y = state.get("lirn_coords", [0, 0])
    if [x, y] == [0, 0]:
        await update.message.reply_text(
            "❌ Встанови координати: `/set_coords X Y`",
            parse_mode="Markdown"
        )
        return
    
    status_msg = await update.message.reply_text(f"⚡ Завантажую чанки ({x}, {y})...")
    
    try:
        os.makedirs("progress", exist_ok=True)
        
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
            f"📍 ({x}, {y}) • {CURRENT_SITE}\n"
            f"⚡ Метод: CHUNKS"
        )
        
        with open("progress/overlay.png", "rb") as f:
            await update.message.reply_document(
                document=f,
                caption=caption,
                parse_mode="Markdown",
                filename="progress.png"
            )
        
        await status_msg.delete()
        
    except asyncio.TimeoutError:
        await status_msg.edit_text(
            "❌ **Таймаут!** Спробуй ще раз або зменши область.",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Помилка check: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Помилка: `{str(e)}`", parse_mode="Markdown")


async def connect_player(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "⚠️ `/connect <нік>`\n\nПриклад: `/connect Puwe`",
            parse_mode="Markdown"
        )
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
            await update.message.reply_text(
                "⚠️ `/connect <нік>`",
                parse_mode="Markdown"
            )
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
        await update.message.reply_text(
            "⚠️ Відповідай на повідомлення!\n`/madd <назва> <1-10>`",
            parse_mode="Markdown"
        )
        return
    
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "⚠️ `/madd <назва> <вага>`\n\nПриклад: `/madd Художник 10`",
            parse_mode="Markdown"
        )
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
    await update.message.reply_text(
        f"✅ Медаль додано!\n\n🏅 **{name}** {stars}",
        parse_mode="Markdown"
    )


async def delete_medal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await update.message.reply_text(
            "⚠️ Відповідай на повідомлення!\n`/mdel <номер>`",
            parse_mode="Markdown"
        )
        return
    
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(
            "⚠️ `/mdel <номер>`",
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
        await update.message.reply_text("❌ Немає медалей!")
        return
    
    if index < 0 or index >= len(state["medals"][target_id]):
        await update.message.reply_text("❌ Медалі з таким номером не існує!")
        return
    
    removed = state["medals"][target_id].pop(index)
    save_state()
    
    await update.message.reply_text(
        f"✅ Видалено: 🏅 {removed['name']}",
        parse_mode="Markdown"
    )


async def change_site(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        sites_list = "\n".join([f"• `{name}` (фракція ID: {FACTION_IDS.get(name, '?')})" for name in SITES.keys()])
        current_faction = FACTION_IDS.get(CURRENT_SITE, "?")
        await update.message.reply_text(
            f"⚠️ `/site_change <сайт>`\n\n"
            f"**Доступні:**\n{sites_list}\n\n"
            f"🌐 Поточний: `{CURRENT_SITE}` (фракція: {current_faction})",
            parse_mode="Markdown"
        )
        return
    
    site_name = context.args[0].lower()
    
    if set_site(site_name):
        state["current_site"] = site_name
        save_state()
        faction_id = FACTION_IDS.get(site_name, "?")
        await update.message.reply_text(
            f"✅ Сайт змінено!\n\n"
            f"🌐 **{site_name}**\n"
            f"🔗 {SITES[site_name]['url']}\n"
            f"🏰 Фракція ID: {faction_id}",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"❌ Невідомий сайт: `{site_name}`\n\n"
            f"Доступні: {', '.join(SITES.keys())}",
            parse_mode="Markdown"
        )


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
    app.add_handler(CommandHandler("get", get_template))
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
    logger.info(f"⚡ МЕТОД: CHUNKS (правильний!)")
    logger.info(f"📐 Розмір чанку: {CHUNK_SIZE}px")
    logger.info(f"🌐 Сайт: {CURRENT_SITE}")
    logger.info(f"🏰 Фракції: pixmap={FACTION_IDS['pixmap']}, pixelya={FACTION_IDS['pixelya']}")
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
