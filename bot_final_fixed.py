#!/usr/bin/env python3
# bot.py - UkrLirn Monitor Bot (Фінальна версія з фіксом 401)

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
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# АВТОРИЗАЦІЯ (КРИТИЧНО ДЛЯ ФІКСА 401!)
AUTH_COOKIE = os.environ.get("AUTH_COOKIE", "s%3AS2qBqqlzYPCWST-OalOz6svoEoTYQIi9.%2BL0JZVkMRNrHr9eQ8WAuf4D9MdthKJP3pHCrqliUmZs")

API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Cookie': f'cpn.session={AUTH_COOKIE}; plang=ru',
    'Accept': 'application/json'
}

COLOR_TOLERANCE = 20
STATE_FILE = "state.json"
FACTION_API_URL = "https://canvaspix.fun/api/faction/my-faction"
STRICT_TILE_SIZE = 256  # Фіксований розмір тайлів

# ======================================================================
# 🌐 САЙТИ
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

CURRENT_SITE = "canvaspix"

def set_site(site_name: str) -> bool:
    global CURRENT_SITE
    if site_name.lower() in SITES:
        CURRENT_SITE = site_name.lower()
        return True
    return False

def get_current_site():
    return SITES[CURRENT_SITE]

# ======================================================================
# 🎨 ДВИЖОК (256px тайли)
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
    site = get_current_site()
    offset = int(-canvas_size / 2)
    tile_size = STRICT_TILE_SIZE
    
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
                offx = ix * tile_size + offset - x
                offy = iy * tile_size + offset - y
                
                if debug_save:
                    logger.info(f"Тайл [{ix},{iy}]: {url}")
                
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
    tw, th = template.size
    if board.size != (tw, th):
        board = board.crop((0, 0, tw, th))
    
    t = np.array(template, dtype=np.float32)
    b = np.array(board, dtype=np.float32)
    
    mask_template = t[..., 3] > 10
    total = int(mask_template.sum())
    
    if total == 0:
        return {"total": 0, "placed": 0, "remaining": 0, "percent": 100.0}

    mask_board = b[..., 3] > 10
    diff = np.sqrt(np.sum((b[..., :3] - t[..., :3]) ** 2, axis=-1))
    within_color = diff <= tolerance
    
    placed = int((mask_template & mask_board & within_color).sum())
    remaining = total - placed
    percent = (placed / total * 100.0) if total > 0 else 100.0
    
    logger.info(f"✅ Правильно: {placed}/{total} ({percent:.2f}%)")
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
    bad = (mask_template & (~within_color | ~mask_board))
    output[bad] = [255, 0, 0, 255]
    
    PIL.Image.fromarray(output.astype(np.uint8)).save(output_path)
    logger.info(f"💾 Overlay: {output_path}")
    return output_path


async def process_lirn_template(template_path, x, y, canvas_id=0, tolerance=20, overlay_path=None, debug_mode=False):
    template = PIL.Image.open(template_path).convert("RGBA")
    width, height = template.size
    logger.info(f"📐 Шаблон: {width}x{height}")
    
    api_me = await fetch_api_me()
    canvas_size = 32768
    if api_me and 'canvases' in api_me:
        info = api_me['canvases'].get(str(canvas_id))
        if info:
            canvas_size = info.get('size', 32768)
            
    board = await get_canvas_area(canvas_id, x, y, width, height, canvas_size, debug_save=debug_mode)
    result = compare_with_template(template, board, tolerance)
    
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
    "current_site": "canvaspix"
}

def load_state():
    global state
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state.update(json.load(f))
                set_site(state.get("current_site", "canvaspix"))
        except Exception as e:
            logger.error(f"Помилка завантаження: {e}")

def save_state():
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Помилка збереження: {e}")


# --- КОМАНДИ ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎨 **UkrLirn Monitor Bot** (256px режим)\n\n"
        "**Шаблон:**\n"
        "• `/upload_template` — завантажити\n"
        "• `/set_coords X Y` — координати\n"
        "• `/check` — прогрес\n"
        "• `/test_check` — тест з дебагом\n\n"
        "**Гравці:**\n"
        "• `/connect <нік>` — прив'язати\n"
        "• `/profile [нік]` — профіль\n"
        "• `/list` — список фракції\n\n"
        "**Медалі (адмін):**\n"
        "• `/madd <назва> <1-10>` (у відповідь)\n"
        "• `/mdel <номер>` (у відповідь)\n\n"
        "**Інше:**\n"
        "• `/site_change <сайт>` — змінити сайт",
        parse_mode="Markdown"
    )


async def set_coords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) != 2:
        await update.message.reply_text("⚠️ Приклад: `/set_coords 4031 -11628`", parse_mode="Markdown")
        return
    try:
        x, y = int(context.args[0]), int(context.args[1])
        state["lirn_coords"] = [x, y]
        save_state()
        await update.message.reply_text(f"✅ Координати: {x}, {y}")
    except ValueError:
        await update.message.reply_text("❌ Координати мають бути числами!")


async def upload_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Надішли PNG файл шаблону:")
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
        f"✅ Шаблон завантажено!\n"
        f"📐 Розмір: {img.size[0]}x{img.size[1]}\n\n"
        f"Тепер: `/set_coords X Y`",
        parse_mode="Markdown"
    )
    return ConversationHandler.END


async def cancel_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Скасовано.")
    return ConversationHandler.END


async def check_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(LIRN_TEMPLATE["file"]):
        await update.message.reply_text("❌ Спочатку завантаж шаблон: /upload_template")
        return
    
    x, y = state.get("lirn_coords", [0, 0])
    if [x, y] == [0, 0]:
        await update.message.reply_text("❌ Встанови координати: /set_coords X Y")
        return
    
    status_msg = await update.message.reply_text(f"⏳ Перевіряю ({x}, {y})...")
    
    try:
        os.makedirs("progress", exist_ok=True)
        res = await process_lirn_template(
            LIRN_TEMPLATE["file"], x, y,
            tolerance=COLOR_TOLERANCE,
            overlay_path="progress/overlay.png"
        )
        
        caption = (
            f"📊 **Прогрес UkrLirn**\n\n"
            f"🎯 Всього: `{res['total']:,}` px\n"
            f"✅ Готово: `{res['placed']:,}` px\n"
            f"❌ Залишилось: `{res['remaining']:,}` px\n"
            f"📈 **{res['percent']:.2f}%**"
        )
        
        with open("progress/overlay.png", "rb") as f:
            await update.message.reply_document(document=f, caption=caption, parse_mode="Markdown")
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"Помилка check: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Помилка: {str(e)}")


async def test_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ТЕСТОВИЙ режим з збереженням тайлів"""
    if not os.path.exists(LIRN_TEMPLATE["file"]):
        await update.message.reply_text("❌ Спочатку завантаж шаблон!")
        return
    
    x, y = state.get("lirn_coords", [0, 0])
    status_msg = await update.message.reply_text(
        f"🔍 **ТЕСТОВИЙ РЕЖИМ**\n\n"
        f"Завантажую тайли та зберігаю в `debug/tiles/`...",
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
            f"📈 **{res['percent']:.2f}%**"
        )
        
        if os.path.exists("debug/board_full.png"):
            with open("debug/board_full.png", "rb") as f:
                await update.message.reply_document(document=f, caption="📄 Склеєна доска")
        
        if os.path.exists("debug/test_overlay.png"):
            with open("debug/test_overlay.png", "rb") as f:
                await update.message.reply_document(document=f, caption=msg, parse_mode="Markdown")
        
        await status_msg.delete()
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Помилка: {str(e)}")


async def connect_player(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Приклад: `/connect Puwo`", parse_mode="Markdown")
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
            await update.message.reply_text("⚠️ Прив'яжи ник: `/connect <нік>`", parse_mode="Markdown")
            return
    
    msg = await update.message.reply_text("🔍 Завантажую дані фракції...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(FACTION_API_URL, headers=API_HEADERS, timeout=10) as resp:
                if resp.status == 401:
                    await msg.edit_text("❌ Помилка 401: Оновіть AUTH_COOKIE!")
                    return
                if resp.status != 200:
                    await msg.edit_text(f"❌ Помилка API: {resp.status}")
                    return
                data = await resp.json()
    except Exception as e:
        await msg.edit_text(f"❌ Помилка: {e}")
        return

    faction = data.get("faction", data)
    found = None
    for member in faction.get("members", []):
        if member.get("User", {}).get("name", "").lower() == nick.lower():
            found = member
            break
    
    if not found:
        await msg.edit_text(f"❌ Гравець **{nick}** не знайдений.", parse_mode="Markdown")
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
    msg = await update.message.reply_text("⏳ Завантажую...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(FACTION_API_URL, headers=API_HEADERS, timeout=10) as resp:
                if resp.status != 200:
                    await msg.edit_text(f"❌ Помилка: {resp.status}")
                    return
                data = await resp.json()
    except Exception as e:
        await msg.edit_text(f"❌ Помилка: {e}")
        return
    
    faction = data.get("faction", data)
    members = faction.get("members", [])
    
    if not members:
        await msg.edit_text("📭 Список порожній.")
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
        txt += f"\n_...та ще {len(members) - 30}_"
    
    await msg.edit_text(txt, parse_mode="Markdown")


async def add_medal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Додати медаль: /madd <назва> <вага 1-10> у відповідь"""
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Відповідай на повідомлення гравця!")
        return
    
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("⚠️ Приклад: /madd Найкращий_художник 10")
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
    await update.message.reply_text(f"✅ Медаль додано!\n\n🏅 **{name}** {stars}", parse_mode="Markdown")


async def delete_medal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Видалити медаль: /mdel <номер> у відповідь"""
    if not update.message.reply_to_message:
        await update.message.reply_text("⚠️ Відповідай на повідомлення гравця!")
        return
    
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("⚠️ Приклад: /mdel 1")
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
    
    await update.message.reply_text(f"✅ Видалено:\n\n🏅 {removed['name']}", parse_mode="Markdown")


async def change_site(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Змінити сайт: /site_change <назва>"""
    if not context.args:
        sites_list = "\n".join([f"• `{name}`" for name in SITES.keys()])
        await update.message.reply_text(
            f"⚠️ Використання: /site_change <назва>\n\n"
            f"**Доступні:**\n{sites_list}",
            parse_mode="Markdown"
        )
        return
    
    site_name = context.args[0].lower()
    
    if set_site(site_name):
        state["current_site"] = site_name
        save_state()
        await update.message.reply_text(
            f"✅ Сайт змінено на: **{site_name}**",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"❌ Невідомий сайт: {site_name}",
            parse_mode="Markdown"
        )


# ======================================================================
# 🚀 ЗАПУСК
# ======================================================================

if __name__ == "__main__":
    if not BOT_TOKEN:
        print("❌ Встанови BOT_TOKEN!")
        sys.exit(1)
    
    load_state()
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    # ConversationHandler для завантаження
    upload_conv = ConversationHandler(
        entry_points=[CommandHandler("upload_template", upload_start)],
        states={UPLOAD_TEMPLATE_WAITING: [MessageHandler(filters.Document.ALL, upload_file)]},
        fallbacks=[CommandHandler("cancel", cancel_upload)]
    )
    
    # Реєструємо команди
    app.add_handler(CommandHandler("start", start_cmd))
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

    logger.info("="*60)
    logger.info("🤖 UkrLirn Bot запущено!")
    logger.info("📐 Режим: 256px тайли")
    logger.info("🔐 Авторизація: активована (фікс 401)")
    logger.info("="*60)
    
    try:
        app.run_polling()
    except KeyboardInterrupt:
        logger.info("\n⛔ Зупинка бота.")
