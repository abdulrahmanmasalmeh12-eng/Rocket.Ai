print(">>> RUNNING THIS MAIN.PY <<<")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import hashlib
import hmac
import requests
import io
import base64
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from urllib.parse import quote
from pypdf import PdfReader
import pandas as pd

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LLM (Groq)
# -----------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"
SESSION_COOKIE = "rocket_session"
RESET_EXPIRY_MINUTES = 30

def ask_llama(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 512
        },
        timeout=120
    )
    if not resp.ok:
        try:
            error_message = resp.json().get("error", {}).get("message", resp.text)
        except ValueError:
            error_message = resp.text
        raise HTTPException(
            status_code=502,
            detail=f"Groq error ({resp.status_code}): {error_message[:300]}"
        )
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise HTTPException(status_code=502, detail=f"Invalid Groq response: {error}")

# -----------------------------
# Database Setup
# -----------------------------
def get_db():
    conn = sqlite3.connect("rocket.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL DEFAULT 'New chat',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    return conn

def now_utc():
    return datetime.now(timezone.utc)

def iso_now():
    return now_utc().isoformat()

def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"

def verify_password(password: str, stored: str) -> bool:
    if stored.startswith("pbkdf2_sha256$"):
        _, salt_hex, digest_hex = stored.split("$", 2)
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), 240_000
        ).hex()
        return hmac.compare_digest(candidate, digest_hex)
    return hmac.compare_digest(stored, hashlib.sha256(password.encode()).hexdigest())

def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

def create_session(conn, user_id: int):
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?)",
        (token_hash(token), user_id, (now_utc() + timedelta(days=30)).isoformat())
    )
    return token

def current_user(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    conn = get_db()
    row = conn.execute(
        """SELECT users.id, users.email FROM sessions
           JOIN users ON users.id = sessions.user_id
           WHERE sessions.token_hash = ? AND sessions.expires_at > ?""",
        (token_hash(token), iso_now())
    ).fetchone()
    conn.close()
    return row

def require_user(request: Request):
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to use saved chats.")
    return user

def auth_response(payload, session_token: str):
    response = JSONResponse(payload)
    response.set_cookie(
        SESSION_COOKIE, session_token, max_age=60 * 60 * 24 * 30,
        httponly=True, samesite="lax", path="/"
    )
    return response

# -----------------------------
# AUTH ROUTES
# -----------------------------
@app.post("/register")
def register(email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, hash_password(password))
        )
        session_token = create_session(conn, cursor.lastrowid)
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    conn.close()
    return auth_response({"message": "Account created", "email": email}, session_token)

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    conn = get_db()
    cur = conn.execute(
        "SELECT id, password_hash FROM users WHERE email = ?",
        (email,)
    )
    row = cur.fetchone()

    if not row or not verify_password(password, row["password_hash"]):
        conn.close()
        raise HTTPException(status_code=400, detail="Invalid credentials")
    if not row["password_hash"].startswith("pbkdf2_sha256$"):
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(password), row["id"])
        )
    session_token = create_session(conn, row["id"])
    conn.commit()
    conn.close()
    return auth_response({"message": "Login successful", "email": email}, session_token)

@app.post("/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    response = JSONResponse({"message": "Logged out"})
    if token:
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),))
        conn.commit()
        conn.close()
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response

@app.get("/me")
def me(request: Request):
    user = current_user(request)
    return {"authenticated": bool(user), "email": user["email"] if user else None}

def send_reset_email(email: str, reset_url: str):
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM", username or "")
    if not all((host, username, password, sender)):
        raise RuntimeError("SMTP_HOST, SMTP_USER, SMTP_PASSWORD, and SMTP_FROM are required")
    message = EmailMessage()
    message["Subject"] = "Reset your Rocket password"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        "Use this link to reset your Rocket password. It expires in 30 minutes:\n\n" + reset_url
    )
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=30) as smtp:
        smtp.starttls()
        smtp.login(username, password)
        smtp.send_message(message)

@app.post("/forgot-password")
def forgot_password(email: str = Form(...)):
    email = email.strip().lower()
    conn = get_db()
    user = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if user:
        raw_token = secrets.token_urlsafe(32)
        conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user["id"],))
        conn.execute(
            "INSERT INTO password_reset_tokens VALUES (?, ?, ?, 0)",
            (token_hash(raw_token), user["id"],
             (now_utc() + timedelta(minutes=RESET_EXPIRY_MINUTES)).isoformat())
        )
        conn.commit()
        base_url = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000")
        try:
            send_reset_email(email, f"{base_url}/?reset_token={raw_token}")
        except (RuntimeError, OSError, smtplib.SMTPException) as error:
            print(f"Password reset email failed: {error}")
            print(f"Development reset URL: {base_url}/?reset_token={raw_token}")
    conn.close()
    return {"message": "If that email exists, a password reset link has been sent."}

@app.post("/reset-password")
def reset_password(token: str = Form(...), password: str = Form(...)):
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    conn = get_db()
    row = conn.execute(
        """SELECT user_id FROM password_reset_tokens
           WHERE token_hash = ? AND used = 0 AND expires_at > ?""",
        (token_hash(token), iso_now())
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=400, detail="This reset link is invalid or expired")
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(password), row["user_id"]))
    conn.execute("UPDATE password_reset_tokens SET used = 1 WHERE token_hash = ?", (token_hash(token),))
    conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["user_id"],))
    email_row = conn.execute("SELECT email FROM users WHERE id = ?", (row["user_id"],)).fetchone()
    session_token = create_session(conn, row["user_id"])
    conn.commit()
    conn.close()
    return auth_response(
        {"message": "Password reset successfully", "email": email_row["email"]},
        session_token
    )

# -----------------------------
# CHAT ROUTE (with attachments)
# -----------------------------
@app.get("/conversations")
def list_conversations(request: Request):
    user = require_user(request)
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
        (user["id"],)
    ).fetchall()
    conn.close()
    return {"conversations": [dict(row) for row in rows]}

@app.post("/conversations")
def create_conversation(request: Request):
    user = require_user(request)
    timestamp = iso_now()
    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO conversations (user_id, title, created_at, updated_at)
           VALUES (?, ?, ?, ?)""",
        (user["id"], "New chat", timestamp, timestamp)
    )
    conn.commit()
    conversation_id = cursor.lastrowid
    conn.close()
    return {"id": conversation_id, "title": "New chat"}

@app.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: int, request: Request):
    user = require_user(request)
    conn = get_db()
    owner = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user["id"])
    ).fetchone()
    if not owner:
        conn.close()
        raise HTTPException(status_code=404, detail="Conversation not found")
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,)
    ).fetchall()
    conn.close()
    return {"messages": [dict(row) for row in rows]}

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, request: Request):
    user = require_user(request)
    conn = get_db()
    cursor = conn.execute(
        "DELETE FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user["id"])
    )
    conn.commit()
    conn.close()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}

@app.post("/generate-image")
def generate_image(prompt: str = Form(...)):
    image_api_url = os.getenv("IMAGE_API_URL")
    image_api_key = os.getenv("IMAGE_API_KEY")
    image_model = os.getenv("IMAGE_MODEL", "dall-e-3")
    if not image_api_url or not image_api_key:
        return {
            "image_url": (
                "https://image.pollinations.ai/prompt/"
                + quote(prompt, safe="")
                + "?width=1024&height=1024&nologo=true"
            )
        }
    response = requests.post(
        image_api_url,
        headers={"Authorization": f"Bearer {image_api_key}"},
        json={"model": image_model, "prompt": prompt, "size": "1024x1024", "n": 1},
        timeout=180
    )
    if not response.ok:
        raise HTTPException(status_code=502, detail="The image provider could not create that image.")
    result = response.json().get("data", [{}])[0]
    if result.get("url"):
        return {"image_url": result["url"]}
    if result.get("b64_json"):
        return {"image_url": "data:image/png;base64," + result["b64_json"]}
    raise HTTPException(status_code=502, detail="The image provider returned no image.")

@app.post("/chat")
async def chat(
    request: Request,
    message: str = Form(...),
    conversation_id: int | None = Form(None),
    file: UploadFile | None = File(None)
):
    user = current_user(request)
    base_prompt = f"User says: {message}\n\n"

    if file:
        content = await file.read()

        # PDF
        filename = (file.filename or "").lower()
        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            base_prompt += f"\nAttached PDF content:\n{text[:8000]}"

        # Excel
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
            summary = df.describe().to_string()
            base_prompt += f"\nAttached spreadsheet summary:\n{summary}"

        # Other files
        else:
            base_prompt += f"\nAttached file: {file.filename}"

    reply = ask_llama(base_prompt)
    if user:
        conn = get_db()
        if conversation_id:
            owner = conn.execute(
                "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user["id"])
            ).fetchone()
            if not owner:
                conn.close()
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            timestamp = iso_now()
            cursor = conn.execute(
                """INSERT INTO conversations (user_id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (user["id"], message[:48] or "New chat", timestamp, timestamp)
            )
            conversation_id = cursor.lastrowid
        timestamp = iso_now()
        conn.executemany(
            """INSERT INTO messages (conversation_id, role, content, created_at)
               VALUES (?, ?, ?, ?)""",
            [
                (conversation_id, "user", message, timestamp),
                (conversation_id, "assistant", reply, timestamp)
            ]
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id)
        )
        conn.commit()
        conn.close()
    return {"reply": reply, "conversation_id": conversation_id}

# -----------------------------
# SERVE FRONTEND
# -----------------------------
@app.get("/manifest.json")
def web_manifest():
    return FileResponse("manifest.json", media_type="application/manifest+json")

@app.get("/sw.js")
def service_worker():
    return FileResponse("sw.js", media_type="application/javascript")

@app.get("/icons/rocket.svg")
def app_icon():
    return FileResponse("icons/rocket.svg", media_type="image/svg+xml")

@app.get("/rocket.png.png")
def logo_image():
    return FileResponse("rocket.png.png", media_type="image/png")

@app.get("/", response_class=HTMLResponse)
def home_page():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
