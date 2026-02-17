"""English Morning Call - Real-time English conversation practice with Gemini Live API."""

import asyncio
import json
import os
import glob
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="English Morning Call")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
TEXT_MODEL = "gemini-2.5-flash"

DATA_DIR = Path(__file__).parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

BASE_SYSTEM_INSTRUCTION = """You are a friendly English conversation tutor. Your job is to help the user practice English conversation.

Rules:
- Speak naturally and at a moderate pace
- If the user makes a grammar mistake, gently correct it after they finish their thought
- Ask follow-up questions to keep the conversation going
- Start by greeting the user and suggesting today's conversation topic
- Keep responses concise (2-3 sentences max) to make it feel like a real conversation
- Occasionally praise good usage of vocabulary or grammar
- If the user speaks Korean, respond briefly in Korean then switch back to English

Today's conversation topics (pick one randomly):
1. Weekend plans and hobbies
2. Favorite movies or TV shows
3. Travel experiences
4. Food and cooking
5. Technology and apps
6. Work-life balance
7. Dreams and goals
8. Interesting news"""

client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1beta"})


def get_past_context():
    """Build context from previous sessions for personalized instruction."""
    sessions = _load_all_sessions()
    if not sessions:
        return ""
    
    recent = sessions[-3:]  # last 3 sessions
    last = sessions[-1]
    
    context_parts = ["\n\n--- CONTEXT FROM PREVIOUS SESSIONS ---"]
    
    # Key expressions to review
    if last.get("feedback", {}).get("key_expressions"):
        exprs = last["feedback"]["key_expressions"]
        context_parts.append(f"Key expressions from last session to challenge the student to use: {', '.join(exprs)}")
    
    # Common weak areas
    all_grammar = []
    for s in recent:
        fb = s.get("feedback", {})
        all_grammar.extend([g.get("explanation", "") for g in fb.get("grammar", [])])
    if all_grammar:
        context_parts.append(f"Common grammar issues: {'; '.join(all_grammar[:5])}")
    
    # Vocabulary level
    levels = [s.get("feedback", {}).get("vocabulary_level", "") for s in recent if s.get("feedback", {}).get("vocabulary_level")]
    if levels:
        context_parts.append(f"Student's recent vocabulary level: {levels[-1]}")
    
    # Tip
    if last.get("feedback", {}).get("tip"):
        context_parts.append(f"Last session's tip: {last['feedback']['tip']}")
    
    return "\n".join(context_parts)


def build_live_config():
    """Build LiveConnectConfig with personalized context."""
    instruction = BASE_SYSTEM_INSTRUCTION + get_past_context()
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=instruction)]
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25600,
            sliding_window=types.SlidingWindow(target_tokens=12800),
        ),
    )


def _load_all_sessions():
    """Load all session files sorted by date."""
    files = sorted(SESSIONS_DIR.glob("*.json"))
    sessions = []
    for f in files:
        try:
            sessions.append(json.loads(f.read_text()))
        except Exception:
            pass
    return sessions


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps({"type": "status", "message": "connected"}))

    transcript_parts = []  # Collect transcript: list of {"role": "ai"|"user", "text": "..."}

    try:
        config = build_live_config()
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            await ws.send_text(json.dumps({"type": "status", "message": "ready"}))
            print("✅ Gemini session connected")

            async def recv_from_gemini():
                try:
                    while True:
                        turn = session.receive()
                        async for response in turn:
                            if data := response.data:
                                await ws.send_bytes(data)
                            if text := response.text:
                                print(f"📝 AI: {text[:80]}...")
                                transcript_parts.append({"role": "ai", "text": text})
                                await ws.send_text(json.dumps({"type": "transcript", "text": text, "role": "ai"}))
                        await ws.send_text(json.dumps({"type": "turn_complete"}))
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"❌ recv error: {e}")
                    try:
                        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
                    except Exception:
                        pass

            async def send_to_gemini():
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if "bytes" in msg and msg["bytes"]:
                            await session.send_realtime_input(
                                audio=types.Blob(data=msg["bytes"], mime_type="audio/pcm;rate=16000")
                            )
                        elif "text" in msg and msg["text"]:
                            data = json.loads(msg["text"])
                            if data.get("type") == "stop":
                                print("🛑 Stop requested")
                                break
                            elif data.get("type") == "user_transcript":
                                # User speech transcript from browser STT
                                transcript_parts.append({"role": "user", "text": data["text"]})
                                print(f"📝 User: {data['text'][:80]}...")
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    print(f"❌ send error: {e}")

            recv_task = asyncio.create_task(recv_from_gemini())
            send_task = asyncio.create_task(send_to_gemini())

            done, pending = await asyncio.wait(
                [recv_task, send_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Send transcript back to client before closing
            if transcript_parts:
                try:
                    await ws.send_text(json.dumps({
                        "type": "full_transcript",
                        "transcript": transcript_parts
                    }))
                except Exception:
                    pass

            print(f"🏁 Session ended, {len(transcript_parts)} transcript parts")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ Outer error: {e}")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


@app.post("/api/feedback")
async def post_feedback(request: Request):
    """Analyze transcript and return feedback JSON."""
    body = await request.json()
    transcript = body.get("transcript", [])
    duration = body.get("duration", 0)

    # Format transcript for analysis
    lines = []
    for t in transcript:
        role = "Student" if t["role"] == "user" else "Tutor"
        lines.append(f"{role}: {t['text']}")
    transcript_text = "\n".join(lines)

    if not transcript_text.strip():
        return JSONResponse({"error": "Empty transcript"}, status_code=400)

    prompt = f"""Analyze this English conversation between a Korean student and an AI tutor. Provide:
1. **Overall Score** (1-10)
2. **Grammar Corrections** — list each mistake with the correction
3. **Good Expressions** — phrases the student used well
4. **Vocabulary Level** — beginner/intermediate/advanced
5. **Key Expressions to Remember** — 3 useful phrases from this conversation
6. **Tip for Next Time** — one specific thing to improve

Return ONLY valid JSON (no markdown, no code fences):
{{
  "score": 8,
  "grammar": [{{"wrong": "I go yesterday", "correct": "I went yesterday", "explanation": "Past tense needed"}}],
  "good_expressions": ["That's a great point", "I couldn't agree more"],
  "vocabulary_level": "intermediate",
  "key_expressions": ["break the ice", "on the same page", "hit the nail on the head"],
  "tip": "Try using more past perfect tense when telling stories"
}}

If the student didn't say much, still provide constructive feedback and score accordingly.

TRANSCRIPT:
{transcript_text}"""

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        
        feedback = json.loads(text)
    except json.JSONDecodeError:
        feedback = {
            "score": 5,
            "grammar": [],
            "good_expressions": [],
            "vocabulary_level": "intermediate",
            "key_expressions": [],
            "tip": "Keep practicing! Try to speak more in the next session.",
            "_raw": text if 'text' in dir() else "parse error"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # Save session
    now = datetime.now()
    session_data = {
        "date": now.isoformat(),
        "duration": duration,
        "transcript": transcript,
        "feedback": feedback,
        "score": feedback.get("score", 0)
    }
    filename = now.strftime("%Y-%m-%d_%H%M%S") + ".json"
    (SESSIONS_DIR / filename).write_text(json.dumps(session_data, ensure_ascii=False, indent=2))

    return JSONResponse(feedback)


@app.get("/api/history")
async def get_history():
    """Return all sessions (lightweight - no full transcript)."""
    sessions = _load_all_sessions()
    result = []
    for s in sessions:
        result.append({
            "date": s.get("date"),
            "duration": s.get("duration", 0),
            "score": s.get("score", 0),
            "vocabulary_level": s.get("feedback", {}).get("vocabulary_level", ""),
            "tip": s.get("feedback", {}).get("tip", ""),
            "key_expressions": s.get("feedback", {}).get("key_expressions", []),
            "good_expressions": s.get("feedback", {}).get("good_expressions", []),
            "grammar_count": len(s.get("feedback", {}).get("grammar", [])),
        })
    return JSONResponse(result)


@app.get("/api/history/{idx}")
async def get_history_detail(idx: int):
    """Return full session detail by index."""
    sessions = _load_all_sessions()
    if 0 <= idx < len(sessions):
        return JSONResponse(sessions[idx])
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.get("/api/stats")
async def get_stats():
    """Aggregate stats."""
    sessions = _load_all_sessions()
    if not sessions:
        return JSONResponse({
            "total_sessions": 0, "total_minutes": 0,
            "current_streak": 0, "avg_score": 0, "common_mistakes": []
        })

    total_minutes = sum(s.get("duration", 0) for s in sessions) / 60
    scores = [s.get("score", 0) for s in sessions if s.get("score")]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0

    # Streak calculation
    dates = set()
    for s in sessions:
        try:
            dates.add(datetime.fromisoformat(s["date"]).date())
        except Exception:
            pass
    
    streak = 0
    d = datetime.now().date()
    while d in dates:
        streak += 1
        d -= timedelta(days=1)
    # If today not practiced but yesterday was, count from yesterday
    if streak == 0:
        d = datetime.now().date() - timedelta(days=1)
        while d in dates:
            streak += 1
            d -= timedelta(days=1)

    # Common mistakes
    all_explanations = []
    for s in sessions:
        for g in s.get("feedback", {}).get("grammar", []):
            all_explanations.append(g.get("explanation", "unknown"))
    
    from collections import Counter
    common = Counter(all_explanations).most_common(3)
    common_mistakes = [{"pattern": p, "count": c} for p, c in common]

    return JSONResponse({
        "total_sessions": len(sessions),
        "total_minutes": round(total_minutes, 1),
        "current_streak": streak,
        "avg_score": avg_score,
        "common_mistakes": common_mistakes
    })


@app.get("/api/next-topic")
async def get_next_topic():
    """Suggest next topic based on past sessions."""
    sessions = _load_all_sessions()
    
    if not sessions:
        return JSONResponse({"topic": "Introduce yourself and talk about your daily routine", "reason": "Great starting topic for new learners!"})

    recent = sessions[-5:]
    summary_parts = []
    for s in recent:
        fb = s.get("feedback", {})
        summary_parts.append(f"Score: {fb.get('score', '?')}, Level: {fb.get('vocabulary_level', '?')}, Tip: {fb.get('tip', 'none')}")
    
    prompt = f"""Based on these recent English practice sessions, suggest ONE specific conversation topic that would help the student improve. Consider their weak areas and level.

Recent sessions:
{chr(10).join(summary_parts)}

Return ONLY JSON: {{"topic": "...", "reason": "..."}}"""

    try:
        response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return JSONResponse(json.loads(text))
    except Exception:
        return JSONResponse({"topic": "Talk about a recent challenge you overcame", "reason": "Good for practicing past tense and storytelling"})


@app.get("/api/weekly-report")
async def get_weekly_report():
    """Weekly summary."""
    sessions = _load_all_sessions()
    week_ago = datetime.now() - timedelta(days=7)
    
    weekly = []
    for s in sessions:
        try:
            if datetime.fromisoformat(s["date"]) >= week_ago:
                weekly.append(s)
        except Exception:
            pass

    if not weekly:
        return JSONResponse({"message": "No sessions this week", "sessions": 0})

    scores = [s.get("score", 0) for s in weekly if s.get("score")]
    total_min = sum(s.get("duration", 0) for s in weekly) / 60
    
    all_grammar = []
    levels = []
    for s in weekly:
        fb = s.get("feedback", {})
        all_grammar.extend([g.get("explanation", "") for g in fb.get("grammar", [])])
        if fb.get("vocabulary_level"):
            levels.append(fb["vocabulary_level"])

    from collections import Counter
    recurring = Counter(all_grammar).most_common(3)

    return JSONResponse({
        "sessions": len(weekly),
        "total_minutes": round(total_min, 1),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        "score_trend": scores,
        "recurring_mistakes": [{"pattern": p, "count": c} for p, c in recurring],
        "vocabulary_levels": levels,
        "message": f"You practiced {len(weekly)} times this week for {round(total_min)}min total!"
    })


ALARM_FILE = DATA_DIR / "alarm.json"


@app.post("/api/alarm")
async def save_alarm(request: Request):
    """Save alarm settings."""
    body = await request.json()
    ALARM_FILE.write_text(json.dumps(body, ensure_ascii=False, indent=2))
    
    status = "enabled" if body.get("enabled") else "disabled"
    days_map = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
    days_str = ", ".join(days_map.get(d, "?") for d in sorted(body.get("days", [])))
    
    return JSONResponse({
        "message": f"Alarm {status}: {body.get('time', '06:30')} on {days_str}" if body.get("enabled") else "Alarm disabled",
        "settings": body
    })


@app.get("/api/alarm")
async def get_alarm():
    """Get alarm settings."""
    if ALARM_FILE.exists():
        return JSONResponse(json.loads(ALARM_FILE.read_text()))
    return JSONResponse({"enabled": False, "time": "06:30", "days": [1, 2, 3, 4, 5]})


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn

    cert_dir = os.path.join(os.path.dirname(__file__), "certs")
    cert_file = os.path.join(cert_dir, "jaydens-macbook-air.tail2cb2b5.ts.net.crt")
    key_file = os.path.join(cert_dir, "jaydens-macbook-air.tail2cb2b5.ts.net.key")

    if os.path.exists(cert_file):
        print(f"🔒 HTTPS enabled: https://jaydens-macbook-air.tail2cb2b5.ts.net:8080")
        uvicorn.run(app, host="0.0.0.0", port=8080, ssl_certfile=cert_file, ssl_keyfile=key_file)
    else:
        print(f"⚠️  No certs found, running HTTP: http://0.0.0.0:8080")
        uvicorn.run(app, host="0.0.0.0", port=8080)
