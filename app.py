import os
import json
import re
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# ThaiLLM Pathumma Config
THAILLM_URL = "http://thaillm.or.th/api/v1/chat/completions"
THAILLM_API_KEY = "D7kwgdJNQyDPQEz0vvV0UMvD8RLxPvdm"
THAILLM_MODEL = "pathumma-thaillm-qwen3-8b-think-3.0.0"


def call_thaillm_summarize(src_text, length_style="medium"):
    """
    Call ThaiLLM (Pathumma Qwen3 8B) for pure Thai text summarization
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {THAILLM_API_KEY}"
    }

    style_instruction = {
        "short": "สรุปใจความสำคัญแบบกระชับสั้น สรุปเนื้อหาสำคัญ 2-3 บรรทัด",
        "medium": "สรุปเนื้อหาสำคัญอย่างครบถ้วนกระชับ ครอบคลุมประเด็นหลักและสาระสำคัญทั้งหมด",
        "bullet": "สรุปใจความสำคัญและประเด็นหลักออกมาเป็นข้อๆ (Bullet points) ชัดเจนและอ่านง่าย"
    }.get(length_style, "สรุปเนื้อหาสำคัญอย่างครบถ้วนกระชับ ครอบคลุมประเด็นหลักและสาระสำคัญทั้งหมด")

    system_prompt = (
        "คุณคือผู้เชี่ยวชาญด้านการสรุปความภาษาไทย หน้าที่ของคุณคืออ่านข้อความต้นฉบับ "
        "และสรุปใจความสำคัญเป็นภาษาไทยที่กระชับ ชัดเจน ถูกต้องตามหลักภาษาไทย "
        "ไม่แต่งเติมข้อมูล และรักษาใจความสำคัญของเรื่องเดิมไว้ทั้งหมด"
    )

    user_prompt = f"""กรุณาสรุปความข้อความภาษาไทยต่อไปนี้:
รูปแบบการสรุป: {style_instruction}

[ข้อความต้นฉบับ]:
{src_text}

[ข้อความสรุป]:"""

    data = {
        "model": THAILLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    try:
        response = requests.post(THAILLM_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            res_json = response.json()
            raw_content = res_json["choices"][0]["message"]["content"]
            
            # Remove <think>...</think> reasoning tags
            cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
            return {
                "ok": True,
                "summary": cleaned_content,
                "src_length": len(src_text),
                "summary_length": len(cleaned_content),
                "model": THAILLM_MODEL,
                "errmsg": None
            }
        else:
            return {
                "ok": False,
                "summary": "",
                "src_length": len(src_text),
                "summary_length": 0,
                "model": THAILLM_MODEL,
                "errmsg": f"ThaiLLM error ({response.status_code}): {response.text}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "ok": False,
            "summary": "",
            "src_length": len(src_text),
            "summary_length": 0,
            "model": THAILLM_MODEL,
            "errmsg": f"Network error: {str(e)}"
        }
    except Exception as e:
        return {
            "ok": False,
            "summary": "",
            "src_length": len(src_text),
            "summary_length": 0,
            "model": THAILLM_MODEL,
            "errmsg": f"Internal error: {str(e)}"
        }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    if request.is_json:
        data = request.get_json() or {}
        src_text = data.get("src", "").strip()
        length_style = data.get("style", "medium")
    else:
        src_text = request.form.get("src", "").strip()
        length_style = request.form.get("style", "medium")

    if not src_text:
        result = {
            "ok": False,
            "summary": "",
            "src_length": 0,
            "summary_length": 0,
            "model": THAILLM_MODEL,
            "errmsg": "กรุณากรอกข้อความภาษาไทยที่ต้องการสรุปความ"
        }
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify(result), 400
        return render_template("index.html", result=result, src_text=src_text, style=length_style), 400

    # Call ThaiLLM Pathumma exclusively
    result = call_thaillm_summarize(src_text, length_style)

    wants_json = (
        request.headers.get("Accept") == "application/json"
        or request.args.get("format") == "json"
        or request.is_json
    )
    if wants_json:
        status_code = 200 if result["ok"] else 500
        return jsonify(result), status_code

    return render_template("index.html", result=result, src_text=src_text, style=length_style)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
