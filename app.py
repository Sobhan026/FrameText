# -*- coding: utf-8 -*-
import os
import re
import io
import json
import time
import base64
from typing import Dict, List, Tuple, Any

# ---- Paddle / network behavior ----
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import numpy as np
import cv2
import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from threading import Lock

from paddleocr import PaddleOCR
from langdetect import detect, DetectorFactory

import argostranslate.package
import argostranslate.translate

import arabic_reshaper
from bidi.algorithm import get_display

DetectorFactory.seed = 0


_OCR = None
_OCR_LOCK = Lock()


def ensure_uint8_rgb(image_np: np.ndarray) -> np.ndarray:
    if not isinstance(image_np, np.ndarray):
        raise gr.Error("Invalid image format.")
    if image_np.dtype != np.uint8:
        mx = float(np.max(image_np)) if image_np.size else 0.0
        if mx <= 1.0:
            image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_np)


def shrink_if_needed(img_rgb: np.ndarray, max_side: int = 2300) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_rgb
    s = max_side / float(m)
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    return cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)


MODEL_PRESETS = {
    "OpenAI": ["gpt-5", "gpt-5-mini", "gpt-4o"],
    "DeepSeek": ["deepseek-reasoner", "deepseek-chat"],
    "Gemini": ["gemini-2.5-pro", "gemini-2.5-flash"],
}

OCR_BACKENDS = [
    ("Paddle (Local)", "paddle_local"),
    ("Google Vision API", "google_vision"),
    ("Azure Read API", "azure_read"),
    ("OpenAI Vision (Hybrid: boxes from Paddle)", "openai_hybrid"),
]

LANG_CHOICES = [
    ("Auto", "auto"),
    ("English", "en"),
    ("Persian", "fa"),
    ("Arabic", "ar"),
    ("Turkish", "tr"),
    ("Russian", "ru"),
    ("French", "fr"),
    ("German", "de"),
    ("Spanish", "es"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
]


# -------------------- Geometry / Drawing --------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
    return rect


def get_res_dict(res_obj: Any) -> Dict:
    if hasattr(res_obj, "json"):
        j = getattr(res_obj, "json")
        if callable(j):
            j = j()
        if isinstance(j, dict):
            return j.get("res", j)
    if hasattr(res_obj, "res"):
        r = getattr(res_obj, "res")
        if isinstance(r, dict):
            return r
    if isinstance(res_obj, dict):
        return res_obj.get("res", res_obj)
    raise TypeError("Could not parse PaddleOCR result format.")


def prepare_rtl_text(text: str, lang_code: str) -> str:
    if lang_code in {"fa", "ar", "ur"}:
        return get_display(arabic_reshaper.reshape(text))
    return text


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    words = text.split()
    if len(words) > 1:
        lines = []
        line = words[0]
        for w in words[1:]:
            test = f"{line} {w}"
            l, t, r, b = draw.textbbox((0, 0), test, font=font)
            if (r - l) <= max_width:
                line = test
            else:
                lines.append(line)
                line = w
        lines.append(line)
        return "\n".join(lines)

    # char-level wrap for single long token
    lines = []
    line = ""
    for ch in text:
        test = line + ch
        l, t, r, b = draw.textbbox((0, 0), test, font=font)
        if (r - l) <= max_width or line == "":
            line = test
        else:
            lines.append(line)
            line = ch
    if line:
        lines.append(line)
    return "\n".join(lines)


def fit_font_and_text(text: str, font_path: str, box_w: int, box_h: int):
    test_img = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(test_img)

    for size in range(max(12, int(box_h * 0.9)), 7, -1):
        font = ImageFont.truetype(font_path, size=size)
        wrapped = wrap_text(draw, text, font, int(box_w * 0.95))
        spacing = max(2, size // 6)
        l, t, r, b = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align="center")
        tw, th = r - l, b - t
        if tw <= int(box_w * 0.98) and th <= int(box_h * 0.95):
            return font, wrapped, tw, th, spacing

    # fallback
    font = ImageFont.truetype(font_path, size=10)
    l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    return font, text, (r - l), (b - t), 2


def overlay_patch_perspective(base_bgr: np.ndarray, patch_rgba: np.ndarray, quad: np.ndarray) -> np.ndarray:
    H, W = base_bgr.shape[:2]
    ph, pw = patch_rgba.shape[:2]

    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    dst = quad.astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        patch_rgba, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    out = base_bgr.astype(np.float32)
    over = warped[:, :, :3].astype(np.float32)
    out = over * alpha + out * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def pick_text_color_rgba(clean_bgr: np.ndarray, poly: np.ndarray):
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(clean_bgr.shape[1], x + w), min(clean_bgr.shape[0], y + h)
    roi = clean_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return (20, 20, 20, 235)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return (20, 20, 20, 235) if float(gray.mean()) > 145 else (245, 245, 245, 235)


def draw_text_in_quad(base_bgr: np.ndarray, quad: np.ndarray, text: str, font_path: str, rgba=(20, 20, 20, 235)) -> np.ndarray:
    quad = order_points(quad)

    w = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    h = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))

    if w < 8 or h < 8 or not text.strip():
        return base_bgr

    patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)

    font, wrapped, tw, th, spacing = fit_font_and_text(text, font_path, w, h)
    draw.multiline_text(
        ((w - tw) / 2, (h - th) / 2),
        wrapped,
        font=font,
        fill=rgba,
        align="center",
        spacing=spacing
    )

    return overlay_patch_perspective(base_bgr, np.array(patch), quad)


# -------------------- OCR / Translation --------------------
def parse_ocr_items(ocr_result) -> List[Tuple[str, np.ndarray]]:
    items: List[Tuple[str, np.ndarray]] = []
    for r in ocr_result:
        d = get_res_dict(r)
        texts = d.get("rec_texts", [])
        polys = d.get("rec_polys", [])
        for t, p in zip(texts, polys):
            txt = str(t).strip()
            if not txt:
                continue
            poly = np.array(p, dtype=np.float32)
            if poly.shape != (4, 2):
                continue
            items.append((txt, poly))
    items = sorted(items, key=lambda x: (float(np.min(x[1][:, 1])), float(np.min(x[1][:, 0]))))
    return items


def ensure_argos_model(src: str, tgt: str):
    try:
        _ = argostranslate.translate.get_translation_from_codes(src, tgt)
        return
    except Exception:
        pass

    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next((p for p in available if p.from_code == src and p.to_code == tgt), None)
    if pkg is None:
        raise RuntimeError(f"Argos model for {src}->{tgt} not found.")
    model_path = pkg.download()
    argostranslate.package.install_from_path(model_path)


def detect_src_lang_offline(texts: List[str]) -> str:
    text = " ".join(texts[:30]).strip()
    if not text:
        return "en"
    try:
        code = detect(text)
    except Exception:
        code = "en"

    mapping = {
        "fa": "fa", "ar": "ar", "en": "en", "tr": "tr", "ru": "ru",
        "fr": "fr", "de": "de", "es": "es", "it": "it", "pt": "pt",
        "zh-cn": "zh", "zh-tw": "zh", "ja": "ja", "ko": "ko"
    }
    return mapping.get(code, "en")


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()


def extract_json_array(raw: str):
    s = _strip_code_fences(raw)

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM output is not valid JSON array.")
    obj = json.loads(s[start:end + 1])
    if not isinstance(obj, list):
        raise ValueError("Parsed JSON is not an array.")
    return obj


def extract_json_object(raw: str):
    s = _strip_code_fences(raw)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM output is not valid JSON object.")
    obj = json.loads(s[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def _http_error_with_body(resp: requests.Response, prefix: str):
    try:
        detail = resp.json()
    except Exception:
        detail = resp.text
    raise RuntimeError(f"{prefix} | HTTP {resp.status_code} | {detail}")


def _extract_openai_text(resp_json: Dict) -> str:
    ot = resp_json.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot

    texts: List[str] = []
    for item in resp_json.get("output", []) or []:
        for c in item.get("content", []) or []:
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t)

    if texts:
        return "\n".join(texts)

    raise RuntimeError(f"OpenAI response text not found. keys={list(resp_json.keys())}")


def llm_batch_translate(
    provider: str,
    model: str,
    api_key: str,
    segments: List[Dict],
    source_lang: str,
    target_lang: str,
    timeout: int = 120
) -> Dict[int, str]:

    rules = (
        "Rules:\n"
        "- Preserve numbers, URLs, emails, hashtags, @mentions.\n"
        "- Keep named entities consistent.\n"
        "- Keep output concise for overlay on image boxes.\n"
        "- No explanations."
    )

    if provider == "OpenAI":
        system = (
            "You are a strict OCR translation engine.\n"
            "Return ONLY JSON in this exact schema:\n"
            '{"translations":[{"id":0,"translation":"..."}]}\n'
            + rules
        )
    else:
        system = (
            "You are a strict OCR translation engine.\n"
            "Return ONLY JSON array exactly in this schema:\n"
            '[{"id": 0, "translation": "..."}]\n'
            + rules
        )

    payload = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "segments": segments
    }
    user_text = json.dumps(payload, ensure_ascii=False)

    if provider == "OpenAI":
        url = "https://api.openai.com/v1/responses"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        # IMPORTANT: top-level must be object for json_schema
        schema = {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "translation": {"type": "string"}
                        },
                        "required": ["id", "translation"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["translations"],
            "additionalProperties": False
        }

        body = {
            "model": model,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system}]
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}]
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ocr_translations",
                    "schema": schema,
                    "strict": True
                }
            }
        }

        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code >= 400:
            _http_error_with_body(r, "OpenAI Responses API error")
        content = _extract_openai_text(r.json())

        obj = extract_json_object(content)
        arr = obj.get("translations")
        if not isinstance(arr, list):
            raise RuntimeError("OpenAI JSON format invalid: 'translations' must be an array.")

    elif provider == "DeepSeek":
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text}
            ]
        }
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code >= 400:
            _http_error_with_body(r, "DeepSeek API error")
        content = r.json()["choices"][0]["message"]["content"]
        arr = extract_json_array(content)

    elif provider == "Gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": api_key}
        prompt = f"{system}\n\n{user_text}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        r = requests.post(url, params=params, json=body, timeout=timeout)
        if r.status_code >= 400:
            _http_error_with_body(r, "Gemini API error")
        data = r.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        arr = extract_json_array(content)

    else:
        raise ValueError("Unknown provider.")

    out: Dict[int, str] = {}
    for item in arr:
        i = int(item["id"])
        out[i] = str(item["translation"])
    return out


# -------------------- OCR Backends --------------------
def _img_to_png_bytes_rgb(img_rgb: np.ndarray) -> bytes:
    pil = Image.fromarray(img_rgb)
    buff = io.BytesIO()
    pil.save(buff, format="PNG")
    return buff.getvalue()


def _data_url_from_rgb(img_rgb: np.ndarray) -> str:
    b = _img_to_png_bytes_rgb(img_rgb)
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")


def _vertices_to_quad(vertices) -> np.ndarray:
    pts = []
    for v in vertices[:4]:
        if isinstance(v, dict):
            x = int(v.get("x", 0) or 0)
            y = int(v.get("y", 0) or 0)
        else:
            x, y = 0, 0
        pts.append([x, y])

    while len(pts) < 4:
        pts.append(pts[-1] if pts else [0, 0])

    return np.array(pts, dtype=np.float32)


def ocr_google_vision(img_rgb: np.ndarray, api_key: str, timeout: int = 120) -> List[Tuple[str, np.ndarray]]:
    if not api_key.strip():
        raise gr.Error("OCR API key is required for Google Vision.")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key.strip()}"
    payload = {
        "requests": [{
            "image": {"content": base64.b64encode(_img_to_png_bytes_rgb(img_rgb)).decode("utf-8")},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
        }]
    }

    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise gr.Error(f"Google Vision Error: {r.status_code} | {r.text[:300]}")
    data = r.json()
    resp = (data.get("responses") or [{}])[0]
    if "error" in resp:
        raise gr.Error(f"Google Vision Error: {resp['error']}")

    items: List[Tuple[str, np.ndarray]] = []
    anns = resp.get("textAnnotations", [])

    for ann in anns[1:]:
        txt = str(ann.get("description", "")).strip()
        if not txt:
            continue
        poly = _vertices_to_quad(ann.get("boundingPoly", {}).get("vertices", []))
        if poly.shape == (4, 2):
            items.append((txt, poly))

    items = sorted(items, key=lambda x: (float(np.min(x[1][:, 1])), float(np.min(x[1][:, 0]))))
    return items


def ocr_azure_read(
    img_rgb: np.ndarray,
    endpoint: str,
    api_key: str,
    api_version: str = "v3.2",
    timeout: int = 120
) -> List[Tuple[str, np.ndarray]]:
    if not api_key.strip():
        raise gr.Error("OCR API key is required for Azure Read.")
    if not endpoint.strip():
        raise gr.Error("Azure endpoint is required for Azure Read. Example: https://<name>.cognitiveservices.azure.com")

    endpoint = endpoint.strip().rstrip("/")
    submit_url = f"{endpoint}/vision/{api_version}/read/analyze"

    headers = {
        "Ocp-Apim-Subscription-Key": api_key.strip(),
        "Content-Type": "application/octet-stream"
    }

    img_bytes = _img_to_png_bytes_rgb(img_rgb)
    r = requests.post(submit_url, headers=headers, data=img_bytes, timeout=timeout)
    if r.status_code >= 400:
        raise gr.Error(f"Azure Read Error: {r.status_code} | {r.text[:300]}")

    op_loc = r.headers.get("Operation-Location")
    if not op_loc:
        raise gr.Error("Azure Read Error: Missing Operation-Location header in the response.")

    poll_headers = {"Ocp-Apim-Subscription-Key": api_key.strip()}
    result_json = None
    for _ in range(120):  # up to ~60 sec
        time.sleep(0.5)
        pr = requests.get(op_loc, headers=poll_headers, timeout=timeout)
        if pr.status_code >= 400:
            raise gr.Error(f"Azure Read Poll Error: {pr.status_code} | {pr.text[:300]}")
        result_json = pr.json()
        status = str(result_json.get("status", "")).lower()
        if status == "succeeded":
            break
        if status == "failed":
            raise gr.Error(f"Azure Read failed: {result_json}")

    if not result_json or str(result_json.get("status", "")).lower() != "succeeded":
        raise gr.Error("Azure Read timeout.")

    items: List[Tuple[str, np.ndarray]] = []
    pages = result_json.get("analyzeResult", {}).get("readResults", [])
    for page in pages:
        for line in page.get("lines", []):
            txt = str(line.get("text", "")).strip()
            bb = line.get("boundingBox", [])
            if not txt or len(bb) != 8:
                continue
            poly = np.array(
                [[bb[0], bb[1]], [bb[2], bb[3]], [bb[4], bb[5]], [bb[6], bb[7]]],
                dtype=np.float32
            )
            items.append((txt, poly))

    items = sorted(items, key=lambda x: (float(np.min(x[1][:, 1])), float(np.min(x[1][:, 0]))))
    return items


def crop_perspective_from_quad(img_rgb: np.ndarray, quad: np.ndarray, min_side: int = 28) -> np.ndarray:
    q = order_points(quad)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    w = max(w, min_side)
    h = max(h, min_side)

    src = q.astype(np.float32)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(
        img_rgb, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    if min(crop.shape[:2]) < 60:
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    return crop


def openai_ocr_text_for_crop(crop_rgb: np.ndarray, api_key: str, model: str, timeout: int = 120) -> str:
    data_url = _data_url_from_rgb(crop_rgb)

    system = (
        "You are an OCR engine. "
        "Extract ONLY the text visible in the image. "
        "Do not translate. Do not explain. "
        "If unreadable/no text, return empty string."
    )

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract exact text from this cropped region. Output text only."},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ]
            }
        ]
    }

    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code >= 400:
        _http_error_with_body(r, "OpenAI OCR error")

    data = r.json()
    text = _extract_openai_text(data).strip().strip('"').strip("'")
    if text.lower() in {"no text", "no_text", "none", "null", "empty", "n/a"}:
        return ""
    return text


def ocr_openai_hybrid(
    img_bgr: np.ndarray,
    paddle_items: List[Tuple[str, np.ndarray]],
    openai_api_key: str,
    openai_model: str = "gpt-4o-mini",
    timeout: int = 120
) -> List[Tuple[str, np.ndarray]]:
    if not openai_api_key.strip():
        raise gr.Error("Please enter the OCR API key for OpenAI OCR.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out_items: List[Tuple[str, np.ndarray]] = []

    for _, poly in paddle_items:
        try:
            crop = crop_perspective_from_quad(img_rgb, poly)
            txt = openai_ocr_text_for_crop(crop, openai_api_key.strip(), openai_model, timeout=timeout).strip()
        except Exception:
            txt = ""

        out_items.append((txt, poly))

    out_items = sorted(out_items, key=lambda x: (float(np.min(x[1][:, 1])), float(np.min(x[1][:, 0]))))
    return out_items


# global OCR singleton
_OCR = None
def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    return _OCR


def run_ocr_backend(
    img_bgr: np.ndarray,
    ocr_backend: str,
    ocr_api_key: str,
    azure_endpoint: str,
    openai_ocr_model: str,
    fallback_openai_key: str
) -> List[Tuple[str, np.ndarray]]:
    ocr = get_ocr()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if ocr_backend == "paddle_local":
        try:
            with _OCR_LOCK:
                ocr_result = ocr.predict(img_rgb)
        except Exception as e:
            raise gr.Error(f"Paddle OCR failed (model download/network/local model issue): {e}")
        return parse_ocr_items(ocr_result)

    if ocr_backend == "google_vision":
        return ocr_google_vision(img_rgb, api_key=ocr_api_key)

    if ocr_backend == "azure_read":
        return ocr_azure_read(img_rgb, endpoint=azure_endpoint, api_key=ocr_api_key)

    if ocr_backend == "openai_hybrid":
        try:
            with _OCR_LOCK:
                ocr_result = ocr.predict(img_rgb)
        except Exception as e:
            raise gr.Error(f"Paddle box detection failed for OpenAI Hybrid: {e}")

        paddle_items = parse_ocr_items(ocr_result)

        key = (ocr_api_key or "").strip() or (fallback_openai_key or "").strip()
        model = (openai_ocr_model or "").strip() or "gpt-4o-mini"

        return ocr_openai_hybrid(
            img_bgr=img_bgr,
            paddle_items=paddle_items,
            openai_api_key=key,
            openai_model=model
        )

    raise gr.Error("Unsupported OCR backend.")


# -------------------- Main Pipeline --------------------
def process_image(
    image_np,
    mode,
    provider,
    model,
    custom_model,
    api_key,
    auto_source,
    src_lang,
    tgt_lang,
    font_path,
    ocr_backend,
    ocr_api_key,
    azure_endpoint,
    openai_ocr_model
):
    if image_np is None:
        raise gr.Error("Please upload an image first.")
    if mode == "LLM" and not api_key.strip():
        raise gr.Error("Translation API key is required for LLM mode.")
    if not font_path.strip():
        raise gr.Error("Please provide a font path (e.g., Arial or Vazirmatn).")

    # OCR API key is required for API-based OCR backends (except fallback behavior in openai_hybrid)
    if ocr_backend in {"google_vision", "azure_read"} and not (ocr_api_key or "").strip():
        raise gr.Error("Please enter the OCR API key for the selected OCR backend.")
    if ocr_backend == "azure_read" and not (azure_endpoint or "").strip():
        raise gr.Error("Please enter the Azure endpoint for Azure Read.")
    if ocr_backend == "openai_hybrid":
        if not ((ocr_api_key or "").strip() or (api_key or "").strip()):
            raise gr.Error("OpenAI Hybrid requires either an OCR API key or an OpenAI translation API key.")

    image_np = ensure_uint8_rgb(image_np)

    # RGB -> BGR
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # OCR
    items = run_ocr_backend(
        img_bgr=img,
        ocr_backend=ocr_backend,
        ocr_api_key=ocr_api_key,
        azure_endpoint=azure_endpoint,
        openai_ocr_model=openai_ocr_model,
        fallback_openai_key=api_key if provider == "OpenAI" else ""
    )

    if not items:
        return image_np, "No text was detected."

    # Filter out empty OCR results
    items = [(t, p) for (t, p) in items if str(t).strip()]
    if not items:
        return image_np, "Bounding boxes were detected, but no usable OCR text was found."

    # mask + inpaint
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for _, poly in items:
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    clean = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    source_effective = "auto" if auto_source else src_lang

    # translate
    translations: Dict[int, str] = {}
    if mode == "Classic (Argos Offline)":
        texts = [t for t, _ in items]
        if source_effective == "auto":
            source_effective = detect_src_lang_offline(texts)

        ensure_argos_model(source_effective, tgt_lang)

        for i, (txt, _) in enumerate(items):
            tr = argostranslate.translate.translate(txt, source_effective, tgt_lang)
            translations[i] = tr

    else:  # LLM
        segs = []
        for i, (txt, _) in enumerate(items):
            prev_t = items[i - 1][0] if i > 0 else ""
            next_t = items[i + 1][0] if i < len(items) - 1 else ""
            segs.append({"id": i, "text": txt, "prev": prev_t, "next": next_t})

        real_model = custom_model.strip() or model

        try:
            translations = llm_batch_translate(
                provider=provider,
                model=real_model,
                api_key=api_key.strip(),
                segments=segs,
                source_lang=source_effective,
                target_lang=tgt_lang
            )
        except Exception as e:
            raise gr.Error(f"LLM translation failed: {e}")

    # render
    out = clean.copy()
    for i, (_, poly) in enumerate(items):
        tr = translations.get(i, "").strip()
        if not tr:
            continue
        draw_text = prepare_rtl_text(tr, tgt_lang)
        color = pick_text_color_rgba(clean, poly)
        out = draw_text_in_quad(out, poly, draw_text, font_path, color)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    log = (
        f"✅ Done | ocr={ocr_backend} | mode={mode} | src={source_effective} | "
        f"tgt={tgt_lang} | boxes={len(items)}"
    )

    out_rgb = shrink_if_needed(out_rgb, max_side=2300)
    out_rgb = np.ascontiguousarray(out_rgb, dtype=np.uint8)

    return out_rgb, log


# -------------------- UI Callbacks --------------------
def on_mode_change(mode):
    llm = mode == "LLM"
    return (
        gr.update(visible=llm),  # provider
        gr.update(visible=llm),  # model
        gr.update(visible=llm),  # custom_model
        gr.update(visible=llm),  # api_key
    )


def on_provider_change(provider):
    choices = MODEL_PRESETS.get(provider, [])
    value = choices[0] if choices else ""
    return gr.update(choices=choices, value=value)


def on_ocr_backend_change(ocr_backend):
    is_api = ocr_backend in {"google_vision", "azure_read", "openai_hybrid"}
    is_azure = ocr_backend == "azure_read"
    is_openai_hybrid = ocr_backend == "openai_hybrid"

    hint = ""
    if ocr_backend == "openai_hybrid":
        hint = "OpenAI Hybrid: Bounding boxes are detected by Paddle, and text for each box is extracted by OpenAI."
    elif ocr_backend == "google_vision":
        hint = "Google Vision: Both text and bounding boxes are provided by the Google Vision API."
    elif ocr_backend == "azure_read":
        hint = "Azure Read: Both text and bounding boxes are provided by the Azure Read API."
    else:
        hint = "Paddle Local: Both text and bounding boxes are extracted offline using PaddleOCR."

    return (
        gr.update(visible=is_api),           # ocr_api_key
        gr.update(visible=is_azure),         # azure_endpoint
        gr.update(visible=is_openai_hybrid), # openai_ocr_model
        gr.update(value=hint)                # ocr_hint
    )


with gr.Blocks(title="Frame Text") as demo:
    gr.Markdown(
        "<div style='text-align:center;font-size:30px;font-weight:700;margin-bottom:4px;'>Frame Text</div>"
        "<div style='text-align:center;opacity:.85;margin-bottom:14px;'>OCR ➜ Translate ➜ Put text back in-place</div>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            # -------- OCR Section --------
            gr.Markdown("### OCR")
            ocr_backend = gr.Dropdown(
                choices=OCR_BACKENDS,
                value="paddle_local",
                label="OCR Backend"
            )
            ocr_api_key = gr.Textbox(
                label="OCR API Key (for selected OCR backend)",
                type="password",
                visible=False
            )
            azure_endpoint = gr.Textbox(
                label="Azure Endpoint (for Azure Read)",
                placeholder="https://<resource>.cognitiveservices.azure.com",
                visible=False
            )
            openai_ocr_model = gr.Textbox(
                label="OpenAI OCR Model (for Hybrid)",
                value="gpt-4o-mini",
                visible=False
            )
            ocr_hint = gr.Markdown("Paddle Local: متن و باکس هر دو به‌صورت آفلاین از PaddleOCR.")

            # -------- Translation Section --------
            gr.Markdown("### Translation")
            mode = gr.Radio(
                ["LLM", "Classic (Argos Offline)"],
                value="LLM",
                label="Translation Engine"
            )
            provider = gr.Dropdown(
                choices=["OpenAI", "DeepSeek", "Gemini"],
                value="OpenAI",
                label="Provider",
                visible=True
            )
            model = gr.Dropdown(
                choices=MODEL_PRESETS["OpenAI"],
                value=MODEL_PRESETS["OpenAI"][0],
                label="Model",
                visible=True
            )
            custom_model = gr.Textbox(
                label="Custom model id (optional)",
                placeholder="e.g. gpt-5 / deepseek-reasoner / gemini-2.5-flash",
                visible=True
            )
            api_key = gr.Textbox(
                label="Translation API Key",
                type="password",
                visible=True
            )

            auto_source = gr.Checkbox(value=True, label="Auto detect source language")
            src_lang = gr.Dropdown(
                choices=LANG_CHOICES,
                value="en",
                label="Source language (used if auto=off)"
            )
            tgt_lang = gr.Dropdown(
                choices=LANG_CHOICES[1:],
                value="fa",
                label="Target language"
            )
            font_path = gr.Textbox(
                value="C:/Windows/Fonts/arial.ttf",
                label="Font path (TTF)"
            )

            run_btn = gr.Button("Translate Image", variant="primary")

        with gr.Column(scale=1):
            image_in = gr.Image(type="numpy", label="Input image")
            image_out = gr.Image(type="numpy", label="Output image")
            log = gr.Textbox(label="Log", lines=5)

    mode.change(
        on_mode_change,
        inputs=[mode],
        outputs=[provider, model, custom_model, api_key]
    )

    provider.change(
        on_provider_change,
        inputs=[provider],
        outputs=[model]
    )

    ocr_backend.change(
        on_ocr_backend_change,
        inputs=[ocr_backend],
        outputs=[ocr_api_key, azure_endpoint, openai_ocr_model, ocr_hint]
    )

    run_btn.click(
        process_image,
        inputs=[
            image_in, mode, provider, model, custom_model, api_key,
            auto_source, src_lang, tgt_lang, font_path,
            ocr_backend, ocr_api_key, azure_endpoint, openai_ocr_model
        ],
        outputs=[image_out, log],
        queue=True
    )

demo.queue(default_concurrency_limit=1, max_size=16)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )