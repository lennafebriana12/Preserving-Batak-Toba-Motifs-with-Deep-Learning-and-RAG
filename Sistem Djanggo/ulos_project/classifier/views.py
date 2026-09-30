import os
import json
import time
import uuid
import requests
import numpy as np
import tensorflow as tf
from google import genai
from google.oauth2 import service_account
from PIL import Image
from pathlib import Path



from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Load .env file jika ada (untuk development lokal)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# =========================
# KONFIGURASI
# =========================
MODEL_PATH = os.path.join(settings.BASE_DIR, "../ulos_final_model.h5")
# N8N_WEBHOOK_URL = "https://n8n-1.saturn.petra.ac.id/webhook-test/tanya-rag"
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "https://n8n-1.saturn.petra.ac.id/webhook-test/tanya-rag")
# # SETUP GEMINI API KEY DI SINI
# genai.configure(api_key=GEMINI_API_KEY)

# SETUP SERVICE ACCOUNT GOOGLE CLOUD DI SINI
SERVICE_ACCOUNT_PATH = os.path.join(settings.BASE_DIR, "../n8n-api-keys-499101-c0270bfd2f80.json")

def get_vertex_client():
    """
    Mengonfigurasi kredensial Google Generative AI (Vertex AI) menggunakan Service Account JSON.
    """
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        try:
            # Set environment variable so the SDK detects it
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
            client = genai.Client(
                vertexai=True,
                project="n8n-api-keys-499101",
                location="us-central1"
            )
            return client
        except Exception as e:
            print(f"Gagal mengonfigurasi Vertex AI client: {e}")
    else:
        print(f"Berkas Service Account tidak ditemukan di: {SERVICE_ACCOUNT_PATH}")
    return None



GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_REQUEST_TIMEOUT = 30
GEMINI_MAX_RETRY_DELAY_SECONDS = 2

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3
USE_ANTIALIAS_RESIZE = True

# Threshold sesuai notebook
NOT_ULOS_THRESHOLD = 0.40
VALID_ULOS_THRESHOLD = 0.71

# Filter gambar polos / satu warna
PLAIN_STD_THRESHOLD = 8.0
PLAIN_UNIQUE_THRESHOLD = 10
PLAIN_EDGE_THRESHOLD = 2.0

CLASS_NAMES = [
    "Antak Antak", "Bintang Maratur", "Harungguan", "Mangiring",
    "Marinjam Sisi", "Pina Lobu Lobu", "Pinussaan", "Ragi Hidup",
    "Ragi Hotang", "Runjak", "Sadum", "Sibolang",
    "Sitolu Tuho", "Suri Suri",
]

TOP_K = 3

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# =========================
# HELPER GEMINI (GATE 2)
# =========================
def analyze_with_gemini(image_path):
    """
    Fallback identifikasi objek non-ulos menggunakan Google Gemini di Vertex AI.
    """
    client = get_vertex_client()
    if not client:
        print("Error Gemini API: Gagal mengonfigurasi Vertex AI client.")
        return "Benda tidak teridentifikasi"

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            prompt = (
                "Identifikasi objek utama pada gambar ini. "
                "Jawab dalam bahasa Indonesia. "
                "Berikan hanya nama objek utama, maksimal 5 kata. "
                "Jika objek tampak seperti makanan, kendaraan, pakaian, alat, hewan, manusia, "
                "atau benda rumah tangga, sebutkan kategori paling masuk akal. "
                "Jangan menjawab 'benda tidak teridentifikasi' kecuali gambar benar-benar tidak jelas."
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[prompt, img]
            )

        text = getattr(response, "text", "").strip()
        if text:
            return text
        print("Gemini response kosong.")
    except Exception as e:
        print(f"Error Gemini API via Vertex AI: {e}")

    return "Benda tidak teridentifikasi"




# =========================
# HELPER CNN (GATE 1)
# =========================
def load_raw_image_rgb(image_path):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(
        image_bytes,
        channels=CHANNELS,
        expand_animations=False,
    )
    image.set_shape([None, None, CHANNELS])

    image = tf.image.resize(
        image,
        IMG_SIZE,
        method="bilinear",
        antialias=USE_ANTIALIAS_RESIZE,
    )

    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    image = tf.ensure_shape(image, (IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    return image


def analyze_plain_image(raw_image):
    arr = raw_image.numpy().astype(np.uint8)

    std_all = float(arr.std())
    quant = (arr // 16).reshape(-1, 3)
    unique_colors = int(np.unique(quant, axis=0).shape[0])

    arr_f = arr.astype(np.float32)
    dx = float(np.abs(np.diff(arr_f, axis=1)).mean()) if arr_f.shape[1] > 1 else 0.0
    dy = float(np.abs(np.diff(arr_f, axis=0)).mean()) if arr_f.shape[0] > 1 else 0.0
    edge_strength = (dx + dy) / 2.0

    is_plain = (
        (std_all < PLAIN_STD_THRESHOLD and unique_colors <= PLAIN_UNIQUE_THRESHOLD)
        or
        (unique_colors <= 6 and edge_strength < PLAIN_EDGE_THRESHOLD)
    )

    return {
        "is_plain_color": bool(is_plain),
        "pixel_std": std_all,
        "unique_colors_4bit": unique_colors,
        "edge_strength": float(edge_strength),
    }


def preprocess_image_for_inference(image_path):
    raw_image = load_raw_image_rgb(image_path)
    image = tf.cast(raw_image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image, raw_image


def predict_single_image(image_path, top_k=TOP_K):
    model_input, raw_image = preprocess_image_for_inference(image_path)
    plain_info = analyze_plain_image(raw_image)

    if plain_info["is_plain_color"]:
        return {
            "filename": os.path.basename(image_path),
            "decision": "ditolak_gambar_satu_warna",
            "result_text": "Gambar ditolak karena terlalu polos atau hampir satu warna, sehingga tidak bisa diprediksi.",
            "reason": "plain_color_image",
            "top1_class": None,
            "top1_score": 0.0,
            "top2_class": None,
            "top2_score": 0.0,
            "margin_top1_top2": 0.0,
            "top_results": [],
            "plain_image_check": plain_info,
            "gemini_result": None
        }

    probs = model.predict(model_input, verbose=0)[0]
    order = np.argsort(probs)[::-1]

    top_results = []
    for idx in order[:top_k]:
        top_results.append({
            "class_name": CLASS_NAMES[int(idx)],
            "score": float(probs[int(idx)]),
        })

    top1 = top_results[0]
    top2 = top_results[1] if len(top_results) > 1 else {"class_name": None, "score": 0.0}
    margin = float(top1["score"] - top2["score"])

    # Logika keputusan Gate 1
    if top1["score"] <= NOT_ULOS_THRESHOLD:
        decision = "bukan_ulos"
        result_text = "Gambar ini bukan ulos atau bukan salah satu dari 14 kelas ulos."
        reason = "score_0_to_0_40"

    elif top1["score"] < VALID_ULOS_THRESHOLD:
        decision = "mungkin_ulos_upload_ulang"
        result_text = "Gambar yang anda upload buram atau gambar tersebut bukan ulos. Silakan upload ulang dengan foto yang lebih jelas."
        reason = "score_0_41_to_0_70"

    else:
        decision = "ulos_terdeteksi"
        result_text = f"Gambar ulos terdeteksi sebagai {top1['class_name']}."
        reason = "score_0_71_up"

    return {
        "filename": os.path.basename(image_path),
        "decision": decision,
        "result_text": result_text,
        "reason": reason,
        "top1_class": top1["class_name"],
        "top1_score": float(top1["score"]),
        "top2_class": top2["class_name"],
        "top2_score": float(top2["score"]),
        "margin_top1_top2": margin,
        "top_results": top_results,
        "plain_image_check": plain_info,
        "gemini_result": None # Akan diisi di views kalau decision-nya 'bukan_ulos'
    }


def save_uploaded_file(uploaded_file):
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    ext = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(upload_dir, safe_name)

    with open(file_path, "wb+") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    image_url = f"{settings.MEDIA_URL}uploads/{safe_name}"
    return file_path, image_url


def print_prediction_terminal(pred, confidence_percentage, image_url=""):
    print("=" * 80)
    print("HASIL PREDIKSI GAMBAR")
    print(f"File            : {pred['filename']}")
    if image_url:
        print(f"Image URL       : {image_url}")
    print(f"Decision        : {pred['decision']}")
    print(f"Hasil           : {pred['result_text']}")
    print(f"Confidence      : {confidence_percentage:.2f}%")
    print(f"Reason          : {pred['reason']}")
    print(
        f"Plain check     : std={pred['plain_image_check']['pixel_std']:.4f}, "
        f"unique4bit={pred['plain_image_check']['unique_colors_4bit']}, "
        f"edge={pred['plain_image_check']['edge_strength']:.4f}"
    )

    if pred["top1_class"] is not None:
        print(f"Prediksi utama  : {pred['top1_class']} ({pred['top1_score']:.6f})")
    
    if pred.get("gemini_result"):
        print(f"Gemini Analysis : {pred['gemini_result']}")
        
    print("=" * 80)


def build_prompt_for_n8n(user_message, pred, confidence_percentage):
    if not user_message:
        return ""

    decision = pred.get("decision")
    
    if decision == "ulos_terdeteksi":
        context = f"[Konteks: Gambar terdeteksi sebagai Ulos {pred['top1_class']} (Confidence: {confidence_percentage:.2f}%)]"
    elif decision == "mungkin_ulos_upload_ulang":
        gemini_res = pred.get("gemini_result", "benda tidak jelas")
        context = f"[Konteks: Gambar buram/kurang jelas untuk diprediksi (Kemungkinan: {gemini_res}). Minta user secara ramah untuk mengunggah foto kain Ulos yang jelas, terang, dan tidak buram agar dapat diidentifikasi.]"
    elif decision == "ditolak_gambar_satu_warna":
        context = "[Konteks: Gambar ditolak karena terlalu polos/satu warna. Jelaskan bahwa sistem memerlukan detail motif/tekstur kain Ulos untuk prediksi dan minta user mengunggah foto kain Ulos yang memperlihatkan motifnya.]"
    elif decision == "bukan_ulos":
        gemini_res = pred.get("gemini_result", "bukan ulos")
        context = f"[Konteks: Gambar bukan Ulos (Terdeteksi sebagai: {gemini_res}). Informasikan secara ramah bahwa sistem mendeteksi benda tersebut dan ingatkan bahwa sistem ini khusus untuk klasifikasi Ulos.]"
    else:
        context = ""

    if context:
        return f"{context}\nPertanyaan: {user_message}"
    return user_message




def call_n8n(message):
    if not message:
        return ""

    try:
        payload = {"chatInput": message}
        headers = {"Content-Type": "application/json"}
        response = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data.get("reply", data.get("output", "Maaf, format balasan AI tidak sesuai."))

        return f"Maaf, AI gagal merespons. Status: {response.status_code}"
    except Exception as e:
        print(f"Error n8n: {e}")
        return "Maaf, gagal terhubung ke server AI."



# =========================
# VIEW UTAMA
# =========================
def home(request):
    # Bersihkan last_prediction dari session jika me-load halaman fresh (GET)
    if request.method == "GET":
        if 'last_prediction' in request.session:
            try:
                del request.session['last_prediction']
            except KeyError:
                pass

    if request.method == "POST" and request.FILES.get("image"):

        uploaded_file = request.FILES["image"]
        user_message = request.POST.get("message", "").strip()
        file_path = None

        try:
            file_path, _ = save_uploaded_file(uploaded_file)
            
            # --- GATE 1 ---
            pred = predict_single_image(file_path, top_k=TOP_K)
            confidence_percentage = round(float(pred["top1_score"]) * 100, 2) if pred["top1_class"] else 0.0

            # --- GATE 2: FALLBACK KE GEMINI KALAU BUKAN ULOS ---
            if pred["decision"] in ["bukan_ulos", "mungkin_ulos_upload_ulang"]:
                hasil_gemini = analyze_with_gemini(file_path)
                pred["gemini_result"] = hasil_gemini

                if pred["decision"] == "bukan_ulos":
                    pred["result_text"] = f"Bukan ulos. Berdasarkan analisis sistem, gambar ini adalah: {hasil_gemini}."
                else:
                    pred["result_text"] = f"Gambar belum terdeteksi sebagai ulos dengan yakin. Berdasarkan analisis sistem, gambar ini kemungkinan adalah: {hasil_gemini}."

            result = pred["result_text"]

            # Konversi file ke base64 Data URL
            import base64
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "image/jpeg"
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                image_url = f"data:{mime_type};base64,{encoded_string}"

            # Cetak detail lengkap ke terminal buat debugging
            print_prediction_terminal(pred, confidence_percentage, image_url=image_url[:100] + "...")

            # Simpan hasil klasifikasi terakhir di session untuk konteks chat n8n
            request.session['last_prediction'] = {
                "decision": pred["decision"],
                "top1_class": pred["top1_class"],
                "gemini_result": pred.get("gemini_result"),
                "confidence_percentage": confidence_percentage
            }
            request.session.modified = True

            # Chat ke n8n jika user_message ada (apapun keputusan klasifikasinya)
            n8n_reply = ""
            if user_message:
                n8n_prompt = build_prompt_for_n8n(user_message, pred, confidence_percentage)
                n8n_reply = call_n8n(n8n_prompt)
            else:
                # Jika user hanya mengirim gambar tanpa teks, sistem yang memberikan komentar ramah
                decision = pred.get("decision")
                gemini_res = pred.get("gemini_result", "benda tidak jelas")
                
                if decision == "ulos_terdeteksi":
                    n8n_reply = f"Gambar berhasil diidentifikasi! Berdasarkan hasil analisis, gambar yang Anda unggah terdeteksi sebagai **Ulos {pred['top1_class']}** dengan tingkat keyakinan (confidence score) sebesar **{confidence_percentage:.2f}%**."
                elif decision == "mungkin_ulos_upload_ulang":
                    n8n_reply = f"Gambar yang Anda unggah terdeteksi kurang jelas, buram, atau memiliki intensitas cahaya yang kurang untuk diidentifikasi dengan yakin (Kemungkinan terdeteksi sebagai: **{gemini_res}**). Mohon untuk mengambil dan mengunggah ulang foto yang lebih jelas, terang, dan tidak buram agar sistem dapat memprediksinya dengan tepat."
                elif decision == "bukan_ulos":
                    n8n_reply = f"Maaf, sistem mendeteksi bahwa gambar yang Anda unggah bukan merupakan kain Ulos. Gambar tersebut diidentifikasi sebagai **{gemini_res}**. Harap diingat bahwa sistem ini khusus dirancang untuk klasifikasi kain Ulos Batak. Silakan unggah foto kain Ulos yang valid."
                elif decision == "ditolak_gambar_satu_warna":
                    n8n_reply = "Maaf, gambar yang Anda unggah terlalu polos atau hanya memiliki satu warna. Sistem kami memerlukan detail motif dan tekstur kain Ulos untuk dapat melakukan prediksi. Silakan unggah foto kain Ulos yang memperlihatkan motifnya dengan jelas."


            # UI / AJAX Response
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse({
                    "status": "success",
                    "result": result,
                    "confidence": confidence_percentage,
                    "image_url": image_url,
                    "n8n_reply": n8n_reply,
                })

            return render(request, "classifier/index.html", {
                "result": result,
                "confidence": confidence_percentage,
                "image_url": image_url,
                "n8n_reply": n8n_reply,
            })

        except Exception as e:
            print(f"Error sistem: {e}")
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse({"status": "error", "result": "Terjadi kesalahan sistem.", "confidence": 0})
            return render(request, "classifier/index.html", {"result": "Terjadi kesalahan sistem.", "confidence": 0})

        finally:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"File sementara berhasil dihapus: {file_path}")
                except Exception as cleanup_err:
                    print(f"Gagal menghapus file sementara: {cleanup_err}")

    return render(request, "classifier/index.html")


# =========================
# CHAT KE N8N TANPA GAMBAR
# =========================
@csrf_exempt
def chat_n8n(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()

            # Ambil konteks prediksi terakhir dari session jika ada
            last_pred = request.session.get('last_prediction')
            if last_pred:
                n8n_prompt = build_prompt_for_n8n(
                    user_message,
                    last_pred,
                    last_pred.get("confidence_percentage", 0.0)
                )
            else:
                n8n_prompt = user_message

            payload = {"chatInput": n8n_prompt}
            headers = {"Content-Type": "application/json"}
            response = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                n8n_data = response.json()
                bot_reply = n8n_data.get("reply", n8n_data.get("output", "Maaf, format balasan AI tidak sesuai."))
                return JsonResponse({"status": "success", "reply": bot_reply})

            return JsonResponse({"status": "error", "reply": f"Error dari n8n: {response.status_code}"})

        except Exception as e:
            print(f"Error Chat RAG: {e}")
            return JsonResponse({"status": "error", "reply": "Maaf, server sedang sibuk atau n8n mati."})


    return JsonResponse({"status": "invalid"}, status=400)
