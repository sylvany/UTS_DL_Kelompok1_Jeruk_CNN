from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import json
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # kurangi log TensorFlow di terminal
try:
    import tensorflow as tf
except ImportError:      # server hosting ringan: TensorFlow tidak dipasang, pakai model_cnn.tflite
    tf = None
import threading
from io import BytesIO
from pathlib import Path
from PIL import Image

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Model hasil training (jeruk_cnn_tensorflow.ipynb).
# Di komputer sendiri: model Keras (model_cnn.keras). Di hosting ringan tanpa TensorFlow: salinan TFLite
# (model_cnn.tflite) dari model yang sama, dibuat oleh sel "Ekspor untuk hosting" di notebook.
if tf is not None:
    model_cnn = tf.keras.models.load_model(BASE_DIR / "model_cnn.keras")
else:
    from ai_edge_litert.interpreter import Interpreter
    _interp = Interpreter(model_path=str(BASE_DIR / "model_cnn.tflite"))
    _interp.allocate_tensors()
    _kunci = threading.Lock()
    _w = np.load(BASE_DIR / "gradcam_bobot.npz")
    _W1, _W2 = _w["w1"], _w["w2"]

# urutan kelas & ukuran input dibaca dari metadata supaya sama persis dengan saat training
with open(BASE_DIR / "class_names.json", encoding="utf-8") as f:
    meta = json.load(f)
categories = meta["class_names"]
IMG_SIZE   = meta["img_size"]

# Fitur unik 2 (confidence threshold): ambang ditulis oleh notebook ke class_names.json (default 0.80)
AMBANG = float(meta.get("confidence_threshold", 0.80))

# Fitur unik 1 (Grad-CAM): peta fitur MaxPooling terakhir (sebelum GlobalAveragePooling) dan skor sebelum softmax
if tf is not None:
    _lapisan_akhir = model_cnn.layers[-1]
    _lapisan_peta  = [l for l in model_cnn.layers if isinstance(l, tf.keras.layers.MaxPooling2D)][-1]
    _grad_model    = tf.keras.Model(model_cnn.inputs, [_lapisan_peta.output, model_cnn.layers[-2].output])

labels = {
    "blackspot": "Blackspot (Bercak Hitam)",
    "canker":    "Canker (Kanker Jeruk)",
    "fresh":     "Fresh (Jeruk Segar)",
    "grenning":  "Grenning (Penyakit Kuning)"
}

def encode_image(img_array, is_gray=False):
    if is_gray:
        img_pil = Image.fromarray(img_array)
    else:
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def preprocess_cnn(img_bgr):
    # preprocessing harus sama dengan di training: RGB -> resize bilinear -> bagi 255
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if tf is not None:
        return tf.image.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return cv2.resize(img.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR) / 255.0

def _jalankan_tflite(x):
    """Mengembalikan (probabilitas, peta fitur 16x16x128, keluaran Dense 128) dari model TFLite."""
    with _kunci:
        _interp.set_tensor(_interp.get_input_details()[0]["index"], x[None, ...].astype(np.float32))
        _interp.invoke()
        keluaran = [_interp.get_tensor(d["index"])[0] for d in _interp.get_output_details()]
    proba = next(o for o in keluaran if o.ndim == 1 and o.shape[0] == len(categories))
    peta  = next(o for o in keluaran if o.ndim == 3)
    h     = next(o for o in keluaran if o.ndim == 1 and o.shape[0] != len(categories))
    return proba, peta, h

def classify_cnn(img_bgr):
    img = preprocess_cnn(img_bgr)
    if tf is not None:
        proba = model_cnn(img[tf.newaxis, ...], training=False).numpy()[0]
    else:
        proba = _jalankan_tflite(img)[0]
    pred  = int(np.argmax(proba))
    return categories[pred], round(float(proba[pred])*100, 1), proba.tolist()

def gradcam_cnn(img_bgr, kelas):
    """Grad-CAM untuk kelas tertentu. Mengembalikan citra 256x256 (BGR) dengan peta panas di atasnya."""
    x = preprocess_cnn(img_bgr)
    if tf is not None:
        x = x[tf.newaxis, ...]
        with tf.GradientTape() as tape:
            peta, h = _grad_model(x, training=False)
            skor = (tf.matmul(h, _lapisan_akhir.kernel) + _lapisan_akhir.bias)[:, kelas]   # skor sebelum softmax
        grad  = tape.gradient(skor, peta)
        bobot = tf.reduce_mean(grad, axis=(1, 2))
        cam   = tf.nn.relu(tf.reduce_sum(peta * bobot[:, None, None, :], axis=-1))[0]
        cam   = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
    else:
        # hasil yang sama dengan versi TensorFlow: gradien skor kelas terhadap tiap peta fitur
        # sama di semua posisi, sehingga bobot Grad-CAM dapat dihitung langsung dari bobot Dense
        _, peta, h = _jalankan_tflite(x)
        bobot = _W1 @ ((h > 0) * _W2[:, kelas])
        cam   = np.maximum((peta * bobot[None, None, :]).sum(-1), 0)
        cam   = cam / (cam.max() + 1e-8)
    cam   = cv2.resize(cam, (256, 256), interpolation=cv2.INTER_LINEAR)
    heat  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    dasar = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(dasar, 0.55, heat, 0.45, 0)

def cek_ambang(proba):
    """True bila probabilitas tertinggi mencapai ambang confidence."""
    return float(np.max(proba)) >= AMBANG

def find_best_circle(mask, img_shape):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(largest)
        return int(x), int(y), int(r)
    h, w = img_shape[:2]
    return w//2, h//2, min(h,w)//2-10

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/proses", methods=["POST"])
def proses():
    file    = request.files["gambar"]
    npimg   = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    hasil   = {}

    hasil["original"] = encode_image(img_bgr)
    h, w = img_bgr.shape[:2]
    hasil["info"] = {
        "ukuran": f"{w} x {h} px",
        "channel": "BGR (3 channel)",
        "total_piksel": f"{w*h:,}",
        "format": "JPEG/PNG"
    }

    img_noisy = img_bgr.copy()
    sp        = np.random.random(img_noisy.shape[:2])
    img_noisy[sp < 0.02] = 0
    img_noisy[sp > 0.98] = 255
    hasil["m2_noisy"]    = encode_image(img_noisy)
    img_denoised         = cv2.medianBlur(img_noisy, 5)
    hasil["m2_denoised"] = encode_image(img_denoised)
    img_resized          = cv2.resize(img_denoised, (256,256), interpolation=cv2.INTER_LINEAR)
    hasil["m2_resize"]   = encode_image(img_resized)

    img_blur = cv2.GaussianBlur(img_resized, (7,7), sigmaX=2)
    hasil["m3_blur"] = encode_image(img_blur)
    kernel_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img_sharp    = cv2.filter2D(img_resized, -1, kernel_sharp)
    hasil["m3_sharp"] = encode_image(img_sharp)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    hasil["m4_hsv"] = encode_image(img_hsv)

    mask_jeruk = cv2.bitwise_or(
        cv2.inRange(img_hsv, np.array([5,50,50]),   np.array([35,255,255])),
        cv2.bitwise_or(
            cv2.inRange(img_hsv, np.array([0,0,150]),  np.array([180,60,255])),
            cv2.inRange(img_hsv, np.array([35,30,30]), np.array([85,255,255]))
        )
    )
    kernel_e    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask_open   = cv2.morphologyEx(mask_jeruk, cv2.MORPH_OPEN,  kernel_e)
    mask_closed = cv2.morphologyEx(mask_open,  cv2.MORPH_CLOSE, kernel_e)
    mask_closed = cv2.dilate(mask_closed, np.ones((15,15), np.uint8), iterations=2)

    hasil["m4_mask"]      = encode_image(mask_closed, is_gray=True)
    hasil["m4_segmented"] = encode_image(cv2.bitwise_and(img_resized, img_resized, mask=mask_closed))

    kernel_vis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    mask_dilasi = cv2.dilate(mask_closed, kernel_vis, iterations=1)
    hasil["m4_dilasi"] = encode_image(mask_dilasi, is_gray=True)

    mask_erosi_in = cv2.erode(mask_closed, kernel_vis, iterations=1)
    mask_tepi = cv2.subtract(mask_dilasi, mask_erosi_in)  # dilasi - erosi = tepi
    hasil["m4_erosi"] = encode_image(mask_tepi, is_gray=True)

    gray_seg    = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray_seg, gray_seg, mask=mask_closed)
    edges_canny = cv2.Canny(gray_masked, 30, 90)
    hasil["m5_edges"] = encode_image(edges_canny, is_gray=True)

    overlay = img_resized.copy()
    overlay[edges_canny > 0] = [0,0,255]
    hasil["m5_overlay"] = encode_image(overlay)

    gray_blur_h = cv2.GaussianBlur(gray_seg, (9,9), 2)
    circles     = cv2.HoughCircles(gray_blur_h, cv2.HOUGH_GRADIENT,
                    dp=1.2, minDist=60, param1=40, param2=22,
                    minRadius=40, maxRadius=150)
    img_hough   = img_resized.copy()
    if circles is not None:
        best = sorted(np.round(circles[0]).astype(int), key=lambda c: c[2], reverse=True)[0]
        x, y, r = best
        cv2.circle(img_hough, (x,y), r, (0,255,0), 3)
        cv2.circle(img_hough, (x,y), 3, (0,0,255), 5)
        orange_circle = (x, y, r)
    else:
        cx, cy, cr = find_best_circle(mask_closed, img_resized.shape)
        cv2.circle(img_hough, (cx,cy), cr, (0,200,255), 3)
        cv2.circle(img_hough, (cx,cy), 3,  (0,0,255), 5)
        orange_circle = (cx, cy, cr)
    hasil["m5_hough"] = encode_image(img_hough)

    cx, cy, cr  = orange_circle
    circle_mask = np.zeros(edges_canny.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (cx,cy), cr, 255, -1)
    total_area  = int(np.sum(circle_mask > 0)) or 1
    damage_area = int(np.sum((edges_canny > 0) & (circle_mask > 0)))
    damage_pct  = round((damage_area/total_area)*100, 2)

    cat_cnn, conf_cnn, proba_cnn = classify_cnn(img_bgr)
    final_cat, final_conf, final_model = cat_cnn, conf_cnn, "CNN (Deep Learning)"
    diterima = cek_ambang(proba_cnn)

    img_result    = img_resized.copy()
    cv2.circle(img_result, (cx,cy), cr, (0,255,0), 2)
    contours_res, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_count = 0
    for cnt in contours_res:
        if cv2.contourArea(cnt) > 30:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(img_result, (bx,by), (bx+bw,by+bh), (0,0,255), 1)
            bbox_count += 1
    cv2.putText(img_result, labels[final_cat] if diterima else "Tidak yakin", (5,18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
    cv2.putText(img_result, f"Conf: {final_conf}%", (5,34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
    hasil["m5_result"]   = encode_image(img_result)
    hasil["damage_pct"]  = damage_pct
    hasil["bbox_count"]  = bbox_count

    hasil["m6_kategori"] = labels[cat_cnn]
    hasil["m6_conf"]     = float(conf_cnn)

    hasil["kategori"]    = labels[final_cat]
    hasil["confidence"]  = float(final_conf)
    hasil["model_used"]  = final_model
    hasil["conf_cnn"]    = float(conf_cnn)
    hasil["cat_cnn"]     = labels[cat_cnn]
    hasil["proba_cnn"]   = [float(x) for x in proba_cnn]
    hasil["status"]      = "tidak_yakin" if not diterima else ("rusak" if final_cat != "fresh" else "normal")
    hasil["damage_pct"]  = float(damage_pct)
    hasil["bbox_count"]  = int(bbox_count)

    # fitur unik: confidence threshold dan Grad-CAM
    hasil["diterima"] = bool(diterima)
    hasil["ambang"]   = round(AMBANG*100, 1)
    hasil["gradcam"]  = encode_image(gradcam_cnn(img_bgr, categories.index(cat_cnn)))

    return jsonify(hasil)

@app.route("/deteksi-cepat", methods=["POST"])
def deteksi_cepat():
    """Endpoint untuk deteksi cepat tanpa visualisasi step-by-step"""
    file    = request.files["gambar"]
    npimg   = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    hasil   = {}

    hasil["original"] = encode_image(img_bgr)

    img_resized  = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_blur     = cv2.GaussianBlur(img_resized, (7,7), sigmaX=2)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    mask_jeruk = cv2.bitwise_or(
        cv2.inRange(img_hsv, np.array([5,50,50]),   np.array([35,255,255])),
        cv2.bitwise_or(
            cv2.inRange(img_hsv, np.array([0,0,150]),  np.array([180,60,255])),
            cv2.inRange(img_hsv, np.array([35,30,30]), np.array([85,255,255]))
        )
    )
    kernel_e    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask_open   = cv2.morphologyEx(mask_jeruk, cv2.MORPH_OPEN,  kernel_e)
    mask_closed = cv2.morphologyEx(mask_open,  cv2.MORPH_CLOSE, kernel_e)
    mask_closed = cv2.dilate(mask_closed, np.ones((15,15), np.uint8), iterations=2)

    gray_seg    = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray_seg, gray_seg, mask=mask_closed)
    edges_canny = cv2.Canny(gray_masked, 30, 90)

    gray_blur_h = cv2.GaussianBlur(gray_seg, (9,9), 2)
    circles     = cv2.HoughCircles(gray_blur_h, cv2.HOUGH_GRADIENT,
                    dp=1.2, minDist=60, param1=40, param2=22,
                    minRadius=40, maxRadius=150)
    if circles is not None:
        best = sorted(np.round(circles[0]).astype(int), key=lambda c: c[2], reverse=True)[0]
        cx, cy, cr = best
    else:
        cx, cy, cr = find_best_circle(mask_closed, img_resized.shape)

    circle_mask = np.zeros(edges_canny.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (cx,cy), cr, 255, -1)
    total_area  = int(np.sum(circle_mask > 0)) or 1
    damage_area = int(np.sum((edges_canny > 0) & (circle_mask > 0)))
    damage_pct  = round((damage_area/total_area)*100, 2)

    cat_cnn, conf_cnn, proba_cnn = classify_cnn(img_bgr)

    diterima = cek_ambang(proba_cnn)

    hasil["kategori"]   = labels[cat_cnn]
    hasil["confidence"] = float(conf_cnn)
    hasil["status"]     = "tidak_yakin" if not diterima else ("rusak" if cat_cnn != "fresh" else "normal")
    hasil["damage_pct"] = float(damage_pct)
    hasil["proba_cnn"]  = [float(x) for x in proba_cnn]

    # fitur unik: confidence threshold dan Grad-CAM
    hasil["diterima"] = bool(diterima)
    hasil["ambang"]   = round(AMBANG*100, 1)
    hasil["gradcam"]  = encode_image(gradcam_cnn(img_bgr, categories.index(cat_cnn)))

    return jsonify(hasil)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)   # reloader dimatikan supaya model TensorFlow tidak dimuat dua kali