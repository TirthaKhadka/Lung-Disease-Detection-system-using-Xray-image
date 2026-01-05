import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import cv2
import os
import base64
import traceback

# ===================== CONFIG =====================
MODEL_PATH = "/Volumes/External/Pneumonia_Detection_system/backend/BackendForML/saved_res_model_dir"
OUTPUT_DIR = "static/"
API_KEY = "SPRING_TO_FASTAPI_SECRET"   # Must match Spring Boot
SPRING_BOOT_ORIGIN = "http://192.168.0.3:8081"
# ==================================================

# --- Load SavedModel ---
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# --- FastAPI setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[SPRING_BOOT_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Image preprocessing ---
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.asarray(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# --- Bounding box overlay ---
def overlay_bounding_box(original_image, mask, output_path,
                          max_box_ratio=0.5, min_box_area=500):
    mask_bin = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    orig_width, orig_height = original_image.size
    mask_height, mask_width = mask.shape[:2]
    scale_x = orig_width / mask_width
    scale_y = orig_height / mask_height

    localized_image = np.array(original_image)
    if len(localized_image.shape) == 2 or localized_image.shape[2] != 3:
        localized_image = cv2.cvtColor(localized_image, cv2.COLOR_GRAY2BGR)

    pneumonia_detected = False

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_area = w * h
        image_area = mask_width * mask_height
        box_ratio = box_area / image_area

        if box_ratio < max_box_ratio and box_area > min_box_area:
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            cv2.rectangle(
                localized_image,
                (x, y),
                (x + w, y + h),
                (168, 85, 230),
                8
            )
            pneumonia_detected = True

    if pneumonia_detected:
        cv2.imwrite(output_path, localized_image)

    return pneumonia_detected, output_path if pneumonia_detected else None

# --- Heatmap overlay ---
def overlay_heatmap(original_image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(
        heatmap,
        (original_image.size[0], original_image.size[1])
    )
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    image_array = np.array(original_image)
    if len(image_array.shape) == 2 or image_array.shape[2] != 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(image_array, 1 - alpha, heatmap_color, alpha, 0)

# --- Routes ---
@app.get("/")
def root():
    return {"message": "ML Pneumonia Detection Service is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        img_bytes = await file.read()
        image = Image.open(BytesIO(img_bytes))
        original_image = image.copy()  # Keep the original image

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Perform inference
        predictions = infer(tf.convert_to_tensor(preprocessed_image, dtype=tf.float32))
        prediction_mask = predictions['output_0'].numpy()[0, :, :, 0]  # Extract mask

        # Calculate sigmoid probability
        probabilities = tf.nn.sigmoid(predictions['output_0']).numpy()
        max_probability = np.max(probabilities)  # Extract the maximum probability

        # Prepare output paths
        output_filename = os.path.join(
            OUTPUT_DIR, f"{os.path.splitext(file.filename)[0]}_localized.png"
        )

        # Overlay bounding box (if any) and check for pneumonia
        pneumonia_detected, localized_image_path = overlay_bounding_box(
            original_image, prediction_mask, output_filename
        )
 
        if pneumonia_detected:
            with open(localized_image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            result = {
                "diagnosis": "Pneumonia",
                "probability": round(float(max_probability), 3),
                "localized_image": img_base64,
                "lung_opacity": "Present"
            }
        else:
            result = {
                "diagnosis": "No Pneumonia",
                "probability": round(float(max_probability), 3),
                "localized_image": "",
                "lung_opacity": "Absent"
            }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})