from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model during startup
MODEL_PATH = r"C:\Users\prate\Desktop\crack_detection_project\models\resnet101_crack_detector.h5"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (224, 224)

app = FastAPI(title='Crack Detection API', description='ResNet101 based API for crack detection in images')

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_arr = preprocess_image(image_bytes)
    try:
        prediction = model.predict(input_arr)
        pred_prob = float(prediction[0][0])
        pred_label = 'crack' if pred_prob > 0.5 else 'no_crack'
        return JSONResponse({
            "predicted_label": pred_label,
            "probability": pred_prob
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)