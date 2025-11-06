# Infrastructure Crack Detection Project

This project implements an AI-powered system to detect cracks in infrastructure images such as concrete structures or metal components. It combines deep learning with transfer learning on pre-trained models, served through a FastAPI backend, and an interactive Streamlit frontend for easy user interactions.

---

## Project Overview

- **Objective:** Detect cracks in images using a pre-trained ResNet101 model fine-tuned with your dataset.
- **Backend:** FastAPI REST API for receiving images and returning crack detection results.
- **Frontend:** Streamlit web app for uploading images and displaying crack presence with confidence.
- **Tech Stack:** Python, TensorFlow, FastAPI, Streamlit, Pillow, NumPy.

---

## Folder Structure (Recommended)

crack_detection_project/
├── app/
│ ├── api.py # FastAPI backend code
│ ├── streamlit_app.py # Streamlit frontend code
├── models/
│ └── resnet101_crack_detector.h5 # Trained model weights
├── src/
│ ├── data_preparation.py # Data loading and augmentation
│ ├── model.py # Model build script
│ └── train.py # Training script
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)
└── .gitignore # Git ignore rules

text

---

## Setup Instructions

### 1. Clone the repository

git clone https://github.com/<your-username>/<repo-name>.git
cd crack_detection_project

text

### 2. Create and activate a Python virtual environment

python -m venv env

Windows
.\env\Scripts\activate

macOS/Linux
source env/bin/activate

text

### 3. Install dependencies

pip install -r requirements.txt

text

---

## Running the Project Locally

### Run FastAPI backend

uvicorn app.api:app --reload

text

- API will be available at `http://127.0.0.1:8000`
- Test API docs at `http://127.0.0.1:8000/docs`

### Run Streamlit frontend

streamlit run app/streamlit_app.py

text

- Frontend UI available at `http://localhost:8501`
- Upload images and get crack detection predictions interactively.

---

## How It Works

- Images uploaded via the frontend are sent to the FastAPI backend.
- Backend preprocesses images (resize, normalize) and runs inference on the ResNet101 model.
- The prediction probability is returned.
- Frontend shows result with orange color for "crack" and green for "no crack".

---

## Data Preparation and Model Training

- Use `src/data_preparation.py` to load and augment crack/no-crack images.
- Use `src/model.py` to build a transfer learning model on ResNet101.
- Train the model using `src/train.py`; the model is saved to `models/resnet101_crack_detector.h5`.

---

## Notes

- The model and preprocessing must be consistent between training and inference.
- Ensure `python-multipart` is installed for file upload handling in FastAPI.
- For deployment, consider deploying FastAPI and Streamlit apps on cloud platforms (Heroku, Streamlit Cloud, AWS).
- Docker support optional but recommended for easy reproducibility.

---

## Requirements

- Python 3.x
- TensorFlow
- FastAPI
- Uvicorn
- Streamlit
- Pillow
- NumPy
- Requests
- python-multipart

---

#Data Credits:
#Acknowledgements
This dataset is taken from the website Mendeley Data - Crack Detection, contributed by Çağlar Fırat Özgenel.

Özgenel, Çağlar Fırat (2019), “Concrete Crack Images for Classification”, Mendeley Data, v2
http://dx.doi.org/10.17632/5y9wdsg2zt.2


This README provides full context on project setup, usage, and development for your crack detection application using FastAPI and Streamlit.