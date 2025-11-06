###Infrastructure Crack Detection Project###


An AI-powered system designed to detect cracks in infrastructure images such as concrete structures and metal components. Utilizing transfer learning on a pre-trained ResNet101 model, the project provides a streamlined Streamlit interface for easy image upload and rapid crack detection—eliminating the need for a separate backend server.

##Project Overview
Objective: Accurately detect cracks in images using a fine-tuned ResNet101 deep learning model.
Frontend: Streamlit web application for uploading images and displaying prediction results.
Tech Stack: Python, TensorFlow, Streamlit, Pillow, NumPy.

##Project Structure

crack_detection_project/
├── app/
│   └── streamlit_app.py               # Streamlit frontend application
├── models/
│   └── resnet101_crack_detector.h5   # Pre-trained model weights
├── src/
│   ├── data_preparation.py            # Data loading and augmentation scripts
│   ├── model.py                      # Model architecture script
│   └── train.py                      # Model training script
├── requirements.txt                  # Python dependencies
├── README.md                        # Project documentation
└── .gitignore                      # Git ignore rules

##Setup Instructions
1. Clone the repository
bash
git clone https://github.com/<your-username>/<repo-name>.git
cd crack_detection_project

2. Create and activate a Python virtual environment
bash
python -m venv env
For Windows:

bash
.\env\Scripts\activate
For macOS/Linux:

bash
source env/bin/activate

3. Install project dependencies
bash
pip install -r requirements.txt
Running the Application Locally
Launch the Streamlit app to interact with the crack detection system:

bash
streamlit run app/streamlit_app.py
The app will be accessible at http://localhost:8501.
Upload images directly through the interface and receive crack detection predictions instantly.

##How It Works
Images uploaded via the Streamlit interface are preprocessed (resized and normalized).
The locally loaded ResNet101 model performs inference within the app.
Predictions are shown clearly, with orange text indicating “Crack Detected” and green text for “No Crack Detected.”

##Data Preparation and Training
Use src/data_preparation.py to load and augment your crack/no-crack data.
Build the model architecture with src/model.py using transfer learning on ResNet101.
Train and fine-tune your model using src/train.py.
The trained model is saved as models/resnet101_crack_detector.h5.

##Notes
Ensure image preprocessing during inference matches that used in training.
The Streamlit app inherently supports model inference without external API calls or backend services.
For production, deploy the Streamlit app to platforms like Streamlit Cloud to enable easy access.

##Dependencies
Python 3.x
TensorFlow
Streamlit
Pillow
NumPy

All listed in requirements.txt.

##Acknowledgements
Dataset sourced from Mendeley Data - Crack Detection, contributed by Çağlar Fırat Özgenel.

Özgenel, Çağlar Fırat (2019), “Concrete Crack Images for Classification”, Mendeley Data, v2
http://dx.doi.org/10.17632/5y9wdsg2zt.2
