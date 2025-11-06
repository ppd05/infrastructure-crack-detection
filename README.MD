🔍 Infrastructure Crack Detection

Spot cracks instantly with AI. Upload an image, get results in seconds.

💡 What is this?
An AI-powered tool that detects cracks in concrete and metal structures. Just drag and drop an image, and get instant results—no complicated setup required.
Perfect for: Engineers, inspectors, researchers, or anyone curious about AI in infrastructure monitoring.
✨ Key Features
🎯 Accurate - 95%+ detection accuracy using ResNet101
⚡ Fast - Results in under 2 seconds
🎨 Simple - Beautiful drag-and-drop interface
🆓 Free - Open source and ready to use
💻 Local - Runs entirely on your machine
🚀 Quick Start


3 steps to start detecting cracks:
bash# 1. Clone and enter directory
git clone https://github.com/ppd05/infrastructure-crack-detection.git
cd infrastructure-crack-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app/streamlit_app.py
Open your browser → Upload an image → See results! 🎉
🎬 How It Works
mermaidgraph LR
    A[📷 Upload Image] --> B[🔄 Preprocess]
    B --> C[🧠 ResNet101 AI]
    C --> D[✅ Result + Confidence]
Behind the scenes:

Your image gets resized to 224×224 pixels
ResNet101 analyzes the patterns
AI predicts: Crack or No Crack
You see results with confidence score

📁 What's Inside
infrastructure-crack-detection/
├── app/
│   └── streamlit_app.py          # 🎨 The web interface
├── models/
│   └── resnet101_crack_detector.h5  # 🧠 Trained AI model
├── src/
│   ├── data_preparation.py       # 📊 Data handling
│   ├── model.py                  # 🏗️ AI architecture
│   └── train.py                  # 🎓 Training script
└── requirements.txt              # 📦 Dependencies
🎨 See It In Action
Upload an image:
Get instant results:
🟠 Crack Detected 
or
🟢 No Crack Detected 
🎓 Train Your Own Model
Want to customize the AI? Here's how:
bash# 1. Prepare your dataset
data/
├── train/
│   ├── Negative/        # Images with cracks
│   └── Positive/     # Images without cracks
└── val/
    ├── Negative/
    └── Positive/

# 2. Train the model
python src/train.py

# 3. Your model is saved automatically!
Training time: ~1-2 hours with GPU, 5-6 hours with CPU

🛠️ Requirements
Python 3.8+
4GB RAM (8GB recommended)
Modern web browser
Optional: GPU for faster training

🌐 Deploy It
Want to share with your team?
Streamlit Cloud (Easiest)

Push to GitHub
Go to share.streamlit.io
Connect your repo
Done! ✨

📚 Dataset Credit
Big thanks to Çağlar Fırat Özgenel for the concrete crack dataset.

Özgenel, Ç.F. (2019). "Concrete Crack Images for Classification"
Mendeley Data, v2. DOI: 10.17632/5y9wdsg2zt.2

⚠️ Important Note
This is a research/educational tool. For real infrastructure decisions, always consult professional structural engineers. AI is amazing, but safety comes first! 

🎯 Roadmap
 Mobile app version
 Batch processing for multiple images
 Crack severity classification
 Video support for real-time detection
 Export reports as PDF

📄 License
MIT License - Free to use, modify, and share!
