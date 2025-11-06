import os
from model import build_model
from data_preparation import prepare_data

def train(data_dir, batch_size=32, img_size=(224, 224), epochs=2):
    # Prepare data generators
    train_gen, val_gen = prepare_data(data_dir, img_size=img_size, batch_size=batch_size)

    # Build model
    model = build_model(img_size=img_size)

    # Train the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        verbose=2
    )

    # Save trained model weights
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "resnet101_crack_detector.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    data_dir = r'C:\Users\prate\Desktop\crack_detection_project\data'  # update if needed
    train(data_dir)