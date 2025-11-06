import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(data_dir, img_size=(224, 224), batch_size=32):
    """
    Prepare training and validation data generators with data augmentation for training.

    Args:
        data_dir (str): Path to the base data directory containing 'train' and 'val' folders.
        img_size (tuple): Target image size, e.g., (224, 224).
        batch_size (int): Number of images per batch.

    Returns:
        train_generator, val_generator: TensorFlow data generators.
    """

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,           # Rotate images randomly from -20 to +20 degrees
        width_shift_range=0.2,       # Horizontal shift
        height_shift_range=0.2,      # Vertical shift
        shear_range=0.2,             # Shear angle in counter-clockwise direction
        zoom_range=0.2,              # Zoom in/out
        horizontal_flip=True,        # Flip images horizontally
        fill_mode='nearest',         # Fill mode for points outside boundaries
        brightness_range=[0.8, 1.2]  # Random brightness augmentation
    )

    # Validation data should NOT have augmentation, only rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator

if __name__ == "__main__":
    data_dir = r'C:\Users\prate\Desktop\crack_detection_project\data'  # Update this if needed
    train_gen, val_gen = prepare_data(data_dir)
    print("Training batches:", train_gen.samples // train_gen.batch_size)
    print("Validation batches:", val_gen.samples // val_gen.batch_size)