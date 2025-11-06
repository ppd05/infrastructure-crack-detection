import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Dropout
from tensorflow.keras.models import Model


def build_model(img_size=(224,224),num_classes=1):
    base_model=ResNet101(weights='imagenet',include_top=False,input_shape=img_size+(3,))
    base_model.trainable=False
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dropout(0.5)(x)
    output=Dense(num_classes,activation='sigmoid')(x)
    model=Model(inputs=base_model.input,outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()