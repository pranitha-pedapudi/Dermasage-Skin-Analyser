import streamlit as st  
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def set_img_as_background(img_path):
    """
    Display an image as the background for the Streamlit app.
    """
    st.image(img_path)

@st.cache(allow_output_mutation=True)
def load_trained_model():
    """
    Load a trained Keras model with EfficientNetB3 base and custom top layers.
    """
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(5, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Unfreeze the last 100 layers for better fine-tuning
    for layer in base_model.layers[-100:]:
        layer.trainable = True
    
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model 

def predict_model(image, model):
    """
    Make predictions using a trained model.
    """
    return model.predict(image, verbose=0)

def get_data_generator():
    """
    Create an ImageDataGenerator with augmentation for training and validation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% of the data for validation
    )
    
    return train_datagen
