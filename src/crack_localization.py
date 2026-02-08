import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

class CrackLocalizer:
    def __init__(self, model):
        """Initialize crack localizer with trained model"""
        self.model = model
        
    def get_activation_map(self, img_array, layer_name='conv5_block3_out'):
        """Get activation map from specific layer"""
        # Create a model that outputs the activation
        layer_output = self.model.get_layer(layer_name).output
        activation_model = Model(inputs=self.model.input, outputs=layer_output)
        
        # Get activation
        activation = activation_model.predict(img_array)
        return activation
    
    def generate_heatmap(self, img_path, alpha=0.4):
        """Generate heatmap overlay on original image"""
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Get activation map
        activation = self.get_activation_map(img_array)
        
        # Average across channels
        heatmap = np.mean(activation[0], axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Load original image
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (224, 224))
        
        # Overlay heatmap on original image
        superimposed_img = cv2.addWeighted(original_img, 1-alpha, heatmap, alpha, 0)
        
        return superimposed_img, heatmap
    
    def detect_crack_bbox(self, img_path, threshold=0.6):
        """Detect bounding box around crack region"""
        # Generate heatmap
        _, heatmap = self.generate_heatmap(img_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Load original image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        # Draw bounding boxes
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return img, bboxes