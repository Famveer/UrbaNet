import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Callable
from pathlib import Path
from PIL import Image

class ImageLIME:
    """LIME explanation for VGG16 model."""
    
    def __init__(self, model, class_names=None):
        """
        Args:
            model: VGG16 model instance
            class_names: List of class names
        """
        self.model = model
        self.model.eval()
        self.class_names = class_names
        
        # Create LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def explain(self, image, transforms_list, target_class=None, num_samples=1000):
        """
        Generate LIME explanation.
        
        Args:
            image: PIL Image or numpy array (H, W, C)
            transforms_list: torchvision transforms to apply
            target_class: Target class to explain (if None, uses predicted class)
            num_samples: Number of samples for LIME
        
        Returns:
            explanation: LIME explanation object
            explained_class: The class that was explained
        """
        # Convert to numpy array in [0, 1] range
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Normalize to [0, 1]
        if image_np.dtype == np.uint8:
            image_normalized = image_np / 255.0
        else:
            image_normalized = image_np
        
        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                # Convert to PIL for transforms
                pil_image = Image.fromarray(image_np) if isinstance(image_np, np.ndarray) else image
                input_tensor = transforms_list(pil_image).unsqueeze(0)
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        def predict_fn(images):
            """
            CRITICAL: LIME passes a BATCH of images (N, H, W, C) in [0, 1] range.
            Must process each image individually.
            """
            batch_predictions = []
            
            for img in images:
                # Convert from [0, 1] float to [0, 255] uint8 for PIL
                img_uint8 = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8)
                
                # Apply your transforms
                preprocessed = transforms_list(pil_img).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(preprocessed)
                    probs = F.softmax(output, dim=1)
                
                batch_predictions.append(probs.cpu().numpy()[0])
            
            return np.array(batch_predictions)
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image_normalized,
            predict_fn,
            labels=[target_class],
            hide_color=0,
            num_samples=num_samples
        )
        
        return explanation, target_class
    
    def visualize(self, image, explanation, target_class=None, positive_only=True, num_features=5):
        """
        Visualize LIME explanation.
        """
        
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image) / 255.0
        else:
            image_np = image if image.max() <= 1.0 else image / 255.0
        
        if target_class is None:
            # Use the first label in the explanation
            target_class = explanation.top_labels[0] if hasattr(explanation, 'top_labels') and explanation.top_labels else list(explanation.local_exp.keys())[0]
        
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=False
        )
        
        # Create visualization
        visualization = mark_boundaries(temp, mask)
        
        return visualization

