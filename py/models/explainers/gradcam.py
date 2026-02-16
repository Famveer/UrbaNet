import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Callable
from pathlib import Path
from PIL import Image

class GradCAM:
    """GradCAM implementation for VGG16 model."""
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: VGG16 model instance
            target_layer: Target layer for GradCAM (default: last conv layer)
        """
        self.model = model
        self.model.eval()
        
        # Use last convolutional layer if not specified
        if target_layer is None:
            self.target_layer = model.feature_maps[-1]  # Last layer in features
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, transforms_list, target_class=None):
        """
        Generate GradCAM heatmap.
        
        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
        
        Returns:
            cam: GradCAM heatmap (H, W)
            predicted_class: Predicted class index
        """
        # Generate tensor
        input_tensor = self.read_and_apply_transformations(input_image, transforms_list)
        
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class
    
    def read_and_apply_transformations(self, 
                        img: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
                        transforms_list: Optional[Callable] = None,
                        ) -> List:
        """
        Load images and apply transformations.
        
        Args:
            images: Single image or list of images (paths or PIL Images)
            transforms_list: Optional custom transforms to apply before CLIP preprocessing
            
        Returns:
            List of PIL Images
        """
        if isinstance(img, (str, Path)):
            pil_images = Image.open(img).convert('RGB')
        elif isinstance(img, Image.Image):
            pil_images = img.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        if transforms_list is not None:
            pil_images = transforms_list(img).unsqueeze(0)
        
        return pil_images
    
    def visualize(self, image, cam, alpha=0.4):
        """
        Visualize GradCAM overlay on original image.
        """
        # Transform np.array
        input_image = np.array(image)
        
        # Resize CAM to match original image
        h, w = input_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        visualization = heatmap * alpha + np.array(input_image) * (1 - alpha)
        visualization = np.uint8(visualization)
            
        return visualization, cam_resized


