import torch
import torch.nn as nn
from torchvision import models


class VGG16(nn.Module):
    """Simplified VGG16 with flexible head configurations."""
    
    def __init__(self, num_classes, use_gap=False, use_mlp=False):
        super(VGG16, self).__init__()
        self.use_gap = use_gap
        self.use_mlp = use_mlp
        
        if use_gap and use_mlp:
            raise ValueError(f"Invalid combination: use_gap: {use_gap} and use_mlp: {use_mlp}, cannot both be True.")
        
        # Load pretrained VGG16 features
        vgg16_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_maps = vgg16_pretrained.features
        self.model_arch = "vgg16"
        
        # Freeze feature extractor
        for param in self.feature_maps.parameters():
            param.requires_grad = False
        
        # Optional: Unfreeze last conv block for fine-tuning
        for param in self.feature_maps[24:].parameters():
           param.requires_grad = True
        
        # Build classifier head
        if use_gap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = self._build_mlp_head(512, num_classes)
            self.model_name = "VGG16_GAP"
            
        elif use_mlp:
            self.gap = None
            self.classifier = self._build_mlp_head(512 * 7 * 7, num_classes)
            self.model_name = "VGG16_MLP"
            
        else:
            self.gap = None
            self.classifier = vgg16_pretrained.classifier
            self.classifier[6] = nn.Linear(4096, num_classes)
            nn.init.xavier_uniform_(self.classifier[6].weight)
            self.model_name = "VGG16"
    
    def _build_mlp_head(self, input_dim, num_classes):
        """Build a simple MLP classification head."""
        layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        # Initialize weights
        nn.init.xavier_uniform_(layers[1].weight)
        nn.init.xavier_uniform_(layers[4].weight)
        return layers
        
    def get_model_arch(self):
        return self.model_arch

    def get_model_name(self):
        return self.model_name
    
    def forward(self, x):
        x = self.feature_maps(x)
        if self.use_gap:
            x = self.gap(x)
        x = self.classifier(x)
        return x

