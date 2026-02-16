import torch
import numpy as np

class BaseModel:
    def __init__(self, model_name, device=None):
        self.device = self.evaluate_device(device)

        self.model = None
        self.model_name = model_name
        self.initialize_model(model_name)
    
    def get_processor(self):
        return self.model.get_processor() if self.model else None

    def get_model_arch(self):
        return self.model.get_model_arch() if self.model else None

    def get_model_name(self):
        return self.model.get_model_name() if self.model else None

    def get_model(self):
        return self.model
    
    def evaluate_device(self, device):
        if isinstance(device, torch.device):
            return device
        elif isinstance(device, str) and device in ["cpu", "cuda"]:
            return torch.device(device)
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def to_device(self, device=None):
        if self.model:
            self.model.to(self.device if device is None else self.evaluate_device(device))
            self.model.device = device

    def eval(self):
        if self.model:
            self.model.eval()

    def train(self):
        if self.model:
            self.model.train()
    
    def freeze_parameters(self, parameter_list):
        # Freeze all layers
        for param in self.model.model.parameters():
            param.requires_grad = False
            
    def print_trainable_parameters(self, log=True, log_params=False):
        if not self.model:
            print("No model initialized.")
            return

        total_params = 0
        trainable_params = 0
        
        if log:
            print("Model arch used:", self.get_model_arch(), "\n")
            print("Trainable parameters in the model:")

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if log_params:
                    print(f"{name}: {param.shape}")
                trainable_params += param.numel()
            total_params += param.numel()

        percent_params = 100 * trainable_params / total_params

        if log:
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Trainable parameters percentage: {percent_params:.4f}%")
        

    def initialize_model(self, model_name):
        """To be implemented in subclasses."""
        raise NotImplementedError("Subclasses must implement initialize_model().")
