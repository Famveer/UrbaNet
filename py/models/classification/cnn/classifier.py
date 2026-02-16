from py.models.basemodel import BaseModel

from .vgg import VGG16

class ConvolutionClassifier(BaseModel):
    def __init__(self, model_name, device=None, num_classes=None):
        self.num_classes = num_classes
        if num_classes is None:
            raise ValueError(f"Invalid numclasses: {num_classes}, must be Int.")
            
        super().__init__(model_name, device)
    
    def model_zoo(self):
        model_zoo = [#ImageNet
                     "VGG16_gap", "VGG16_mlp", "VGG16",
                     ]
    
        print( "Model zoo:", model_zoo, "\n" )
        
    def initialize_model(self, model_name):
        
        if "vgg16" in model_name.lower(): # vgg16_mlp vgg16_gap vgg16
            set_gap = True if "gap" in model_name.lower() else False
            set_mlp = True if "mlp" in model_name.lower() else False
            self.model = VGG16(num_classes=self.num_classes, 
                               use_mlp=set_mlp,  
                               use_gap=set_gap,
                               )
        
        else:
            self.model = None
            
        self.model_name = self.get_model_name()
    
