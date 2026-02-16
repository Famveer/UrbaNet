from torchvision import transforms

class ImageTransforms():
    def __init__(self):
        pass
    
    def get(self, model_name=None, type_transform=None):
    
        if model_name is not None and type_transform is None:
            transforms_list = {
                'train': self.train_transforms(model_name),
                'val': self.test_transforms(model_name)
            }
            return transforms_list
    
        if type_transform is not None:
            if type_transform=="train":
                return self.train_transforms(model_name)
            else:
                return self.test_transforms(model_name)
            
        return None
    
    def test_transforms(self, model_name):
        
        if "ssgan" in model_name.lower():
            IMG_SIZE=int( model_name.lower().replace("ssgan", "") )
            
            transforms_list = transforms.Compose([
                                  transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])
        
        elif "vit_b16" in model_name.lower():
            transforms_list = transforms.Compose([
                                  transforms.Resize((384, 384)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                              ])
            
        elif "vit_h14" in model_name.lower():
            transforms_list = transforms.Compose([
                                  transforms.Resize((518, 518)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                              ])
            
        else:
            transforms_list = transforms.Compose([
                              transforms.Resize((224, 224)), 
                              transforms.ToTensor(), 
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ])
        
        return transforms_list
    
    def train_transforms(self, model_name):
        
        if "lip" in model_name.lower():
            transforms_list = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ])
        
        elif "vit_b16" in model_name.lower():
            transforms_list = transforms.Compose([
                                  transforms.Resize((384, 384)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                              ])
            
        elif "vit_h14" in model_name.lower():
            transforms_list = transforms.Compose([
                                  transforms.Resize((518, 518)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                              ])
        
        elif "ssgan" in model_name.lower():
            IMG_SIZE=int( model_name.lower().replace("ssgan", "") )
            
            transforms_list = transforms.Compose([
                                  transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])

        else:
            # transforms.Grayscale(num_output_channels=3),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust colors
            transforms_list = transforms.Compose([
                              transforms.Resize((224, 224)), 
                              transforms.ToTensor(), 
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ])
        
        return transforms_list

