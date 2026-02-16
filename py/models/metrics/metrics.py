import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

import torch
import torchvision.models.inception as inception
import torchvision.transforms as transforms

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from scipy.linalg import sqrtm
from PIL import Image

class EvaluationMetrics:
    def __init__(self, task, device='cpu'):
        self.task = task
        self.device = device
        
        if "image_generations" == self.task.lower():

            self.model = inception.inception_v3(pretrained=True, transform_input=False).to(self.device)
            self.model.eval()
            
            self.transformations = transforms.Compose([
                                      transforms.Resize((299, 299)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                  ])
            
    def _get_inception_activations(self, images):
        processed_images = torch.stack([self.transformations(Image.fromarray(img)) for img in images]).to(self.device)
        with torch.no_grad():
            activations = self.model(processed_images).cpu().numpy()
        
        return activations
    
    def calculate(self, y_true, y_pred):
        if "regression" in self.task.lower() or "reg" in self.task.lower():
            return self.regression_metrics(y_true, y_pred)
            
        elif "classification" in self.task.lower() or "class" in self.task.lower():
            return self.classification_metrics(y_true, y_pred)
            
        elif "image_generations" == self.task.lower():
            return self.image_generation_metrics(y_true, y_pred)
        
        else:
            raise("No tasks defined")
            return None

    # -------------------- FID – Fréchet Inception Distance -------------------- 
    def calculate_fid(self, real_images, generated_images):
        self.model.fc = torch.nn.Identity()
        act1 = self._get_inception_activations(real_images)
        act2 = self._get_inception_activations(generated_images)

        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2)
        covmean = sqrtm(sigma1.dot(sigma2)).real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    # -------------------- IS – Inception Score -------------------- 
    def calculate_fid(self, generated_images, splits=10):
        self.model.fc = torch.nn.Softmax(dim=1)
        preds = self._get_inception_activations(generated_images)
        
        scores = []
        N = len(preds)
        for i in range(splits):
            part = preds[i * N // splits: (i + 1) * N // splits]
            py = np.mean(part, axis=0)
            kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
            kl = np.sum(kl, axis=1)
            scores.append(np.exp(np.mean(kl)))
        return np.mean(scores), np.std(scores)

    # -------------------- PSNR - Peak Signal-to-Noise Ratio --------------------
    def calculate_psnr(self, real_images, generated_images):
        psnr_values = [
            compare_psnr(real, gen, data_range=255)
            for real, gen in zip(real_images, generated_images)
        ]
        return np.mean(psnr_values)
    
    # -------------------- SSIM - Structural Similarity Index Measure --------------------
    def calculate_ssim(self, real_images, generated_images):
        ssim_values = [
            compare_ssim(real, gen, channel_axis=-1, data_range=255)
            for real, gen in zip(real_images, generated_images)
        ]
        return np.mean(ssim_values)
    
    # Image Generation Metrics
    def image_generation_metrics(self, real_images, generated_images):
        '''
        ✅ FID → Distance between real and generated distributions.
                  Measures how close the distribution of generated images is to that of real images.
                  Lower FID = lower distance of distributions between real and generated images.
                  
        ✅ IS → Sharpness and diversity of generated images
                 Evaluates image quality and diversity of generated images
                 Higher IS = sharper & more diverse images.
                 
        ✅ PSNR → Pixel-level accuracy (error)
                   Measures reconstruction quality by comparing pixel-level differences.
                   Higher PSNR = less distortion (better reconstruction)
                 
        ✅ SSIM → Structural similarity (perception)
                   Measures perceived quality (luminance, contrast, and structure) between two images.
                   Higher SSIM = better quality (closer to original).
        '''
        assert real_images is not None and generated_images is not None, "Input images must not be None."
        assert len(real_images) > 0 and len(generated_images) > 0, "Input image lists must not be empty."
        assert len(real_images) == len(generated_images), "real_images and generated_images must be the same length."
        
        fid = self.calculate_fid(real_images, generated_images)
        IS = self.calculate_is(real_images, generated_images)
        psnr = self.calculate_psnr(real_images, generated_images)
        ssim = self.calculate_ssim(generated_images)
        
        print(f'FID: {fid:.4f} IS: {IS:.4f}')
        print(f'PSNR: {psnr:.4f} SSIM: {psnr:.4f}')
        return {
            'fid': fid,
            'is': IS,
            'psnr': psnr,
            'ssim': ssim,
        }
        

    # Classification Metrics
    def classification_metrics(self, y_true, y_pred):
        '''
        ✅ Accuracy → Good for balanced datasets.
        ✅ Precision → Important when false positives are costly.
        ✅ Recall → Important when false negatives are costly.
        ✅ F1 Score → Best for imbalanced datasets (weighted average of precision & recall).
        '''
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        
        # For AUC, you need to handle multi-class separately
        # For simplicity, we'll skip multi-class AUC here, but you can apply One-vs-Rest
        try:
            auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
        except ValueError:
            auc = float('nan')
        
        print(f'Acc: {acc:.4f} AUC: {auc:.4f}')
        print(f'F1: {f1:.4f} Precision: {precision:.4f} Recall: {recall:.4f}')
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    # Regression Metrics
    def regression_metrics(self, y_true, y_pred):
        '''
        ✅ RMSE → if large errors are more critical.
        ✅ MSE → if you want a simple, interpretable error measure.
        ✅ MAE → if you want a simple, interpretable average error measure.
        ✅ R² → if you want to understand how well the model explains variance.
        '''
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        print(f'R^2: {r2:.4f} MAE: {mae:.4f} MSE: {mse:.4f} RMSE: {rmse:.4f}')
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
        }
    

import numpy as np

def calculate_iou(mask1, mask2, class_label, threshold=0.3):
    """
    Calculates the IoU between a thresholded Grad-CAM heatmap and a specific
    element in a segmentation mask.

    Args:
        mask1 (np.ndarray): The Grad-CAM heatmap.
        mask2 (np.ndarray): The segmentation mask.
        class_label (int): The label of the element to consider in the segmentation mask.
        threshold (float): The threshold to apply to the Grad-CAM heatmap.

    Returns:
        float: The IoU value.
    """

    # 1. Threshold the Grad-CAM heatmap
    thresholded_gradcam = (mask1 > threshold).astype(np.uint8)

    # 2. Create the binary segmentation mask for the specific class
    element_mask = (mask2 == class_label).astype(np.uint8)

    # 3. Calculate the intersection
    intersection = np.logical_and(thresholded_gradcam, element_mask)
    intersection_area = np.sum(intersection)

    # 4. Calculate the union
    union = np.logical_or(thresholded_gradcam, element_mask)
    union_area = np.sum(union)

    # 5. Calculate IoU
    if union_area == 0:
        iou = 0.0  # Avoid division by zero
    else:
        iou = intersection_area / union_area

    return iou, thresholded_gradcam, element_mask, intersection, union
    
