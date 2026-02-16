import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

class FisherVectorEncoder:
    def __init__(self, n_components=64):
        self.n_components = n_components
        self.gmm = None
        
    def fit(self, descriptors_list):
        """
        Fit GMM on training descriptors
        descriptors_list: list of descriptor arrays from multiple images
        """
        # Filter out None values
        descriptors_list = [d for d in descriptors_list if d is not None and len(d) > 0]
        
        if not descriptors_list:
            raise ValueError("No valid descriptors provided for training")
        
        # Stack all descriptors
        all_descriptors = np.vstack(descriptors_list)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            random_state=42
        )
        self.gmm.fit(all_descriptors)
        
    def encode(self, descriptors):
        """
        Compute Fisher Vector for a set of descriptors
        """
        if descriptors is None or len(descriptors) == 0:
            raise ValueError("Empty descriptors provided")
            
        if self.gmm is None:
            raise ValueError("GMM not fitted. Call fit() first.")
        
        # Get GMM parameters
        means = self.gmm.means_
        covariances = self.gmm.covariances_
        weights = self.gmm.weights_
        
        # Compute responsibilities
        posteriors = self.gmm.predict_proba(descriptors)
        
        # Initialize Fisher Vector
        d = descriptors.shape[1]
        fv = np.zeros(2 * self.n_components * d)
        
        # Compute first and second order differences
        for k in range(self.n_components):
            # First order (CORRECTED: normalize by standard deviation)
            diff = (descriptors - means[k]) / np.sqrt(covariances[k])
            fv_mean = np.sum(posteriors[:, k:k+1] * diff, axis=0)
            fv_mean /= (np.sqrt(weights[k]) * len(descriptors))
            
            # Second order
            diff_unnormalized = descriptors - means[k]
            diff_sq = (diff_unnormalized ** 2) / covariances[k] - 1
            fv_var = np.sum(posteriors[:, k:k+1] * diff_sq, axis=0)
            fv_var /= (np.sqrt(2 * weights[k]) * len(descriptors))
            
            # Stack
            fv[k*d:(k+1)*d] = fv_mean
            fv[(self.n_components + k)*d:(self.n_components + k + 1)*d] = fv_var
        
        # Power normalization (signed square root)
        fv = np.sign(fv) * np.sqrt(np.abs(fv))
        
        # L2 normalization
        norm = np.linalg.norm(fv)
        if norm > 0:
            fv = fv / norm
        
        return fv

# Example usage
def extract_sift_descriptors(image_path):
    """Extract SIFT descriptors from image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return descriptors
    
