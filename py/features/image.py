import cv2
import numpy as np
from PIL import Image
from sklearn.mixture import GaussianMixture

from .gist import extract_gist_features
from .sift import extract_sift_descriptors
from .fisher import FisherVectorEncoder

class ImageFeatureExtractor:
    def __init__(self, n_gaussians=64):
        self.fv_encoder = FisherVectorEncoder(n_gaussians)
        
    def extract_all_features(self, image_path):
        """Extract both GIST and Fisher Vector features"""
        features = {}
        
        # Extract GIST
        features['gist'] = extract_gist_features(image_path)
        
        # Extract Fisher Vector
        descriptors = extract_sift_descriptors(image_path)
        if descriptors is not None and self.fv_encoder.gmm is not None:
            features['fisher_vector'] = self.fv_encoder.encode(descriptors)
        
        return features
