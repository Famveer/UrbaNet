from skimage import color, filters
from skimage.transform import resize
import numpy as np
from PIL import Image

def extract_gist_features(image_path, n_blocks=4):
    # Load and convert to grayscale
    img = np.array(Image.open(image_path))
    img_gray = color.rgb2gray(img)
    img_resized = resize(img_gray, (128, 128))
    
    # Apply Gabor filters at different scales and orientations
    features = []
    frequencies = [0.1, 0.2, 0.3, 0.4]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for freq in frequencies:
        for theta in orientations:
            filtered = filters.gabor(img_resized, frequency=freq, theta=theta)[0]
            
            # Divide into blocks and compute mean
            h, w = filtered.shape
            block_h, block_w = h // n_blocks, w // n_blocks
            
            for i in range(n_blocks):
                for j in range(n_blocks):
                    block = filtered[i*block_h:(i+1)*block_h, 
                                   j*block_w:(j+1)*block_w]
                    features.append(block.mean())
    
    return np.array(features)
