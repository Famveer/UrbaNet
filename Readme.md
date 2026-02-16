# UrbaNet

Exploring the PlacePulse 2.0 dataset and analyzing the urban safety perception through visual features. [Paper](https://fmorenovr.github.io/documents/papers/conferences/2021_WIIAT.pdf).

# Requirements

- **Python**>=3.12

# Installation
```
  pip install -r requirements.txt
```

# Data

Obtain the Place Pulse 2.0 dataset [here](https://drive.google.com/drive/folders/1V1EjMaz-qqSLzMS4f8N-XkUhAUIqLKa7?usp=sharing).

### Data Preparation

* Download images and `pp2_raw_data.zip`.  
* Create a `.env` file, and add the path of the data downloaded and models.  
  ```
    DATA_PATH=/path_to/datasets/
    MODEL_PATH=/path_to/models/
  ```
* First, run the notebook `notebooks/Data/Organize_Information.ipynb`.  
  Second, run the notebook `notebooks/Data/Process_XX_scores.ipynb`.  
  Then, run the notebook `notebooks/Data/Statistics.ipynb`.  
  Next, run the notebook `notebooks/Data/Feature_Extraction.ipynb`.  

* Train models running `notebooks/Models/`.  
  Then, run explanations at `notebooks/Explanations/`.  

# Citation

```
@inproceedings{moreno2021quantifying,
    author = {Moreno-Vera, Felipe and Lavi, Bahram and Poco, Jorge},
    title = {Quantifying Urban Safety Perception on Street View Images},
    year = {2021},
    isbn = {9781450391153},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3486622.3493975},
    doi = {10.1145/3486622.3493975},
    booktitle = {IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology},
    pages = {611–616},
    numpages = {6},
    keywords = {Image Pre-processing, Urban Perception, Place Pulse, Perception Computing, Safety Perception, Perception Learning, Computer Vision, Street View, Deep Learning, Feature Extraction, Street-level imagery, Urban Computing, Cityscape},
    location = {Melbourne, VIC, Australia},
    series = {WI-IAT '21}
}
```

# Contact us  
For any issue please kindly email to `felipe [dot] moreno [at] fgv [dot] br`
