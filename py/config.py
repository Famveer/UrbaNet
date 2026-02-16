import os
from pydantic_settings import BaseSettings
import ast
from dotenv import load_dotenv
load_dotenv()

class Config(BaseSettings):

    DATA_PATH: str = os.getenv('DATA_PATH')
    MODEL_PATH: str = os.getenv('MODEL_PATH')

    ML_TASK: str = os.getenv('ML_TASK', "classifications")
    MODEL_FEATURE_NAME: str = os.getenv('MODEL_TASK_NAME', "vgg16_mlp")
    BATCH_SIZE: int = int(os.getenv('BATH_SIZE', "256"))
    NUM_EPOCHS: int = int(os.getenv('NUM_EPOCHS', "50"))
    
    PERCEPTION_METRIC: str = os.getenv('PERCEPTION_METRIC', "safety")
    CITY_STUDIED: str = os.getenv('CITY_STUDIED', "Rio De Janeiro")
    DELTA: float  = float(os.getenv('DELTA', "0.42"))

    RANDOM_STATE: int = int(os.getenv('RANDOM_STATE', "42"))
