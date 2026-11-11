import torch
import os
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')
print('Working directory:',os.getcwd())
print('Training archive present:', os.path.exists('/content/stage2_training.zip'))
