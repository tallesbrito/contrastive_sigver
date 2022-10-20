""" This example shows how to extract features for a new signature,
    using the CNN trained with Triplet loss on the GPDS dataset. 
"""
import torch

from skimage.io import imread
from skimage import img_as_ubyte

from sigver.preprocessing.normalize import preprocess_signature
from sigver.featurelearning.models import SigNet

canvas_size = (952, 1360)  # Maximum signature size

# If GPU is available, use it:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))

# Load and pre-process the signature
original = img_as_ubyte(imread('data/some_signature.png', as_gray=True))
processed = preprocess_signature(original, canvas_size)

# Note: the image needs to be a pytorch tensor with pixels in the range [0, 1]
input = torch.from_numpy(processed).view(1, 1, 150, 220)
input = input.float().div(255).to(device)

# Load the model
state_dict, _, _ = torch.load('models/triplet_01/model.pth')
base_model = SigNet().to(device).eval()
base_model.load_state_dict(state_dict)

# Extract features
with torch.no_grad(): # We don't need gradients. Inform torch so it doesn't compute them
    features = base_model(input)

features = features.cpu()[0]
print('Feature vector size:', len(features))
print('Feature vector:', features)