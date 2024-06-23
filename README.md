# phicloudmask

paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10505181
repo: https://github.com/aliFrancis/SEnSeIv2
hf: https://huggingface.co/aliFrancis/SEnSeIv2


The code of Francis was quite hard to understand, so I decided to rewrite it in a more pythonic way. I think we can use this code to start to build our own model.


Load the data and weights:

```python
import torch
import pickle
import pathlib
import numpy as np
from phicloudmask import CloudMask
from phicloudmask.constant import SENTINEL2_DESCRIPTORS

# Define the semantic categories
# land(0), water(1), snow(2), thin_cloud(3), thick_cloud(4), shadow(5), no_data(6)
cloudsen12_style = {
    0: 0, 1: 0, 2: 0, 3: 0, 6: 0,
    4: 1, 3: 2, 5: 3
}
map_values = lambda x: mapping.get(x, x)

# Load the weights
weights_folder = pathlib.Path("weights/")
embedding_weights = torch.load(weights_folder / "spectral_embedding.pt")
cloudmask_weights = torch.load(weights_folder / "cloudmask_weights.pt")

# Load a sample image
with open(weights_folder / "demo.pkl", "rb") as f:
    dict_demo = pickle.load(f)

# Load a sample image
with open(weights_folder / "demo.pkl", "rb") as f:
    dict_demo = pickle.load(f)

    # S2 L1C image
    array_demo = dict_demo["s2"][:, 0:512, 0:512]
    
    # (1, 1068, 1068)  Original senseiv2 results
    mask_demo = dict_demo["cloud_mask"][:, 0:512, 0:512]
```

Predict for all Sentinel-2 bands

```python
# Load the cloud mask model
model = CloudMask(descriptor=SENTINEL2_DESCRIPTORS, device="cuda")
model.embedding_model.load_state_dict(embedding_weights)
model.cloud_model.load_state_dict(cloudmask_weights)
model.eval()

# Get the cloud mask
with torch.no_grad():
    image = torch.from_numpy(array_demo[None]).float().to("cuda")
    cloud_probs_all = model(image)
    cloud_mask_all = cloud_probs_all.argmax(dim=0).cpu().numpy()
    cloud_4class_all = np.vectorize(cloudsen12_style)(cloud_mask_all)
```

Predict for RGB bands


```python
# Define the RGB bands descriptors
RGB_DESCRIPTORS = [
    {"band_type": "TOA Reflectance", "min_wavelength": 645.5, "max_wavelength": 684.5},
    {"band_type": "TOA Reflectance", "min_wavelength": 537.5, "max_wavelength": 582.5},
    {"band_type": "TOA Reflectance", "min_wavelength": 446.0, "max_wavelength": 542.0},
]

# Prepare the model
model = CloudMask(descriptor=RGB_DESCRIPTORS, device="cuda")
model.embedding_model.load_state_dict(embedding_weights)
model.cloud_model.load_state_dict(cloudmask_weights)
model.eval()

# Load a sample image
with torch.no_grad():
    image = torch.from_numpy(array_demo[[3, 2, 1]][None]).float().to("cuda")
    cloud_probs_rgb = model(image)
    cloud_mask_rgb = cloud_probs_rgb.argmax(dim=0).cpu().numpy()
    cloud_4class_rgb = np.vectorize(cloudsen12_style)(cloud_mask_rgb)
```

Check the results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
axes[0].imshow(image[0].permute(1, 2, 0).cpu().numpy()*5)
axes[0].set_title("RGB Image")
axes[1].imshow(mask_demo[0])
axes[1].set_title("Original Mask - SEnSeIv2")
axes[2].imshow(cloud_4class_all)
axes[2].set_title("All Bands Mask - phiCloudMask")
axes[3].imshow(cloud_4class_rgb)
axes[3].set_title("RGB Bands Mask - phiCloudMask")
plt.show()
```

