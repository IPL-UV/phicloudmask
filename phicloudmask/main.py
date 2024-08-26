import torch
from typing import List, Union
from phicloudmask.encoding import SEnSeIv2Encoding
from phicloudmask.embedding import SpectralEmbedding
from phicloudmask.segmodel import CustomSegformer


class CloudMask(torch.nn.Module):
    def __init__(
        self,
        descriptor: List[dict],
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()

        # Prepare the encoding model for the descriptor
        self.band_descriptor = SEnSeIv2Encoding(
            emb_size=74,
            N_embeddings=32,
            device="cpu"
        )([descriptor])
        self.device = device

        # Define the embedding model
        self.embedding_model = SpectralEmbedding(
            in_features=74,
            device=device
        )

        # Define the cloud mask model
        self.cloud_model = CustomSegformer(
            pretrained_model_name="nvidia/mit-b2",
            num_labels=7,
            num_channels=32
        ).segmenter

        # to device
        self.embedding_model.to(device)
        self.cloud_model.to(device)

    def forward(self, img: torch.Tensor):
        # Embed the input image translate to the image device
        band_descriptor = self.band_descriptor.to(self.device).clone()
        embedding = self.embedding_model(img, band_descriptor)
        
        # Forward pass through the cloud mask model
        cloud_probs = self.cloud_model(embedding).logits
        
        # Interpolate the mask to the original image size
        resampled = torch.nn.functional.interpolate(cloud_probs, img.shape[2:], mode='bilinear', antialias=True)
        outputs = torch.nn.functional.softmax(resampled, dim=1).squeeze()
        
        return outputs