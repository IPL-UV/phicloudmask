import torch
import einops

from typing import List, Optional, Tuple, Union

class GlobalStats(torch.nn.Module):
    """
    Calculates various percentiles of the input bands and 
    concatenates them to the descriptors. This allows SEnSeI
    to learn something about the reflectance values in the
    bands, without having to consider the entire image space
    (would be too memory intensive).
    """
    def __init__(
        self,
        in_features: int,
        percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],        
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        self.device = torch.device(device) # Convert device to torch.device
        self.percentiles = torch.tensor(percentiles, device=self.device)
        self.in_features = in_features

    def forward(
        self,
        bands: torch.Tensor,
        descriptor_enc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GlobalStats module.

        Args:
            bands: Input bands tensor
            descriptor_enc: Input descriptors tensor

        Returns:
            Updated bands tensor and updated descriptors tensor
        """

        # Calculate percentiles for each band
        batch_size, num_bands, _, _ = bands.shape
        num_percentiles = len(self.percentiles)
        stats = torch.zeros(
            (batch_size, num_bands, num_percentiles), device=self.device
        )

        # Reshape bands for percentile calculation
        reshaped_bands = bands.view(batch_size, num_bands, -1)

        # Calculate percentiles for each band
        for i, percentile in enumerate(self.percentiles):
            stats[:, :, i] = torch.quantile(reshaped_bands, percentile, dim=2)

        # Concatenate descriptors with calculated percentiles
        new_descriptor_enc = torch.cat((descriptor_enc, stats), dim=2)

        return new_descriptor_enc

    def get_output_size(self) -> int:
        """
        Returns the output size of the GlobalStats module.
        """
        return self.in_features + len(self.percentiles)


class FCLBlock(torch.nn.Module):
    """
    Fully Connected Layer Block (FCLBlock)

    This module defines a sequence of fully connected 
    layers with LayerNorm and ReLU activations.

    Args:
        in_features (int): Number of input features. Default is 79.
        blocks (List[int]): List of integers defining the number of 
            neurons in each layer. Default is [128, 128, 128].
    """
    
    def __init__(
        self,
        in_features: int = 79,
        blocks: List[int] = [128, 128, 128],
        skip_connections: bool = True
    ):
        super().__init__()
        
        # Create a list to hold the layers
        self.layers = torch.nn.ModuleList()
        
        # Append input normalization and activation layers
        self.layers.append(torch.nn.LayerNorm(in_features))
        self.layers.append(torch.nn.ReLU())
        
        # Append the first linear layer
        self.layers.append(torch.nn.Linear(in_features, blocks[0]))
        if skip_connections:
            self.layers.append(torch.nn.Identity())
        
        # Iterate over the blocks to add subsequent layers
        for i in range(len(blocks) - 1):
            self.layers.append(torch.nn.LayerNorm(blocks[i]))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Linear(blocks[i], blocks[i + 1]))
            if skip_connections:
                self.layers.append(torch.nn.Identity())
        
    def forward(self, descriptor_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the FCLBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the layers.
        """
        batch_size, num_bands, descriptors_size = descriptor_enc.shape
        
        # Merge the batch and band dimensions
        descriptor_enc_reshaped = einops.rearrange(
            tensor=descriptor_enc,
            pattern='b n d -> (b n) d'
        )
        
        # Iterate over the layers
        for index, layer in enumerate(self.layers):
            
            # Apply the layer to the reshaped descriptors
            new_descriptor_enc_reshaped = layer(descriptor_enc_reshaped)            
            
            # This is important because in the first iteration, the tensor are
            # not the same shape, so we need to skip the "skip connection"
            condition = new_descriptor_enc_reshaped.shape == descriptor_enc_reshaped.shape

            # Skip connection if the layer is an identity
            if isinstance(layer,torch.nn.Identity) and condition:
                new_descriptor_enc_reshaped = (
                    descriptor_enc_reshaped + new_descriptor_enc_reshaped
                )
            
            # Update the reshaped descriptors
            descriptor_enc_reshaped = new_descriptor_enc_reshaped
        
        # Reshape the descriptors back to the original shape
        new_descriptor_enc = einops.rearrange(
            tensor=descriptor_enc_reshaped,
            pattern='(b n) d -> b n d',
            b=batch_size,
            n=num_bands
        )

        return new_descriptor_enc
    
    def get_output_size(self):
        """
        Returns the output size of the FCLBlock.
        """
        return self.layers[-2].out_features


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int = 79,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_layers: int = 2
    ):
        super().__init__()
        
        # Define the list of Transformer Encoder layers
        self.d_model = d_model
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead, 
                    dim_feedforward=dim_feedforward, 
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        
        # Check if the input descriptor dims are correct
        if in_features != self.get_output_size():
            raise ValueError(
                'Input descriptor dims must be equal to num_heads*dims_per_head'
            )

        if in_features % nhead != 0:
            raise ValueError(
                'Input descriptor dims must be divisible by nhead'
            )

    def forward(self, descriptor_enc):

        # Run through layers
        for index, layer in enumerate(self.layers):
            descriptor_enc = layer(descriptor_enc)

        return descriptor_enc

    def get_output_size(self):
        """
        Returns the output size of the AttentionBlock.
        """
        return self.d_model


class BandEmbedding(torch.nn.Module):
    """
    A way of intelligently embedding the band values into the 
    output latent space of SEnSeI v2

    This is done by learning a frequency, phase offset, and gain
    as functions of the descriptor vectors, and then using these 
    to embed the band values with scaled sinusoidal functions. This
    allows the model to (hypothetically) select where and how to 
    encode each band in the latent space.
    """
    
    def __init__(
        self, 
        in_features: int,
        embedding_dims: int,
        head_layer_sizes: List[int],
        skips_heads: bool,
        normalize: bool
    ):
        super().__init__()
        
        self.embedding_dims = embedding_dims
        self.head_layer_sizes = head_layer_sizes
        self.skips_heads = skips_heads
        self.in_features = in_features
        self.normalize = normalize
        
        # Simple FCL feedforward networks to learn the embedding parameters                
        self.frequency_head = FCLBlock(
            in_features = in_features,
            blocks = head_layer_sizes,
            skip_connections = False
        )

        self.phase_offset_head = FCLBlock(
            in_features = in_features,
            blocks = head_layer_sizes,
            skip_connections = False
        )
        
        self.gain_head = FCLBlock(
            in_features = in_features,
            blocks = head_layer_sizes,
            skip_connections = False
        )

        # Optional batch normalization
        if self.normalize:
            self.norm = torch.nn.BatchNorm2d(
                self.embedding_dims,
                momentum=0.05
            )

    def forward(self, bands, descriptor_enc):
        """
        Forward pass through the network. Calculates embedding parameters and applies scaled sinusoidal functions.
        
        Args:
            bands (torch.Tensor): Input tensor representing bands.
            descriptors (torch.Tensor): Input tensor representing descriptors.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """
        
        # Get the device of the bands tensor
        device = bands.device

        # Calculate embedding parameters for all descriptors
        # Add extra dimensions for the width and height of the bands tensor
        # B x C x E x H x W
        freq = einops.rearrange(self.frequency_head(descriptor_enc), 'b c e -> b c e 1 1')
        phase = einops.rearrange(self.phase_offset_head(descriptor_enc), 'b c e -> b c e 1 1')
        gains = einops.rearrange(self.gain_head(descriptor_enc), 'b c e -> b c e 1 1')
        
        # Initialize embeddings tensor B x E x H x W
        embeddings = torch.zeros(
            bands.shape[0], self.embedding_dims, bands.shape[-2], bands.shape[-1]
        ).to(device)
        
        # Add extra dimensions for the embedding dimensions
        bands = einops.rearrange(bands, 'b c h w -> b c 1 h w')
        
        # Calculate embeddings in a loop (slower but more memory efficient)
        for i in range(bands.shape[1]):
            # freq -> B x C x E x 1 x 1 | bands -> B x C x 1 x H x W | phase -> B x C x E x 1 x 1
            modulated_signal = freq[:, i, ...] * (bands[:, i, ...] + phase[:, i, ...])
            embeddings = embeddings + gains[:, i, ...] * torch.sin(modulated_signal)

        # Apply normalization if enabled
        if self.normalize:
            embeddings = self.norm(embeddings)
        
        return embeddings

    def get_output_size(self):
        """
        Get the size of the output embedding.
        
        Returns:
            int: Size of the output embedding.
        """
        return self.embedding_dims
    
class SpectralEmbedding(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """
    
    def __init__(
        self,
        in_features: int,
        m1_percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        m2_blocks: List[int] = [128, 128, 128],
        m3_d_model: int = 128,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32,
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        
        # Initialize the GlobalStats module
        self.global_stats = GlobalStats(
            in_features=in_features,
            percentiles=m1_percentiles,
            device=device
        )
        
        # Initialize the FCLBlock module
        self.fcl_block = FCLBlock(
            in_features=self.global_stats.get_output_size(),
            blocks=m2_blocks
        ).to(device)
        
        # Initialize the AttentionBlock module
        self.attention_block = AttentionBlock(
            in_features=self.fcl_block.get_output_size(),
            d_model=m3_d_model,
            nhead=m3_nhead,
            dim_feedforward=m3_dim_feedforward,
            dropout=m3_dropout,
            num_layers=m3_num_layers
        ).to(device)
        
        # Initialize the BandEmbedding module
        self.band_embedding = BandEmbedding(
            in_features=self.attention_block.get_output_size(),
            embedding_dims=m4_embedding_dims,
            head_layer_sizes=m4_head_layer_sizes,
            skips_heads=m4_skips_heads,
            normalize=m4_normalize
        ).to(device)
        
    def forward(self, bands, descriptor_enc):
        """
        Forward pass through the SpectralEmbedding module.
        
        Args:
            bands (torch.Tensor): Input tensor representing bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """
        
        # Calculate global statistics
        descriptor_enc = self.global_stats(bands, descriptor_enc)

        # Pass through the FCLBlock
        descriptor_enc = self.fcl_block(descriptor_enc)

        # Pass through the AttentionBlock
        descriptor_enc = self.attention_block(descriptor_enc)

        # Pass through the BandEmbedding
        embeddings = self.band_embedding(bands, descriptor_enc)

        return embeddings