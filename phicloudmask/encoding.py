import torch
import numpy as np

from typing import Optional, Union

class SEnSeIv2Encoding(torch.nn.Module):
    """
    Converts a list of descriptor dictionaries into a tensor.

    This is necessarily application-specific. The current 
    implementation is focused on cloud masking in multispectral
    instruments, with some additional functionality for cocontemporaneous 
    SAR and DEM data.

    The design principle here is that the dictionaries are relatively easy for a 
    user of the cloud mask algorithm to engineer, requiring only a few lines of
    configuration. The downside is that the implementation is not very flexible,
    and will require editing for other applications.
    """
    
    def __init__(
        self,
        emb_size: int,
        N_embeddings: int,
        input_descriptor_dims: Optional[int] = None,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        
        self.emb_size = emb_size
        self.N_embeddings = N_embeddings
        self.input_descriptor_dims = input_descriptor_dims
        self.device = device

    def forward(self, descriptor_dicts):
        """
        Forward pass to convert descriptor dictionaries into a tensor.
        
        Args:
            descriptor_dicts (list of dict): List of descriptor dictionaries.
        
        Returns:
            torch.Tensor: Tensor representation of the descriptor dictionaries.
        """
        
        # Initialize the output descriptors tensor
        output_descriptors = torch.zeros(
            len(descriptor_dicts),
            len(descriptor_dicts[0]),
            self.emb_size,
            device=self.device
        )
        
        # Iterate over each batch of descriptor dictionaries
        for b, batch_dicts in enumerate(descriptor_dicts):
            for i, d_dict in enumerate(batch_dicts):
                band_type = d_dict['band_type']
                
                # Handle different band types
                if band_type == 'TOA Reflectance' or band_type == 'TOA Normalised Brightness Temperature':
                    min_wavelength = d_dict['min_wavelength']
                    max_wavelength = d_dict['max_wavelength']

                    # Encode the min and max wavelengths separately
                    min_wavelength_enc = self.position_encoding(min_wavelength, self.N_embeddings)
                    max_wavelength_enc = self.position_encoding(max_wavelength, self.N_embeddings)

                    # Add the encoded wavelengths to the output tensor
                    # The final element is a flag to indicate the presence of a band
                    output_descriptors[b, i, :self.N_embeddings] = min_wavelength_enc
                    output_descriptors[b, i, self.N_embeddings:self.N_embeddings * 2] = max_wavelength_enc
                    output_descriptors[b, i, self.N_embeddings * 2] = 1

                    # Add a element if the band is a brightness temperature
                    if band_type == 'TOA Normalised Brightness Temperature':                        
                        output_descriptors[b, i, self.N_embeddings * 2 + 3] = 1

                    # Add a element if the band is multitemporal
                    if d_dict.get('multitemporal', False):
                        output_descriptors[b, i, -1] = 1

                elif 'SAR' in band_type:
                    output_descriptors[b, i, self.N_embeddings * 2 + 1] = 1
                    if 'VV' in band_type:
                        output_descriptors[b, i, :self.N_embeddings] = 1
                    elif 'VH' in band_type:
                        output_descriptors[b, i, self.N_embeddings:self.N_embeddings * 2] = 1
                    elif 'Angle' in band_type:
                        output_descriptors[b, i, :self.N_embeddings * 2] = 1

                elif 'DEM' in band_type:
                    output_descriptors[b, i, self.N_embeddings * 2 + 2] = 1

                elif band_type == 'fill':
                    output_descriptors[b, i, :] = 0

                else:
                    raise ValueError('Unknown band type: {}'.format(band_type))

        return output_descriptors

    def position_encoding(self, val, N):
        """
        Generates sinusoidal position encoding for a given value.
        
        Args:
            val (float): Value to encode.
            N (int): Number of embeddings.
        
        Returns:
            torch.Tensor: Position encoding tensor.
        """
        
        # Calculate position encoding
        position_enc = np.array([val / np.power(10000, 2 * i / N) for i in range(N)])
        position_enc[0::2] = np.sin(position_enc[0::2])
        position_enc[1::2] = np.cos(position_enc[1::2])
        
        return torch.from_numpy(position_enc).type(torch.FloatTensor).to(self.device)

    def get_output_size(self):
        """
        Get the size of the output descriptor.
        
        Returns:
            int: Size of the output descriptor.
        """
        return self.emb_size
