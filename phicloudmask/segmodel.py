from transformers import SegformerForSemanticSegmentation, logging

# Get rid of annoying warnings from transformers
logging.set_verbosity_error()

class CustomSegformer:
    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        num_channels: int
    ) -> None:
        
        # Load the pretrained Segformer model
        self.segmenter = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels
        )

        # Modify the model configuration
        self.segmenter.config.num_channels = num_channels
        self.segmenter.config.num_labels = num_labels
        
        # Make new model to get a first layer with correct num. of channels
        new_model = SegformerForSemanticSegmentation(self.segmenter.config)

        # Replace first layer of pretrained model
        self.segmenter.segformer.encoder.patch_embeddings[0] = new_model.segformer.encoder.patch_embeddings[0]

    def forward(self, x):
        return self.segmenter(x)