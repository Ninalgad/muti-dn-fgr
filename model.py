from pathlib import Path
from medpy.io.load import load
import numpy as np
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniqueImagesValidator,
    UniquePathIndicesValidator,
)

from ternausnet import SimpNet
from utils import predict_probabilities
from postprocess_probability_maps import postprocess_single_probability_map

RESOURCE_PATH = Path("resources")


class FetalAbdomenSegmentation(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # Initialize the predictor
        self.predictor = self.threshold = None
        self.initialize_predictor()

    def initialize_predictor(self, checkpoint="checkpoint_final.pt"):
        """
        Initializes the nnUNet predictor
        """
        # instantiates the predictor
        self.predictor = SimpNet()

        # initializes the network architecture, loads the checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor.to(device)
        checkpoint = torch.load(RESOURCE_PATH / checkpoint, device)
        self.threshold = checkpoint['best_thresh']
        self.predictor.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, input_img_path, debug=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """
        # ideally we would like to use predictor.predict_from_files but this docker container will be called
        # for each individual test case so that this doesn't make sense
        image_np, _ = load(input_img_path)
        if debug:
            image_np = image_np[:, :, :2]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        probabilities, relative_distances = predict_probabilities(image_np, self.predictor, device)
        return probabilities, relative_distances

    def postprocess(self, probability_map):
        """
        Postprocess the nnUNet output to generate the final AC segmentation mask
        """
        # Define the postprocessing configurations
        configs = {
            "threshold": self.threshold
        }

        # Postprocess the probability map
        mask_postprocessed = postprocess_single_probability_map(
            probability_map, configs)
        print('Postprocessing done')
        return mask_postprocessed
