from medpy.io.load import load
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniqueImagesValidator,
    UniquePathIndicesValidator,
)

from ternausnet import SimpNet
from utils import predict_probabilities, smooth
from postprocess_probability_maps import postprocess_single_probability_map


class FetalAbdomenSegmentation(SegmentationAlgorithm):
    def __init__(self, checkpoint=None):
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
        self.initialize_predictor(checkpoint)

    def load_checkpoint(self, checkpoint):
        """
        Loads predictor weights
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint, device)
        self.threshold = checkpoint['best_thresh']
        self.predictor.load_state_dict(checkpoint['model_state_dict'])

    def initialize_predictor(self, checkpoint=None):
        """
        Initializes the UNet predictor
        """
        # instantiates the predictor
        self.predictor = SimpNet()
        # initializes the network architecture
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor.to(device)

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def predict(self, input_img_path, debug=False):
        """
        Use trained UNet network to generate segmentation masks
        """
        # ideally we would like to use predictor.predict_from_files but this docker container will be called
        # for each individual test case so that this doesn't make sense
        image_np, _ = load(input_img_path)
        if debug:
            image_np = image_np[:, :, :2]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seg_probabilities, frame_probabilities = predict_probabilities(image_np, self.predictor, device)
        return seg_probabilities, frame_probabilities

    def postprocess(self, probability_map, suitability_scores):
        """
        Postprocess the UNet output to generate the final AC segmentation mask
        and suitability scores
        """
        # Define the postprocessing configurations
        configs = {
            "threshold": self.threshold
        }

        # Postprocess the probability map
        mask_postprocessed = postprocess_single_probability_map(
            probability_map, configs)

        # Postprocess suitability score
        scores_postprocessed = smooth(suitability_scores.astype("float32"))
        print('Postprocessing done')

        return mask_postprocessed, scores_postprocessed
