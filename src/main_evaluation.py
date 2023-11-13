import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import lightning.pytorch as pl
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from lightning import Fabric
from torch import Tensor
from tqdm import tqdm

from data.custom_datasets.straight_line import StraightLine
from lateral_connections.s1_lateral_connections import LateralNetwork
from main_lateral_connections import configure, setup_fabric, setup_feature_extractor, setup_lateral_network
from tools import AverageMeter
from tools.store_load_run import load_run
from utils import print_start


def parse_args(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """
    Parse arguments from command line.
    :param parser: Optional ArgumentParser instance.
    :return: Parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Model Evaluation")

    parser.add_argument("--n_samples",
                        type=int,
                        metavar="N",
                        default=300,
                        help="Number of samples to evaluate."
                        )
    parser.add_argument('--simplified',
                        action='store_true',
                        default=False,
                        help='Use simple dataset only containing lines with angels of 0°, 45°, -45°, and 90°.'
                        )
    parser.add_argument('--add_noise',
                        action='store_true',
                        default=False,
                        help='Add noise to evaluation samples.'
                        )
    parser.add_argument("--line_interrupt",
                        type=int,
                        metavar="N",
                        default=0,
                        help="Number of pixels to remove from the line."
                        )
    parser.add_argument("--fps",
                        type=int,
                        metavar="N",
                        default=10,
                        help="Number of samples to evaluate."
                        )

    return parser


def load_models() -> Tuple[Dict[str, Any], Fabric, pl.LightningModule, pl.LightningModule]:
    """
    Loads the config, fabric, and models
    :return: Config, fabric, feature extractor, lateral network, l2 network
    """
    config = configure(parse_args())
    fabric = setup_fabric(config)
    feature_extractor = setup_feature_extractor(config, fabric)
    lateral_network = setup_lateral_network(config, fabric)

    assert 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None', \
        "Please specify a path to a trained model."

    config, state = load_run(config, fabric)
    feature_extractor.load_state_dict(state['feature_extractor'])
    lateral_network.load_state_dict(state['lateral_network'])

    feature_extractor.eval()
    lateral_network.eval()

    return config, fabric, feature_extractor, lateral_network


def get_datapoints_simple(n_points) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    points = []
    for idx in range(n_points):
        if idx % 4 == 0:
            points.append(((2, 16), (30, 16)))
        elif idx % 4 == 1:
            points.append(((16, 2), (16, 30)))
        elif idx % 4 == 2:
            points.append(((2, 2), (30, 30)))
        elif idx % 4 == 3:
            points.append(((2, 30), (30, 2)))
    return points


def get_datapoints(n_points) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    def rotate_point_square(p: Tuple[int, int]) -> Tuple[int, int]:
        """
        Rotates a point ccw along a square trajectory -> required for straight line dataset
        :param p: Current position
        :return: Next position
        """
        (x, y) = p
        if x == 2:
            if y < 30:
                y += 1
            else:
                x += 1
        elif x < 30:
            if y == 2:
                x -= 1
            else:
                x += 1
        else:
            if y > 2:
                y -= 1
            else:
                x -= 1
        return (x, y)

    p1 = (2, 16)
    p2 = (30, 16)
    points = [(p1, p2)]

    for idx in range(n_points - 1):
        p1 = rotate_point_square(p1)
        p2 = rotate_point_square(p2)
        points.append((p1, p2))
    return points


class CustomImage:
    """
    Custom Image class to draw the current state of the network.
    """

    def __init__(self):
        """
        Initialize the class.
        """
        self.img_size = 256
        self.img_template = self.create_template_image()

    def to_mask(self, mask: np.array) -> np.array:
        """
        Convert a mask with to an image with 3 channels.
        :param mask: The mask, np array with shape (c, height, width)
        :return: The image, np array with shape (height, width, 3)
        """

        mask_colors = matplotlib.colormaps['gist_rainbow'](range(0, 256, 256 // mask.shape[0]))
        result = np.zeros((3, mask.shape[1], mask.shape[2]))
        for channel in range(mask.shape[0]):
            mask_c = np.ones_like(result) * (mask_colors[channel, :3] * 255).astype(int).reshape(3, 1, 1)
            mask_idx = np.repeat((mask[channel] > 0.5)[np.newaxis, :, :], 3, axis=0)
            result[mask_idx] = np.clip(result[mask_idx] + mask_c[mask_idx], a_min=0, a_max=255)
        return result.astype("uint8").transpose(1, 2, 0)

    def to_heatmap(self, activation_probabilities: np.array) -> np.array:
        """
        Convert the activation probabilities to a heatmap.
        :param activation_probabilities: The activation probabilities, np array with shape (c, height, width)
        :return: The heatmap, np array with shape (height, width, 3)
        """
        heatmap = (np.max(activation_probabilities, axis=0) * 255).astype("uint8")
        heatmap = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
        return heatmap

    def create_template_image(self) -> Image:
        """
        Create a template image with all static elements.
        :return: The template image.
        """
        outer_padding = 30
        inner_padding = 80
        inner_dist = 80
        font_size_padded = 30
        font_size = 20
        line_padding = 2

        title_size = int(font_size * 1.7)
        self.height = 2 * outer_padding + inner_padding + 2 * font_size_padded + 1 * self.img_size + title_size
        self.width = 2 * outer_padding + 3 * inner_dist + 4 * self.img_size
        output = Image.new("RGB", (self.width, self.height), (255, 255, 255))

        # Paste Images
        self.h_center, self.w1 = self.height // 2 - (self.img_size + font_size_padded) // 2 + title_size, outer_padding
        self.w2 = self.w1 + self.img_size + inner_dist  # S1
        self.w3 = self.w2 + self.img_size + inner_dist  # S2
        self.w4 = self.w3 + self.img_size + inner_dist  # S2 Probabilities

        logo = Image.open("../fonts/ZHAW.png").resize((60, 60), Image.Resampling.LANCZOS)
        output.paste(logo, (20, self.height - 20 - 60), mask=logo)

        # Add Texts
        font_title = ImageFont.truetype("../fonts/calibrib.ttf", title_size)
        font = ImageFont.truetype("../fonts/calibrib.ttf", font_size)
        font_foot = ImageFont.truetype("../fonts/calibri_italic.ttf", int(font_size * 0.8))
        draw = ImageDraw.Draw(output)
        draw.text((outer_padding, outer_padding), "Net Fragments as the Brain’s Neural Code to Prevent Early Commitment", (0, 100, 166), font=font_title)
        draw.text((self.w1, self.h_center - font_size_padded), "Input Image", (40, 40, 40), font=font)
        draw.text((self.w2, self.h_center - font_size_padded), "Feature Activation (S1)", (40, 40, 40), font=font)
        draw.text((self.w3, self.h_center - font_size_padded), "Net Fragments (S2)", (40, 40, 40), font=font)
        draw.text((self.w4, self.h_center - font_size_padded), "Net Fragments (S2) Probabilities", (40, 40, 40), font=font)

        draw.text((20 + 60 + 10, self.height - 20 - font_size_padded), "Sager et al.", (128, 128, 128),
                  font=font_foot)

        # Draw rectangle
        draw.rounded_rectangle((self.w2 - 20, self.h_center - font_size_padded - 10, self.width - 15, self.height - 80),
                               outline="#0064a6",
                               width=3,
                               radius=7)

        # Draw arrows using cv2
        output = np.array(output)
        output = cv2.arrowedLine(output, (self.w1 + self.img_size + line_padding, self.h_center + self.img_size // 2),
                                 (self.w2 - line_padding, self.h_center + self.img_size // 2), (128, 128, 128), 5)
        output = cv2.arrowedLine(output, (self.w2 + self.img_size + line_padding, self.h_center + self.img_size // 2),
                                 (self.w3 - line_padding, self.h_center + self.img_size // 2),
                                 (128, 128, 128), 5)

        return Image.fromarray(output)

    def create_image(self, img: Tensor, in_features: Tensor, l1_act: Tensor, l1_act_prob: Tensor) -> np.array:
        """
        Creates the image for the current step
        :param img: Input image that was fed into the network with shape (1, v, 1, h, w)
        :param in_features: The input features of the sensory system with shape (1, v, 4, h, w)
        :param l1_act: The activations of the L1 layer with shape (1, v, t, 4, h, w)
        :param l1_act_prob: The activation probabilities of the L1 layer with shape (1, v, t, 4, h, w)
        :return: The image of the network state as a numpy array
        """
        assert 0. <= img.min() and img.max() <= 1., "img must be in [0, 1]"
        assert 0. <= in_features.min() and in_features.max() <= 1., "in_features must be in [0, 1]"
        assert 0. <= l1_act.min() and l1_act.max() <= 1., "l1_act must be in [0, 1]"
        assert 0. <= l1_act_prob.min() and l1_act_prob.max() <= 1., "l1_act_prob must be in [0, 1]"

        img = Image.fromarray((img * 255).squeeze().cpu().numpy().astype("uint8")).convert("RGB")
        in_features = Image.fromarray(self.to_mask((in_features * 255).squeeze().cpu().numpy()))
        l1_act = Image.fromarray(self.to_mask((l1_act * 255).squeeze().cpu().numpy()))
        l1_act_prob = Image.fromarray(self.to_heatmap((l1_act_prob).squeeze().cpu().numpy()))

        # Resize Images
        img = img.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        in_features = in_features.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        l1_act = l1_act.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        l1_act_prob = l1_act_prob.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        output = self.img_template.copy()

        # Paste Images
        output.paste(img, (self.w1, self.h_center))
        output.paste(in_features, (self.w2, self.h_center))
        output.paste(l1_act, (self.w3, self.h_center))
        output.paste(l1_act_prob, (self.w4, self.h_center))

        # output.show()
        return np.array(output)


def get_data_generator(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get data generator
    :param config: Configuration.
    :return: Data generator.
    """
    if config['simplified']:
        points = get_datapoints_simple(config['n_samples'])
    else:
        points = get_datapoints(config['n_samples'])

    dataset = StraightLine(split="train", num_images=len(points), num_aug_versions=0)
    for i in range(len(points)):
        img, meta = dataset.get_item(i, line_coords=points[i], n_black_pixels=config['line_interrupt'])
        yield img, meta



def merge_alt_channels(config: Dict[str, Optional[Any]], lateral_features: List[Tensor]) -> List[Tensor]:
    """
    Merge the alternative channels of the lateral features to the original channels.
    (Possible since only one channel per alternative channels ist active).
    -> This allows better visualization of the lateral features (4 channels vs. 80 channels)

    :param lateral_features: The lateral features with alternative channels
    :return: The lateral features with original channels
    """
    n_alt = config['n_alternative_cells']
    n_channels = config['lateral_model']['channels']

    result = []
    for lf in lateral_features:
        lf = lf.reshape(-1, n_channels, n_alt, lf.shape[2], lf.shape[3])
        assert torch.sum((lf > 0), dim=2).max() <= 1, "Only one channel per alternative channel can be active"
        result.append(torch.max(lf, dim=2)[0])

    return result


def analyze_noise(noise: Tensor, random_mask: Tensor, lateral_features: List[Tensor]) -> float:
    """
    Analyzes how well noise can be reduced.

    :param noise:
    :param features:
    :param random_mask:
    :param lateral_features:
    :return: The ratio of removed noise
    """
    lateral_features = lateral_features[-1].view(-1)
    lateral_features_noise = lateral_features[random_mask]
    removed_noise_ratio = torch.sum(lateral_features_noise != noise) / torch.numel(noise)
    return removed_noise_ratio.item()


def predict_sample(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        batch: Tensor,
        batch_idx: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """
    Predicts the features for a given sample
    :param config: Configuration
    :param fabric: Fabric instance
    :param feature_extractor: Feature extractor
    :param lateral_network: Lateral network (L1)
    :param batch: Data batch
    :param batch_idx: Batch index
    :return: Features from feature extractor, lateral network, and l2 network
    """
    input, input_features, lateral_features, lateral_features_float = [], [], [], []

    with torch.no_grad():
        batch = batch[0].to(fabric.device)
        features = feature_extractor(batch.unsqueeze(0))
        features = feature_extractor.binarize_features(features).squeeze(1)

        if config['add_noise']:
            features_s = features.shape
            num_elements = features.numel()
            num_flips = int(0.005 * num_elements)
            random_mask = torch.randperm(num_elements)[:num_flips]
            random_mask = torch.zeros(num_elements, dtype=torch.bool).scatter(0, random_mask, 1)
            features = features.view(-1)
            noise = 1.0 - features[random_mask]
            features[random_mask] = noise
            features = features.view(features_s)

        lateral_network.new_sample()
        z = torch.zeros((features.shape[0], lateral_network.model.out_channels, features.shape[2],
                         features.shape[3]), device=batch.device)

        for t in range(config["lateral_model"]["max_timesteps"]):
            lateral_network.model.update_ts(t)
            x_in = torch.cat([features, z], dim=1)
            z_float, z = lateral_network(x_in)

            input.append(batch)
            input_features.append(features)
            lateral_features.append(z)
            lateral_features_float.append(z_float)

    lateral_features = merge_alt_channels(config, lateral_features)
    lateral_features_float = merge_alt_channels(config, lateral_features_float)
    removed_noise = analyze_noise(noise, random_mask, lateral_features) if config['add_noise'] else 0
    return (torch.stack(input), torch.stack(input_features), torch.stack(lateral_features),
            torch.stack(lateral_features_float), removed_noise)


def process_data(
        generator,
        config: Dict[str, Any],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        lateral_network: pl.LightningModule,
):
    """
    Processes the data and store the network activations as video
    :param generator: Data generator
    :param eval_args: Evaluation arguments
    :param config: Configuration
    :param fabric: Fabric instance
    :param feature_extractor: Feature extractor
    :param lateral_network: Lateral network (L1)
    """
    ci = CustomImage()
    avg_noise_meter = AverageMeter()
    fp = f"../tmp/v2/{config['run']['load_state_path'].split('.')[0]}_{'noise' if config['add_noise'] else 'no-noise'}_li-{config['line_interrupt']}.mp4"
    if Path(fp).exists():
        Path(fp).unlink()
    out = cv2.VideoWriter(fp, cv2.VideoWriter_fourcc(*'mp4v'), config['fps'],
                          (ci.width, ci.height))
    for i, img in tqdm(enumerate(generator), total=config["n_samples"]):
        inp, inp_features, l1_act, l1_act_prob, removed_noise = predict_sample(config, fabric, feature_extractor, lateral_network, img, i)
        l1_act_prob = torch.where((l1_act > 0.) | (inp_features > 0.), l1_act_prob, torch.zeros_like(l1_act_prob))
        avg_noise_meter(removed_noise)
        for timestep in range(l1_act.shape[0]):
            result = ci.create_image(inp[timestep], inp_features[timestep, 0], l1_act[timestep, 0], l1_act_prob[timestep, 0])
            out.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    out.release()
    print("Video stored at", fp)
    print(f"Average Noise Reduction: {avg_noise_meter.mean}")


def main():
    """
    Main function
    """
    print_start("Starting python script 'main_evaluation.py'...",
                title="Evaluating Model and Print activations")
    config, fabric, feature_extractor, lateral_network = load_models()
    args = parse_args()
    config = config | vars(args)
    generator = get_data_generator(config)
    process_data(generator, config, fabric, feature_extractor, lateral_network)

if __name__ == "__main__":
    main()
