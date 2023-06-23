import random
import warnings
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from lightning import Fabric
from torch import Tensor
from tqdm import tqdm

from data.custom_datasets.straight_line import StraightLine
from s1_toy_example import configure, cycle, setup_fabric, setup_feature_extractor, setup_l2, setup_lateral_network
from stage_1.lateral.l2_rbm import L2RBM
from stage_1.lateral.lateral_connections_toy import LateralNetwork
from tools.store_load_run import load_run
from utils import print_start

config_demo = {
    "n_cycles": 200,
    "cycle_length": 1,
    "noise":
        {
            "min": 0.0,
            "max": 0.005,
            "probability": 0.2,
        },
    "discontinuous_line":
        {
            "probability": 0.2,
            "min_black": 0,
            "max_black": 5,
        },
    "line":
        {
            "min_length": 15,
            "strategy": "fixed",  # "random" or "fixed"
        }
}


class CustomImage:
    """
    Custom Image class to draw the current state of the network.
    """

    def __init__(self):
        """
        Initialize the class.
        """
        self.img_size = 128
        self.img_template = self.create_template_image()

    def to_mask(self, mask: np.array) -> np.array:
        """
        Convert a mask with 4 channels to an image with 3 channels.
        :param mask: The mask, np array with shape (4, height, width)
        :return: The image, np array with shape (height, width, 3)
        """
        mask_colors = [[255, 0, 0], [0, 255, 0], [180, 180, 0], [0, 0, 255]]
        result = np.zeros((3, mask.shape[1], mask.shape[2]))
        for channel in range(mask.shape[0]):
            mask_c = np.ones_like(result) * np.array(mask_colors[channel]).reshape(3, 1, 1)
            mask_idx = np.repeat((mask[channel] > 0.5)[np.newaxis, :, :], 3, axis=0)
            result[mask_idx] = np.clip(result[mask_idx] + mask_c[mask_idx], a_min=0, a_max=255)
        return result.astype("uint8").transpose(1, 2, 0)

    def create_template_image(self) -> Image:
        """
        Create a template image with all static elements.
        :return: The template image.
        """
        outer_padding = 30
        inner_padding = 80
        inner_dist = 80
        font_size_padded = 20
        font_size = 14
        line_padding = 2

        title_size = font_size * 2
        self.height = 2 * outer_padding + inner_padding + 2 * font_size_padded + 2 * self.img_size + title_size
        self.width = 2 * outer_padding + 2 * inner_dist + 3 * self.img_size
        output = Image.new("RGB", (self.width, self.height), (255, 255, 255))

        # Paste Images
        self.h_center, self.w1 = self.height // 2 - (self.img_size + font_size_padded) // 2 + title_size, outer_padding
        self.w2 = self.w1 + self.img_size + inner_dist  # Sensory System
        self.w3 = self.w2 + self.img_size + inner_dist  # L1 / L2
        self.h1 = outer_padding + font_size_padded + title_size  # L1 Lateral
        self.h2 = self.h1 + self.img_size + inner_padding + font_size_padded  # L2 Prototypes
        self.h3 = self.h_center + self.img_size + 50  # Hidden Activations L2

        logo = Image.open("../fonts/ZHAW.png").resize((30, 30), Image.Resampling.LANCZOS)
        output.paste(logo, (20, self.height - 20 - 30), mask=logo)

        # Add Texts
        font_title = ImageFont.truetype("../fonts/calibrib.ttf", font_size * 2)
        font = ImageFont.truetype("../fonts/calibrib.ttf", font_size)
        font_foot = ImageFont.truetype("../fonts/calibri_italic.ttf", int(font_size * 0.8))
        draw = ImageDraw.Draw(output)
        draw.text((outer_padding, outer_padding), "Lateral Model", (0, 0, 0), font=font_title)
        draw.text((self.w1, self.h_center - font_size_padded), "Input Image", (0, 0, 0), font=font)
        draw.text((self.w2, self.h_center - font_size_padded), "Sensory System", (0, 0, 0), font=font)
        draw.text((self.w3, self.h1 - font_size_padded), "L1 (Lateral)", (0, 0, 0), font=font)
        draw.text((self.w3, self.h2 - font_size_padded), "L2 (Prototypes)", (0, 0, 0), font=font)
        draw.text((self.w2, self.h3 - font_size_padded), "L2 hidden Activations", (0, 0, 0), font=font)

        draw.text((20 + 30 + 10, self.height - 20 - font_size_padded), "MSc. Thesis Pascal Sager", (128, 128, 128),
                  font=font_foot)

        # Draw rectangle
        draw.rounded_rectangle((self.w2 - 20, self.h1 - font_size_padded - 10, self.width - 20, self.height - 20),
                               outline="#0064a6",
                               width=3,
                               radius=7)

        # Draw arrows using cv2
        output = np.array(output)
        output = cv2.arrowedLine(output, (self.w1 + self.img_size + line_padding, self.h_center + self.img_size // 2),
                                 (self.w2 - line_padding, self.h_center + self.img_size // 2), (128, 128, 128), 2)
        output = cv2.arrowedLine(output, (self.w2 + self.img_size + line_padding, self.h_center + self.img_size // 2),
                                 (self.w3 - line_padding, self.h1 + self.img_size - line_padding), (128, 128, 128), 2)
        output = cv2.arrowedLine(output,
                                 (self.w3 + self.img_size // 2 - 10, self.h1 + self.img_size + 2 * line_padding),
                                 (self.w3 + self.img_size // 2 - 10,
                                  self.h2 - font_size_padded - (font_size_padded - font_size)), (128, 128, 128), 2)
        output = cv2.arrowedLine(output, (self.w3 + self.img_size // 2 + 10, self.h2 - font_size_padded -
                                          (font_size_padded - font_size)),
                                 (self.w3 + self.img_size // 2 + 10, self.h1 + self.img_size + 2 * line_padding),
                                 (128, 128, 128), 2)
        output = cv2.arrowedLine(output, (self.w3 - line_padding, self.h3 + 14),
                                 (self.w2 + 8 * 14 + 10 + line_padding, self.h3 + 14), (128, 128, 128), 2)

        return Image.fromarray(output)

    def create_image(self, img: Tensor, in_features: Tensor, l1_act: Tensor, l2_act: Tensor, h: Tensor) -> np.array:
        """
        Creates the image for the current step
        :param img: Input image that was fed into the network with shape (1, v, 1, h, w)
        :param in_features: The input features of the sensory system with shape (1, v, 4, h, w)
        :param l1_act: The activations of the L1 layer with shape (1, v, t, 4, h, w)
        :param l2_act: The activations of the L2 layer with shape (1, v, t, 4, h, w)
        :param h: The hidden activations of the L2 layer with shape (1, v, t, 16)
        :return: The image of the network state as a numpy array
        """
        assert 0. <= img.min() and img.max() <= 1., "img must be in [0, 1]"
        assert 0. <= in_features.min() and in_features.max() <= 1., "in_features must be in [0, 1]"
        assert 0. <= l1_act.min() and l1_act.max() <= 1., "l1_act must be in [0, 1]"
        assert 0. <= l2_act.min() and l2_act.max() <= 1., "l2_act must be in [0, 1]"

        img = Image.fromarray((img * 255).squeeze().cpu().numpy().astype("uint8")).convert("RGB")
        in_features = Image.fromarray(self.to_mask((in_features * 255).squeeze().cpu().numpy()))
        l1_act = Image.fromarray(self.to_mask((l1_act * 255).squeeze().cpu().numpy()))
        l2_act = Image.fromarray(self.to_mask((l2_act * 255).squeeze().cpu().numpy()))

        # Resize Images
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        in_features = in_features.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        l1_act = l1_act.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        l2_act = l2_act.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)

        output = self.img_template.copy()

        # Paste Images
        output.paste(img, (self.w1, self.h_center))
        output.paste(in_features, (self.w2, self.h_center))
        output.paste(l1_act, (self.w3, self.h1))
        output.paste(l2_act, (self.w3, self.h2))

        # Draw hidden activations L2
        draw = ImageDraw.Draw(output)
        for i in range(h.shape[0]):
            fill = (255, 0, 0) if h[i] == 0 else (0, 255, 0)
            pos1 = (self.w2 + (i % 8) * 14, self.h3 + (i // 8) * 14)
            pos2 = (pos1[0] + 10, pos1[1] + 10)
            draw.ellipse((pos1 + pos2), fill=fill)

        # output.show()
        return np.array(output)


def rotate_point_square(p: Tuple[int, int]) -> Tuple[int, int]:
    """
    Rotates a point ccw along a square trajectory
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


def get_strategy(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a strategy for the dataset, i.e. a sequence of noise, black and line values that can be generated as
    images
    :param config: Configuration of the demo, describing the strategy
    :return: The strategy
    """
    strategy = {"noise": [0.0], "black": [0], "line": [((2, 15), (30, 15))]}
    for cycle in range(config["n_cycles"]):
        if random.random() <= config["noise"]["probability"]:
            noise = random.uniform(config["noise"]["min"], config["noise"]["max"])
        else:
            noise = 0.0
        if random.random() <= config["discontinuous_line"]["probability"]:
            black = random.randint(config["discontinuous_line"]["min_black"], config["discontinuous_line"]["max_black"])
        else:
            black = 0

        prev_noise = strategy["noise"][-1]
        prev_black = strategy["black"][-1]
        prev_line = strategy["line"][-1]

        strategy["noise"].extend(list(np.linspace(prev_noise, noise, config["cycle_length"])))
        strategy["black"].extend(list(np.linspace(prev_black, black, config["cycle_length"]).round().astype(int)))

        if config["line"]["strategy"] == "random":
            final_line_length = 0
            while final_line_length < config["line"]["min_length"]:
                final_line = np.random.randint(1, 31, size=(2, 2))
                if final_line[0][0] > final_line[1][0]:
                    final_line = np.flip(final_line, axis=0)
                final_line_length = np.sqrt(np.sum(np.square(final_line[0] - final_line[1])))

            for x1, y1, x2, y2 in zip(
                    list(np.linspace(prev_line[0][0], final_line[0][0], config["cycle_length"]).round().astype(int)),
                    list(np.linspace(prev_line[0][1], final_line[0][1], config["cycle_length"]).round().astype(int)),
                    list(np.linspace(prev_line[1][0], final_line[1][0], config["cycle_length"]).round().astype(int)),
                    list(np.linspace(prev_line[1][1], final_line[1][1], config["cycle_length"]).round().astype(int))):
                strategy["line"].append((((x1, y1), (x2, y2))))


        elif config["line"]["strategy"] == "fixed":
            p1, p2 = prev_line
            for cycle in range(config["cycle_length"]):
                p1 = rotate_point_square(p1)
                p2 = rotate_point_square(p2)
                strategy["line"].append((p1, p2))
        else:
            raise ValueError(f"""Invalid line strategy {config["line"]["strategy"]}.""")

    return strategy


def get_dataset(strategy: Dict[str, Any]) -> StraightLine:
    """
    Generates a dataset for the given strategy
    :param strategy: The strategy
    :return: SrtaightLine dataset
    """
    return StraightLine(split="test",
                        num_images=len(strategy["line"]),
                        num_aug_versions=0,
                        )


def get_data_gen(strategy: Dict[str, Any], dataset: StraightLine):
    """
    Data generator for the given strategy and dataset
    :param strategy: The strategy
    :param dataset: The dataset
    :return: Image generator
    """
    for i in range(config_demo["n_cycles"] + 1):
        images, metas = [], []
        for _ in range(config_demo["cycle_length"]):
            img, meta = dataset.get_item(i, line_coords=strategy["line"][i], noise=strategy["noise"][i],
                                         n_black_pixels=strategy["black"][i])
            images.append(img.unsqueeze(0))
            metas.append(meta)
        yield torch.vstack(images).unsqueeze(0), metas


def load_models() -> Tuple[Dict[str, Any], Fabric, pl.LightningModule, pl.LightningModule, pl.LightningModule]:
    """
    Loads the config, fabric, and models
    :return: Config, fabric, feature extractor, lateral network, l2 network
    """
    config = configure()
    fabric = setup_fabric(config)
    feature_extractor = setup_feature_extractor(config, fabric)
    lateral_network = setup_lateral_network(config, fabric)
    l2 = setup_l2(config, fabric)

    assert 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None', \
        "Please specify a path to a trained model."

    config, state = load_run(config, fabric)
    feature_extractor.load_state_dict(state['feature_extractor'])
    lateral_network.load_state_dict(state['lateral_network'])
    l2.load_state_dict(state['l2'])

    feature_extractor.eval()
    lateral_network.eval()
    l2.eval()

    return config, fabric, feature_extractor, lateral_network, l2


def load_data_generator() -> Iterator[Tuple[Tensor, List[Dict[str, Any]]]]:
    """
    Loads the data generator
    :return: Data generator
    """
    strategy = get_strategy(config_demo)
    dataset = get_dataset(strategy)
    generator = get_data_gen(strategy, dataset)
    return generator


def predict_sample(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        l2: L2RBM,
        batch: Tensor,
        batch_idx: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Predicts the features for a given sample
    :param config: Configuration
    :param fabric: Fabric instance
    :param feature_extractor: Feature extractor
    :param lateral_network: Lateral network (L1)
    :param l2: L2 network
    :param batch: Data batch
    :param batch_idx: Batch index
    :return: Features from feature extractor, lateral network, and l2 network
    """
    with torch.no_grad():
        batch = batch[0].to(fabric.device)
        features, input_features, lateral_features, lateral_features_f, l2_features = cycle(config,
                                                                                            feature_extractor,
                                                                                            lateral_network, l2,
                                                                                            batch,
                                                                                            batch_idx,
                                                                                            epoch=1,
                                                                                            # no lateral feedback
                                                                                            store_tensors=True,
                                                                                            mode="eval")
    return input_features, lateral_features, l2_features


def process_data(
        generator: Iterator[Tuple[Tensor, List[Dict[str, Any]]]],
        config: Dict[str, Any],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        lateral_network: pl.LightningModule,
        l2: pl.LightningModule,
        video_fp: str = '../tmp/demo/output.mp4',
):
    """
    Processes the data and store the network activations as video
    :param generator: Data generator
    :param config: Configuration
    :param fabric: Fabric instance
    :param feature_extractor: Feature extractor
    :param lateral_network: Lateral network (L1)
    :param l2: L2 network
    :param video_fp: Video file path
    """
    fps = 10.
    ci = CustomImage()
    out = cv2.VideoWriter(video_fp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ci.width, ci.height))
    for i, img in tqdm(enumerate(generator), total=config_demo["n_cycles"] + 1):
        inp_features, l1_act, l2_act = predict_sample(config, fabric, feature_extractor, lateral_network, l2, img,
                                                      i)
        for view in range(img[0].shape[1]):
            for timestep in range(l1_act.shape[2]):
                result = ci.create_image(img[0][0, view], inp_features[0, view], l1_act[0, view, timestep],
                                         l2_act[0, view, timestep, :-1], l2_act[0, view, timestep, -1, 0, :16])
                out.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    out.release()


def main():
    """
    Main function
    """
    print_start("Starting python script 'changing_line_demo.py'...",
                title="Demo Lines: Creating a Video of a Changing Line")
    config, fabric, feature_extractor, lateral_network, l2 = load_models()
    generator = load_data_generator()
    process_data(generator, config, fabric, feature_extractor, lateral_network, l2)


if __name__ == "__main__":
    main()
    warnings.warn("Set cycle_length to a value that corresponds to a 45° rotation.")
