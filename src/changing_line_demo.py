import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from data.custom_datasets.straight_line import StraightLine

config = {
    "n_cycles": 10,
    "cycle_length": 20,
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


def rotate_point_square(p):
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


def get_strategy(config):
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


def get_dataset(strategy):
    return StraightLine(split="test",
                        num_images=len(strategy["line"]),
                        num_aug_versions=0,
                        )


def get_data_gen(strategy, dataset):
    for i in range(config["n_cycles"] * config["cycle_length"] + 1):
        yield dataset.get_item(i, line_coords=strategy["line"][i], noise=strategy["noise"][i],
                               n_black_pixels=strategy["black"][i])


class CustomImage:

    def __init__(self):
        self.img_size = 128
        self.img_template = self.create_template_image()

    def to_mask(self, mask):
        mask_colors = [[255, 0, 0], [0, 255, 0], [180, 180, 0], [0, 0, 255]]
        result = np.zeros((3, mask.shape[1], mask.shape[2]))
        for channel in range(mask.shape[0]):
            mask_c = np.ones_like(result) * np.array(mask_colors[channel]).reshape(3, 1, 1)
            mask_idx = np.repeat((mask[channel] > 0.5)[np.newaxis, :, :], 3, axis=0)
            result[mask_idx] = np.clip(result[mask_idx] + mask_c[mask_idx], a_min=0, a_max=255)
        return result.astype("uint8").transpose(1, 2, 0)

    def create_template_image(self):
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
        self.w2 = self.w1 + self.img_size + inner_dist

        self.w3 = self.w2 + self.img_size + inner_dist
        self.h1 = outer_padding + font_size_padded + title_size
        self.h2 = self.h1 + self.img_size + inner_padding + font_size_padded

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
                                 (self.w3 + self.img_size // 2 - 10, self.h2 - font_size_padded), (128, 128, 128), 2)
        output = cv2.arrowedLine(output, (self.w3 + self.img_size // 2 + 10, self.h2 - font_size_padded),
                                 (self.w3 + self.img_size // 2 + 10, self.h1 + self.img_size + 2 * line_padding),
                                 (128, 128, 128), 2)

        return Image.fromarray(output)

    def create_image(self, img, in_features, l1_act, l2_act):
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

        # output.show()
        return np.array(output)


def process_data(generator):
    fps = 10.
    ci = CustomImage()
    out = cv2.VideoWriter('../tmp/demo/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (ci.width, ci.height))
    for img in tqdm(generator, total=config["n_cycles"] * config["cycle_length"] + 1):
        result = ci.create_image(img[0], img[0].repeat(4, 1, 1), img[0].repeat(4, 1, 1), img[0].repeat(4, 1, 1))
        out.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    out.release()


if __name__ == "__main__":
    strategy = get_strategy(config)
    dataset = get_dataset(strategy)
    generator = get_data_gen(strategy, dataset)
    process_data(generator)
