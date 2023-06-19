import random
import cv2
import numpy as np

from data.custom_datasets.straight_line import StraightLine

config = {
    "n_cycles": 100,
    "cycle_length": 20,
    "noise":
        {
            "min": 0.0,
            "max": 0.005,
            "probability": 0.2,
        },
    "discontinous_line":
        {
            "probability": 0.2,
            "min_black": 0,
            "max_black": 5,
        },
    "line":
        {
            "min_length": 15,
        }
}


def get_strategy(config):
    strategy = {"noise": [0.0], "black": [0], "line": [((2, 15), (30, 15))]}
    for cycle in range(config["n_cycles"]):
        if random.random() <= config["noise"]["probability"]:
            noise = random.uniform(config["noise"]["min"], config["noise"]["max"])
        else:
            noise = 0.0
        if random.random() <= config["discontinous_line"]["probability"]:
            black = random.randint(config["discontinous_line"]["min_black"], config["discontinous_line"]["max_black"])
        else:
            black = 0

        prev_noise = strategy["noise"][-1]
        prev_black = strategy["black"][-1]
        prev_line = strategy["line"][-1]

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

        strategy["noise"].extend(list(np.linspace(prev_noise, noise, config["cycle_length"])))
        strategy["black"].extend(list(np.linspace(prev_black, black, config["cycle_length"]).round().astype(int)))

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


def process_data(generator):
    fps = 10.
    out = cv2.VideoWriter('../tmp/demo/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (32, 32), False)
    for img in generator:
        out.write((img[0]*255).squeeze().cpu().numpy().astype("uint8"))
    out.release()



if __name__ == "__main__":
    strategy = get_strategy(config)
    dataset = get_dataset(strategy)
    generator = get_data_gen(strategy, dataset)
    process_data(generator)
