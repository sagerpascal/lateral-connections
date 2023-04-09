from utils import get_config
from data import loaders_from_config, undo_norm, undo_norm_from_conf, plot_images, show_grid

# TODO: https://github.com/Lightning-Universe/lightning-bolts

config = get_config("base_config")
train_loader, _, test_loader = loaders_from_config(config)

img, lbl = next(iter(train_loader))
img = undo_norm_from_conf(img, config)

plot_images(img, lbl)


