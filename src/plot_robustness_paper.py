import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

JSON_FP = Path(".").absolute().parent / "tmp" / "experiment_results.json"


def replace_square_list(sl):
    if sl == [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]:
        return "1.2 + 0.2t"
    elif sl == [0.7, 0.9, 1.1, 1.3, 1.5, 1.7]:
        return "0.7 + 0.2t"
    elif sl == [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]:
        return "2.0 + 0.1t"
    else:
        raise AttributeError()


def get_data():
    results = []
    with open(str(JSON_FP.absolute())) as f:
        filecontents = f.readlines()
        for entry in filecontents:
            data = json.loads(entry)
            result = {
                'noise': data['config']['noise'],
                'line_interrupt': data['config']['line_interrupt'],
                'act_threshold': data['config']['lateral_model']['l1_params']['act_threshold'],
                'square_factor': replace_square_list(data['config']['lateral_model']['l1_params']['square_factor']),
                'noise_reduction': data['noise_reduction'],
                'avg_line_recon_accuracy_meter': data['avg_line_recon_accuracy_meter'],
                'recon_accuracy': data['recon_accuracy'],
                'recon_recall': data['recon_recall'],
                'recon_precision': data['recon_precision'],
            }
            results.append(result)
    return pd.DataFrame.from_dict(results)


def feature_noise_to_location_noise(feature_noise, round_=False):
    # calculate probability of noise at each spatial location (can occur at each of the 4 feature channels)
    result = 1 - (1-feature_noise)**4
    if round_:
        result = np.round(result, 2)
    return result

def plot_line(data, x_key, x_label, y_key, y_label, z_key, z_label, plot_key, plot_label, x2_func=None, x2_label=None):

    fig, axs = plt.subplots(ncols=len(data[plot_key].unique()), figsize=(18, 6), dpi=100)

    for ax, pk in zip(axs, data[plot_key].unique()):
        data_ = data[data[plot_key] == pk]

        z_values = list(data_[z_key].unique())
        z_values = sorted(z_values)

        for zv in z_values:
            z = data_[data_[z_key] == zv]
            ax.plot(z[x_key].values, z[y_key].values, label="{} = {}".format(z_label, zv))

        ax.set_title("{} = {}".format(plot_label, pk))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if x2_func is not None:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(ax.get_xticks()[1:-1])
            ax2.set_xticklabels(x2_func(ax.get_xticks()[1:-1], round_=True))
            ax2.set_xlabel(x2_label)

        ax.legend()
        ax.grid()

    plt.show()



def plot(data):
    # cleanup data
    data.loc[data.noise == 0, 'noise_reduction'] = 1.0

    # plot noise only
    data_1 = data[data['line_interrupt'] == 0]
    plot_line(data_1, x_key="noise", x_label="Feature Noise", y_key="noise_reduction", y_label="Noise Reduction Rate",
              z_key='act_threshold', z_label='Act. Threshold', plot_key='square_factor', plot_label='Square Factor',
              x2_func=feature_noise_to_location_noise, x2_label="Spatial Noise")

    plot_line(data_1, x_key="noise", x_label="Feature Noise", y_key="recon_recall", y_label="Recall",
              z_key='act_threshold', z_label='Act. Threshold', plot_key='square_factor', plot_label='Square Factor',
              x2_func=feature_noise_to_location_noise, x2_label="Spatial Noise")

    plot_line(data_1, x_key="noise", x_label="Feature Noise", y_key="recon_precision", y_label="Precision",
               z_key='act_threshold', z_label='Act. Threshold', plot_key='square_factor', plot_label='Square Factor',
              x2_func=feature_noise_to_location_noise, x2_label="Spatial Noise")

    # plot line interrupt only
    data_1 = data[data['noise'] == 0.0]
    plot_line(data_1, x_key="line_interrupt", x_label="Line Interrupt", y_key="avg_line_recon_accuracy_meter",
              y_label="Feature Reconstruction Rate",
              z_key='act_threshold', z_label='Act. Threshold', plot_key='square_factor', plot_label='Square Factor')
    plot_line(data_1, x_key="line_interrupt", x_label="Line Interrupt", y_key="recon_recall", y_label="Recall",
              z_key='act_threshold', z_label='Act. Threshold', plot_key='square_factor', plot_label='Square Factor')
    plot_line(data_1, x_key="line_interrupt", x_label="Line Interrupt", y_key="recon_precision", y_label="Precision",
              z_key='act_threshold', z_label='Act. Threshold', plot_key='square_factor', plot_label='Square Factor')


if __name__ == '__main__':
    data = get_data()
    plot(data)
