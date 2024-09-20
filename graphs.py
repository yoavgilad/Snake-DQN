import numpy as np
from matplotlib import pyplot as plt


# installing matplotlib on interpreter interferes with docstring, causing :param & :return to be shown as plain text as well.
# deleting matplotlib doesn't help, the only solution is to not install it at first place.
# matplotlib is the most comfortable graphs package I currently know, a decent alternative might be worth switching to.


def new_graph(data, view: bool, save_name: str = None) -> None:
    """
    required data format:
    data = [general labels, line, ..., line]
    general labels = [x-axis label, y-axis label, title]
    line = [[xs], [ys], label]

    :param data:
    :param view:
    :param save_name: None = won't save graph. other = will be saved as png with the given name
    """
    fig, ax = plt.subplots()
    labels = data[0]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(labels[2])
    for line in data[1:]:
        if isinstance(line[-1], float):
            xs, ys, label, op = line
        else:
            op = 1
            xs, ys, label = line
        ax.plot(xs, ys, label=label, alpha=op)
    ax.legend()
    if save_name is not None:
        plt.savefig(save_name + '.png')
    if view:
        plt.show()
    plt.close()


def smoothen(data, filter_size):
    return np.convolve(data, np.ones(filter_size) / filter_size, mode='valid')


def plot_metrics(metrics: list, filter_size: int, save_prename: str | None = None):
    for metric, data in metrics:
        episodes = [i for i in range(len(data))]
        plot_data = [['episodes', metric, f'{metric} over training episodes'], [episodes, data, 'Raw', 0.2],
                     [episodes[filter_size - 1:], smoothen(data, filter_size), f'Moving average ({filter_size})']]
        if save_prename is None:
            new_graph(plot_data, True)
        else:
            new_graph(plot_data, False, save_prename + ' ' + metric)
