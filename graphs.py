import numpy as np
from matplotlib import pyplot as plt


# Installing matplotlib on interpreter interferes with docstring display in the IDE,
# causing :param & :return to be shown as plain text as well as formatted text.
# Deleting matplotlib doesn't help, the only solution is to not install it at first place.
# Matplotlib is the most comfortable graphs package I currently know,
# but a decent alternative might be worth switching to.


def new_graph(data, view: bool, save_name: str = None) -> None:
    """
    Required data format:

    data = [general labels, line, ..., line]

    general labels = [x-axis label, y-axis label, title]
    line = [[xs], [ys], label, opacity (optional)]

    :param data: According to the specified format
    :param view: whether to display the graph on creation
    :param save_name: None = won't save graph. other = will be saved as png with the given name
    """
    fig, ax = plt.subplots()  # Create a new plot
    # Set texts
    labels = data[0]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(labels[2])
    # Add all the lines
    for line in data[1:]:
        if len(line) == 4:
            xs, ys, label, opacity = line
        else:
            opacity = 1
            xs, ys, label = line
        ax.plot(xs, ys, label=label, alpha=opacity)

    ax.legend()  # Add legend
    if save_name is not None: plt.savefig(save_name + '.png')  # Save if needed
    if view: plt.show()  # Display if needed
    plt.close()


def smoothen(data, filter_size):
    """

    :param data: Single dimensional sequence
    :param filter_size: Size of Moving average filter
    :return:
    """
    return np.convolve(data, np.ones(filter_size) / filter_size, mode='valid')


def plot_metrics(metrics: list, filter_size: int, save_prefix: str | None = None):
    """

    :param metrics: [[name, data], ...]
    :param filter_size: Size of Moving average filter
    :param save_prefix: Graphs will be saved as {prefix} + {metric} + .png
    :return:
    """
    for metric, data in metrics:
        episodes = [game for game in range(len(data))]  # Generate x values
        # Define data according to the format
        plot_data = [['episodes', metric, f'{metric} over training episodes'],
                     [episodes, data, 'Raw', 0.2],
                     [episodes[filter_size - 1:], smoothen(data, filter_size),
                      f'Moving average ({filter_size})']]
        # Generate graph and save if needed
        if save_prefix is None:
            new_graph(plot_data, True)
        else:
            new_graph(plot_data, False, save_prefix + ' ' + metric)
