"""
This is to make the plotting of the analyzer outputs uniform
"""
import torch
import matplotlib.pyplot as plt

params_dict = {
    'EI': {
        'label': 'EI',
        'marker': 'x',
        'linestyle': 'dotted',
        'errorevery': 1
    },
    'MES': {
        'label': 'MES',
        'marker': 'x',
        'linestyle': 'dashed',
        'errorevery': 1
    },
    'qKG': {
        'label': 'KG',
        'marker': 'x',
        'linestyle': 'dashdot',
        'errorevery': 1
    },
    'UCB': {
        'label': 'UCB',
        'marker': 'x',
        'linestyle': (0, (5, 10)),
        'errorevery': 1
    },
    'classical_random': {
        'label': 'random',
        'marker': 'x',
        'linestyle': (0, (3, 10, 1, 10)),
        'errorevery': 1
    },
    'random': {
        'label': '$\\rho$-random',
        'marker': 'p',
        'errorevery': 5
    },
    'tts_kgcp_q=1': {
        'label': '$\\rho$KG$^{apx}$',
        'marker': '*',
        'errorevery': 5
    },
    'tts_varkg_q=1': {
        'label': '$\\rho$KG',
        'marker': 's',
        'errorevery': 5
    }
}
"""
Things to add here:
    ylim if needed
    plot style
    markers
    
"""

# This is the multiplier for the confidence intervals
beta = 1.

# Moving average window
ma_window = 1


def plot_out(output, title, ylabel, plot_log):
    """

    :param output: output dict
    :param title: plot title
    :param ylabel: y label
    :return: None
    """
    plt.figure(figsize=(8, 6))
    for key in params_dict.keys():
        try:
            x = output[key]['x']
            avg_log_gap = torch.mean(torch.log10(output[key]['y']), dim=0)
            std_log_gap = torch.std(torch.log10(output[key]['y']), dim=0) / torch.sqrt(
                torch.tensor(output[key]['y'].size(0), dtype=torch.float))
            avg_gap = torch.mean(output[key]['y'], dim=0)
            std_gap = torch.std(output[key]['y'], dim=0) / torch.sqrt(
                torch.tensor(output[key]['y'].size(0), dtype=torch.float))
            # change these to switch between log and value
            if plot_log:
                avg = avg_log_gap
                std = std_log_gap
            else:
                avg = avg_gap
                std = std_gap

            if ma_window > 1 and key in ['random', 'tts_kgcp_q=1', 'tts_varkg_q=1']:
                temp_avg = torch.empty(avg.size())
                temp_std = torch.empty(std.size())
                for i in range(avg.size(0)):
                    l_ind = max(0, i - ma_window)
                    temp_avg[i] = torch.mean(avg[l_ind:i+1], dim=0)
                    temp_std[i] = torch.mean(std[l_ind:i+1], dim=0)
                avg = temp_avg
                std = temp_std
            # plt.plot(x, avg, **params_dict[key])
            # plt.fill_between(x, avg - beta * std, avg + beta * std, alpha=0.2)
            markers, caps, bars = plt.errorbar(x, avg, yerr=beta*std, **params_dict[key], ms=5)
            [bar.set_alpha(0.5) for bar in bars]
        except KeyError:
            continue

    plt.xlabel("# of evaluations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(ncol=2)
    plt.show()
