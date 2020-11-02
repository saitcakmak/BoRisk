"""
This is to make the plotting of the analyzer outputs uniform
"""
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

# default color list
# [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2',
# u'#7f7f7f', u'#bcbd22', u'#17becf']

params_dict = {
    "EI": {
        "label": "EI",
        "marker": "x",
        "linestyle": "dotted",
        "errorevery": 1,
        "color": u"#1f77b4",
    },
    "MES": {
        "label": "MES",
        "marker": "x",
        "linestyle": "dashed",
        "errorevery": 1,
        "color": u"#ff7f0e",
    },
    "qKG": {
        "label": "KG",
        "marker": "x",
        "linestyle": "dashdot",
        "errorevery": 1,
        "color": u"#2ca02c",
    },
    "UCB": {
        "label": "UCB",
        "marker": "x",
        "linestyle": (0, (5, 10)),
        "errorevery": 1,
        "color": u"#d62728",
    },
    "classical_random": {
        "label": "random",
        "marker": "x",
        "linestyle": (0, (3, 10, 1, 10)),
        "errorevery": 1,
        "color": u"#9467bd",
    },
    "random": {
        "label": "$\\rho$-random",
        "marker": "p",
        "errorevery": 5,
        "color": u"#8c564b",
    },
    # "tts_kgcp_q=1": {"label": "$\\rho$KG$^{apx}$", "marker": "*", "errorevery": 5},
    "tts_apx_q=1": {
        "label": "$\\rho$KG$^{apx}$",
        "marker": "+",
        "errorevery": 5,
        "color": u"#e377c2",
    },
    # "tts_w_apx_q=1": {"label": "$w-\\rho$KG$^{apx}$", "marker": "v", "errorevery":
    #     5},
    # "apx_cvar_q=1": {"label": "ApxCVaRKG", "marker": "x", "errorevery": 5},
    # "tts_apx_cvar_q=1": {"label": "TTSApxCVaRKG", "marker": "x", "errorevery": 5},
    # "tts_varkg_q=1": {"label": "$\\rho$KG", "marker": "s", "errorevery": 5},
    "tts_rhoKG_q=1": {
        "label": "$\\rho$KG",
        "marker": "s",
        "errorevery": 5,
        "color": u"#7f7f7f",
    },
    # "one_shot_q=1": {"label": "$OS\\rho$KG", "marker": "v", "errorevery": 5},
}

# This is the multiplier for the confidence intervals
beta = 1.0

# Moving average window
ma_window = 3


def plot_out(output, title, ylabel, plot_log):
    """

    :param output: output dict
    :param title: plot title
    :param ylabel: y label
    :return: None
    """
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    for key in params_dict.keys():
        try:
            x = output[key]["x"]
            avg_log_gap = torch.mean(torch.log10(output[key]["y"]), dim=0)
            std_log_gap = torch.std(torch.log10(output[key]["y"]), dim=0) / torch.sqrt(
                torch.tensor(output[key]["y"].size(0), dtype=torch.float)
            )
            avg_gap = torch.mean(output[key]["y"], dim=0)
            std_gap = torch.std(output[key]["y"], dim=0) / torch.sqrt(
                torch.tensor(output[key]["y"].size(0), dtype=torch.float)
            )
            # change these to switch between log and value
            if plot_log:
                avg = avg_log_gap
                std = std_log_gap
            else:
                avg = avg_gap
                std = std_gap

            if ma_window > 1:
                temp_avg = torch.empty(avg.size())
                temp_std = torch.empty(std.size())
                for i in range(avg.size(0)):
                    l_ind = max(0, i - ma_window)
                    temp_avg[i] = torch.mean(avg[l_ind : i + 1], dim=0)
                    temp_std[i] = torch.mean(std[l_ind : i + 1], dim=0)
                avg = temp_avg
                std = temp_std
            # plt.plot(x, avg, **params_dict[key])
            # plt.fill_between(x, avg - beta * std, avg + beta * std, alpha=0.2)
            markers, caps, bars = plt.errorbar(
                x.cpu(), avg.cpu(), yerr=beta * std.cpu(), **params_dict[key], ms=5
            )
            [bar.set_alpha(0.5) for bar in bars]
        except KeyError:
            continue

    plt.xlabel("# of evaluations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if "Covid" in title:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.legend(ncol=2)
    plt.savefig(title + ".pdf", dpi=300)
    plt.show()
