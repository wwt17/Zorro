"""
compare last or best accuracy between models
"""
from collections import defaultdict
import numpy as np

from zorro import configs
from zorro.visualizer import VisualizerBars, ParadigmDataBars
from zorro.utils import prepare_data_for_plotting, get_phenomena_and_paradigms
from zorro.io import get_group2model_output_paths

# get files locally, where we have runs at single time points only
runs_path = configs.Dirs.runs_local
configs.Eval.local_runs = True

group_names = sorted([p.name for p in runs_path.glob('*')])
print(f'Found {group_names}')

# get list of (phenomenon, paradigm) tuples
phenomena_paradigms = get_phenomena_and_paradigms()

# collects and plots each ParadigmData instance in 1 multi-axis figure
v = VisualizerBars(phenomena_paradigms=phenomena_paradigms)

# for all paradigms
for phenomenon, paradigm in phenomena_paradigms:

    # load model output at all available steps
    group_name2model_output_paths = get_group2model_output_paths(group_names,
                                                                 runs_path,
                                                                 phenomenon,
                                                                 paradigm,
                                                                 )

    # init data
    group_name2template2acc = defaultdict(dict)
    group_name2rep2acc = defaultdict(dict)

    # calc + collect accuracy
    template2group_name2accuracies = prepare_data_for_plotting(group_name2model_output_paths,
                                                               phenomenon,
                                                               paradigm,
                                                               )

    # collect average performance in each paradigm, grouped by replication - allows computation of statistics
    for group_name, accuracies in template2group_name2accuracies['all templates'].items():
        for rep, acc in enumerate(accuracies):
            group_name2rep2acc[group_name][rep] = acc  # collapsed over templates

    # collect average performance in each paradigm, grouped by template
    for template, group_name2accuracies in template2group_name2accuracies.items():
        if template == 'all templates':
            continue
        for group_name, accuracies in group_name2accuracies.items():
            acc_avg_over_reps = np.mean(accuracies)  # average over reps
            group_name2template2acc[group_name][template] = acc_avg_over_reps

    pd = ParadigmDataBars(
        phenomenon=phenomenon,
        paradigm=paradigm,
        group_name2model_output_paths=group_name2model_output_paths,
        group_name2template2acc=group_name2template2acc,
        group_name2rep2acc=group_name2rep2acc,
    )

    # plot each paradigm in separate axis
    v.update(pd)

v.plot_summary()