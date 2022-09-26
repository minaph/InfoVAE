import math
import numpy as np
import torch
from torchvision import datasets
from matplotlib import pyplot as plt

import optuna
from optuna.trial import TrialState
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
    plot_pareto_front
)


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples):
    cnt, height, width = (
        int(math.floor(math.sqrt(samples.shape[0]))),
        samples.shape[1],
        samples.shape[2],
    )
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height * cnt, width * cnt])
    return samples


def check_dataset_output(
    tensho: datasets.VisionDataset, image_width: int, image_height: int
):
    batch_size = 100
    train_dataloader = torch.utils.data.DataLoader(
        tensho, batch_size=batch_size, shuffle=True
    )
    imgs, _ = iter(train_dataloader).__next__()

    print("images shape ==>;", imgs.shape)
    plt.imshow(
        convert_to_display(
            imgs.reshape(batch_size, image_width, image_height, 1).numpy()
        ),
        cmap="gray",
    )
    plt.show()


def show_samples_in_grid(samples: torch.Tensor, img_side_length: int):
    plt.imshow(
        convert_to_display(
            samples.reshape(-1, img_side_length, img_side_length, 1).data.cpu()
        ),
        cmap="Greys_r",
    )
    plt.show()


def scatter_3d(data, label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=label)
    plt.show()

def report_optuna_study(study: optuna.study.Study):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    if study._is_multi_objective():
        print("Best trials:")
        for trial in study.best_trials:
            print("  Trial Num: ", trial.number)
            print("    Values: ", trial.values)

            print("    Params: ")
            for key, value in trial.params.items():
                print("    {}={},".format(key, value))
        return

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}={},".format(key, value))


def plot_all_graphs(study: optuna.study.Study):
    if study._is_multi_objective():
        plotters = [
            plot_contour,
            plot_edf,
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        ]
        figs = [plot_pareto_front(study)]
        for i in range(len(study.directions)):
            target_name=f"Objective {i}"
            print(target_name)
            for plotter in plotters:
                fig = plotter(study, target_name=target_name, target=lambda x: x.values[i])
                figs.append(fig)
            
        return figs

    plotters = [
        plot_contour,
        plot_edf,
        plot_intermediate_values,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
    ]

    return [plotter(study) for plotter in plotters]
    