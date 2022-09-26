
import gc
import torch
from torch import optim
import optuna
from pathlib import Path
from math import isnan
from statistics import median

# from memory_profiler import profile

# from tqdm import trange

from ..util import make_dataset
from ..config import setup, get_study, get_study_multi_obj, get_default_args, save, load, delete_trial
from .. import config

# from ..util import report_optuna_study

z_dim = 3
optuna_path = Path.cwd() / "Tensho_InfoVAE" / "optuna"


def make_optimizer(mode=str):
    # N_TRAIN_EXAMPLES = config.get("train_batch_size") * 40
    # N_VALID_EXAMPLES = config.get("val_batch_size") * 10

    train_data, valid_data, _ = make_dataset()

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.get("train_batch_size"), shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=config.get("val_batch_size"), shuffle=True
    )

    if config.get("usecuda"):
        device = torch.device(config.get("idgpu"))
    if config.get("usemps"):
        device = torch.device("mps")

    model = setup()
    try:
        load("tuning", model)
    except FileNotFoundError:
        pass
    save("tuning", model)
    
    def objective(trial: optuna.Trial):
        config.set_optuna(trial, mode)
        load("tuning", model)
        
        if mode == "loss":
            model.kernel_type = config.get("kernel_type")
            model.latent_var = config.get("latent_var")

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate"),
            weight_decay=config.get("weight_decay"),
        )

        model.set_randn(
            config.get("epochs") * len(train_loader),
            config.get("train_batch_size"),
            device=device,
        )

        for epoch in range(config.get("epochs")):
            # N_TRAIN_EXAMPLES_FOR_THIS_EPOCH = N_TRAIN_EXAMPLES

            for batch_idx, (train_x, _) in enumerate(train_loader):
                # if (batch_idx - epoch) * config.get(
                #     "train_batch_size"
                # ) > N_TRAIN_EXAMPLES_FOR_THIS_EPOCH:
                #     break
                train_x = train_x.to(device)

                optimizer.zero_grad()
                output = model(train_x, training=(batch_idx != len(train_loader) - 1))
                loss = model.loss_function(*output)
                if isnan(loss["loss"].data.item()) or loss["loss"].data.item() > 0.9:
                    # raise ValueError("Loss is NaN or too large")
                    return float("nan")
                loss["loss"].backward()
                del loss["loss"], loss, output, train_x
                optimizer.step()

            if mode == "learning":
                loss_list = []
                with torch.no_grad():
                    for batch_idx, (valid_x, _) in enumerate(valid_loader):
                        # if batch_idx * config.get("val_batch_size") > N_VALID_EXAMPLES:
                        #     break

                        valid_x = valid_x.to(device)

                        output = model(valid_x, training=False)
                        loss = model.loss_function(*output, M_N=1.0)
                        loss_v = loss["loss"].data.item()

                        del loss["loss"], loss, output, valid_x

                        if isnan(loss_v):
                            continue
                        loss_list.append(loss_v)
                    
                    trial.report(median(loss_list), epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                del loss_list


        recon_list = []
        mmd_list = []
        loss_list = []
        with torch.no_grad():
            for batch_idx, (valid_x, _) in enumerate(valid_loader):
                # if batch_idx * config.get("val_batch_size") > N_VALID_EXAMPLES:
                #     break

                valid_x = valid_x.to(device)

                output = model(valid_x, training=False)
                loss = model.loss_function(*output, M_N=1.0)
                recon_list.append(loss["Reconstruction_Loss"].data.item())
                mmd_list.append(loss["MMD"].data.item())
                loss_list.append(loss["loss"].data.item())
                del loss["loss"]
        
        if mode == "learning":
            result = median(loss_list)
        elif mode == "loss":
            result = [median(recon_list), median(mmd_list)]

        del recon_list, mmd_list, loss_list, optimizer

        delete_trial()
        torch.cuda.empty_cache()
        gc.collect()

        return result

    def optimze(n_trials: int):
        study_name = f"tuning_{mode}_{config.get('hidden_dims')}"
        if mode == "learning":
            study = get_study(study_name)
        elif mode == "loss":
            study = get_study_multi_obj(study_name)
        # study.enqueue_trial(get_default_args())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study

    # def release():
    #     del model, train_loader, valid_loader, train_data, valid_data, device, _
    #     gc.collect()

    return optimze


