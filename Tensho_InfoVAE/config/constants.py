from typing import Any, Literal
from argparse import Namespace
from optuna import Trial
from pathlib import Path

__called: set = set()
__trial: Trial = None
__optuna_names: list[str] = []
__optuna_mode: Literal["learning", "loss"] = None


# x=0.7796223602963481
# y=0.2597602249094708
# y *= 1 - x
# z = 1.0 - x - y

scale_factor = 3.5

__namespace = Namespace(
    usecuda=False,
    usemps=True,
    idgpu=2,
    epochs=10,
    model_path=Path(__file__).parent.parent / "checkpoints",
    # constrain_learned_variance=False,
    
    kernel_type="rbf",
    # learning_rate=0.008,
    # learning_rate=0.00996,
    weight_decay=0.0001,

    train_batch_size=int(60 * scale_factor),
    val_batch_size=100,
    test_batch_size=64,
    hidden_dims=[16, 32, 128, 128, 128],

    # alpha=-1.0,
    reg_weight=1.0,
    # kld_weight=y/2,
    # beta=z,

    latent_var=3.6866278135394097,
    learning_rate=(0.00036859862595040017) * scale_factor,

    # latent_var=3,
    scheduler_gamma=0.95,
)


def get_default_args():
    global __namespace
    copy_kv = filter(lambda v: v[0] in __optuna_names, __namespace.__dict__.items())
    return dict(copy_kv)


def set_optuna(trial: Trial, mode: Literal["learning", "loss"]):
    global __trial, __optuna_mode
    __trial = trial
    __optuna_mode = mode

    if mode == "learning":
        __optuna_names.extend(
            "learning_rate".split(" ")
        )
    elif mode == "loss":
        __optuna_names.extend("latent_var".split(" "))


def namespace_optuna():
    trial = __trial
    # hidden_dims = [
    #     trial.suggest_int("hidden_dims_0", 16, 512, step=16),
    #     trial.suggest_int("hidden_dims_1", 16, 512, step=16),
    #     trial.suggest_int("hidden_dims_2", 16, 512, step=16),
    #     trial.suggest_int("hidden_dims_3", 16, 512, step=16),
    #     # trial.suggest_int("hidden_dims_4", 16, 512, step=16),
    #     # 512
    # ]
    # hidden_dims = [32, 64, 128, 256, 512]
    # layer_count = trial.suggest_int("layer_count", 1, 5)

    if __optuna_mode == "learning":
        return Namespace(
            learning_rate=trial.suggest_float("learning_rate", 0.00025, 0.001, log=True) * scale_factor,
            # weight_decay=trial.suggest_float("weight_decay", 1e-10, 1e-1, log=True),
            # latent_var=trial.suggest_float("latent_var", 0.1, 20),
            # train_batch_size=trial.suggest_int("train_batch_size", 20, 200, step=20),
            # scheduler_gamma=trial.suggest_float("scheduler_gamma", 0.9, 0.999, log=True), # scheduler is not implemented yet
            # hidden_dims=hidden_dims[:layer_count],
        )
    elif __optuna_mode == "loss":
        # x = trial.suggest_float("reg_weight", 1e-10, 1.0)
        # y = trial.suggest_float("kld_weight", 1e-10, 1.0)
        # y *= 1 - x
        # z = 1.0 - x - y
        # return Namespace(
        #     reg_weight=x + 2,
        #     kld_weight=y/2,
        #     # alpha=-trial.suggest_float("alpha", 1e-10, 20, log=True),
        #     beta=z,
        # )
        return Namespace(
            latent_var=trial.suggest_float("latent_var", 2, 100, log=True),
            # kernel_type=trial.suggest_categorical("kernel_type", ["rbf", "imq"]),
            # reg_weight=trial.suggest_float("reg_weight", 1e-10, 1.0),
        )


def get(name: str) -> Any:
    global __called, __namespace
    if (name in __optuna_names) and (__trial is not None):
        return getattr(namespace_optuna(), name)
    __called.add(name)
    return getattr(__namespace, name)

def delete_trial():
    global __trial
    __trial = None