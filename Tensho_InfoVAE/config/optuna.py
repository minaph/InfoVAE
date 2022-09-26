import optuna
import time
import logging
import sys

current_time = time.strftime("%Y%m%d-%H%M%S")

optuna.logging.disable_default_handler()
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
# study_name = "Tensho_InfoVAE_" + current_time  # Unique identifier of the study.
# storage_name = "sqlite:///Tensho_InfoVAE/optuna/{}.db".format(study_name)

def get_study(name: str):
    return optuna.create_study(
        study_name=name,
        storage=f"sqlite:///Tensho_InfoVAE/optuna/{name}.db",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        load_if_exists=True,
    )

def get_study_multi_obj(name: str):
    study_multi_obj = optuna.create_study(
        storage=f"sqlite:///Tensho_InfoVAE/optuna/{name}.db",
        study_name=name,
        directions=["minimize"] * 2,
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
    )
    return study_multi_obj


