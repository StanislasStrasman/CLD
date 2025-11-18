import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd
import math
import time
import os
import ot
import logging
from datetime import datetime
import pytz
from copy import deepcopy
from typing import Any, Iterable, List, final

from config import get_config
from CLD import CriticallyDampedLangevin
from model import DualDecoder
from data import make_training_data, normalize, unnormalize
from loss import loss_hsm
from ema import EMA
from train import train_one_epoch
from utils import create_logger


torch.set_default_dtype(torch.float32)


def run_cld(
    config: Any,
    dataloader_SGM,
    training_sample_position: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    result_folder: str,
) -> pd.DataFrame:

    base_cfg = deepcopy(config)
    a_values = base_cfg.a
    eps_values = base_cfg.epsilon

    all_dfs = []

    for a in a_values:
        logger.info(f"Run fr a = {a}")
        simulation_results_hsm = []

        for e in eps_values:
            run_cfg = _with_overrides(base_cfg, a=a, epsilon=e)
            is_cld = math.isclose(e, 0.0, abs_tol=1e-12)
            logger.info(f"Optimization for epsilon = {e}")
            sde = CriticallyDampedLangevin(run_cfg)

            for j in range(run_cfg.rep_size):
                net = DualDecoder(2 * run_cfg.d, run_cfg.mid_features, run_cfg.num_layers, is_cld=is_cld).to(run_cfg.device)
                opt = Adam(net.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.weight_decay)
                loss_fn = loss_hsm(net, sde, run_cfg.loss_eps)
                ema = EMA(net, decay=run_cfg.ema_decay)

                for epoch in range(1, run_cfg.n_epochs + 1):
                    avg_loss = train_one_epoch(loss_fn, net, dataloader_SGM, run_cfg, opt, ema, is_cld)

                    if epoch % run_cfg.eval_interval == 0 or epoch == run_cfg.n_epochs:
                        with torch.no_grad():
                            ema.apply_shadow()
                            init = sde.sample_final(run_cfg.sampling_size, run_cfg.d)
                            sample = sde.Euler_Maruyama_discr_sampler(init, net, run_cfg.num_steps, run_cfg.sampling_eps, is_cld=is_cld)
                            sample = unnormalize(sample, mean, std)
                            w2_sliced = ot.sliced.sliced_wasserstein_distance(
                                training_sample_position, sample, n_projections=2000, p=2
                            )
                            ema.restore()

                        row = {"a": a, "epsilon": e, "replication": j, "epoch": epoch, "w2": w2_sliced}
                        if j == 0 and epoch == run_cfg.n_epochs:
                            os.makedirs(os.path.join(result_folder, "sample"), exist_ok=True)
                            sample_filename = f"sample_eps_{e}_a_{a}.pt"
                            save_path = os.path.join(result_folder, "sample", sample_filename)
                            torch.save(sample.cpu(), save_path)
                            row["sample_file"] = sample_filename

                        simulation_results_hsm.append(row)
                        logger.info(f"[a={a} | epsilon={e} | rep={j+1} | epoch={epoch}] W2 = {w2_sliced:.4f}")

        # Per-epsilon summary for this 'a'
        df = pd.DataFrame(simulation_results_hsm)
        for e in eps_values:
            last_rows = df[df["epsilon"] == e].sort_values("epoch").groupby("replication").last()
            avg_w2 = last_rows["w2"].mean()
            logger.info(f"Average W2 for a={a}, epsilon={e}: {avg_w2:.4f}")

        logger.info("=================================================================")
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return final_df


def _with_overrides(cfg: Any, **overrides) -> Any:
    new = deepcopy(cfg)
    for k, v in overrides.items():
        setattr(new, k, v)
    return new


if __name__ == "__main__":

    config = get_config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    process_start_time = datetime.now(pytz.timezone("Europe/Paris"))
    result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '_' + config.dataset + '_train'
    
    create_logger(result_folder)
    logger = logging.getLogger('root')
    logger.info("Configuration:")
    for attr, value in vars(config).items():
        logger.info(f"{attr}: {value}")
    logger.info('=================================================================')

    if config.device.type == 'cpu':
        # Running on CPU: reduce workload for quicker tests/debugging
        config.d = 2
        config.n_samples = 500
        config.batch_size = 100
        config.n_epochs = 10
        config.sampling_size = 50
        config.num_steps = 2

    # ==== Data Preparation ====
    logging.info("Preparing data")
    training_sample_position = make_training_data(config)
    if config.dataset == 'diamond' or config.dataset == 'multimodal_swissroll':
        config.d = 2  # Ensure d is set to 2 for these datasets

    norm, mean, std = normalize(training_sample_position)
    training_sample_velocity = torch.randn(config.n_samples, config.d, device=training_sample_position.device) * (config.v0_var ** 0.5)
    training_sample = torch.cat([norm, training_sample_velocity], dim=1)
    dataloader_SGM = DataLoader(norm.to(config.device), batch_size=config.batch_size, shuffle=True)

    # ----- Save results -----
    save_dir = os.path.join(result_folder, "sample")
    os.makedirs(save_dir, exist_ok=True) 
    save_path = os.path.join(save_dir, "training_samples.pt")
    torch.save(training_sample_position, save_path)

    # ==== Training ====
    logging.info("Starting training")
    total_start = time.time()
    final_df = run_cld(config, dataloader_SGM, training_sample_position, mean, std, result_folder)
    final_df.to_csv(result_folder + "/simulation_df_hsm.csv", index=False)
    total_time = time.time() - total_start
    logging.info(f"Total training time: {total_time:.2f} seconds")
    logging.info("All Done!!!")
