import json
import os
import pickle as pkl
import random
import warnings
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import timedelta

from dataset.dataset import SaladsDataset
from ddpm.ddpm_multinomial import Diffusion
from denoisers.ConditionalUnetDenoiser import ConditionalUnetDenoiser
from denoisers.ConditionalUnetMatrixDenoiser import ConditionalUnetMatrixDenoiser
from denoisers.ConditionalUnetGraphDenoiser import ConditionalUnetGraphDenoiser
from denoisers.ConditionalUnetHeteroGraphDenoiser import ConditionalUnetHeteroGraphDenoiser
from utils.initialization import initialize
from utils.pm_utils import discover_dk_process, remove_duplicates_dataset, pad_to_multiple_of_n, conformance_measure
from utils.graph_utils import get_process_model_reachability_graph_transition_matrix, \
    get_process_model_petri_net_transition_matrix, get_process_model_reachability_graph_transition_multimatrix, \
    prepare_process_model_for_gnn, prepare_process_model_for_heterognn

warnings.filterwarnings("ignore")


def save_ckpt(model, opt, epoch, cfg, train_loss, test_loss, best=False):
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    torch.save(ckpt, os.path.join(cfg.summary_path, 'last.ckpt'))
    if best:
        torch.save(ckpt, os.path.join(cfg.summary_path, 'best.ckpt'))


def evaluate(diffuser, denoiser, test_loader, transition_matrix,
             process_model, init_marking, final_marking, cfg, summary, epoch, logger):
    denoiser.eval()
    total_loss = 0.0
    sequence_loss = 0.0
    matrix_loss = 0.0
    results_accumulator = {'x': [], 'y': [], 'x_hat': []}
    l = len(test_loader)
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.permute(0, 2, 1).to(cfg.device).float()
            y = y.permute(0, 2, 1).to(cfg.device).float()
            x_hat, matrix_hat, loss, seq_loss, mat_loss = \
                diffuser.sample_with_matrix(denoiser, y.shape[0], cfg.num_classes, denoiser.max_input_dim,
                                            transition_matrix.shape[-1], transition_matrix, x, y,
                                            cfg.predict_on)
            results_accumulator['x'].append(x)
            results_accumulator['y'].append(y)
            results_accumulator['x_hat'].append(x_hat.permute(0, 2, 1))
            total_loss += loss
            sequence_loss += seq_loss
            matrix_loss += mat_loss
            summary.add_scalar("test_loss", loss, global_step=epoch * l + i)
            summary.add_scalar("test_sequence_loss", seq_loss, global_step=epoch * l + i)
            summary.add_scalar("test_matrix_loss", mat_loss, global_step=epoch * l + i)

        x_argmax = torch.argmax(torch.cat(results_accumulator['x'], dim=0), dim=1).to('cpu')
        y_cat = torch.cat(results_accumulator['y'], dim=0)
        x_hat_logit = torch.cat(results_accumulator['x_hat'], dim=0)

        x_argmax_flat = x_argmax.reshape(-1).to('cpu')
        x_hat_flat = x_hat_logit.reshape(-1, cfg.num_classes).to('cpu')
        x_hat_prob_flat = torch.softmax(x_hat_flat, dim=1).to('cpu')
        x_hat_argmax_flat = torch.argmax(x_hat_prob_flat, dim=1).to('cpu')
        x_hat_prob = torch.softmax(x_hat_logit, dim=2).to('cpu')
        x_hat_argmax = torch.argmax(x_hat_prob, dim=2)  # recovered deterministic traces tensor

        # alignment = np.mean(
        #    conformance_measure(x_hat_argmax, process_model, init_marking, final_marking, cfg.activity_names,
        #                        limit=1000, remove_duplicates=True, approximate=False)
        # )
        alignment = 0

        try:
            auc = roc_auc_score(x_argmax_flat, x_hat_prob_flat, multi_class='ovr', average='macro')
        except ValueError:
            logger.info("excuse me wtf")
            auc = -1
            with open(os.path.join(cfg.summary_path, f"test_epoch_{epoch}_auc_error.pkl"), "wb") as f:
                pkl.dump({"x_argmax_flat": x_argmax_flat, "x_hat_prob_flat": x_hat_prob_flat}, f)

        w2 = np.mean([wasserstein_distance(xi, xhi) for xi, xhi in zip(x_argmax, x_hat_argmax)])
        accuracy = accuracy_score(x_argmax_flat, x_hat_argmax_flat)
        precision = precision_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
        recall = recall_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
        f1 = f1_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
        average_loss = total_loss / l
        average_first_loss = sequence_loss / l
        average_second_loss = matrix_loss / l
        with open(os.path.join(cfg.summary_path, f"epoch_{epoch}_test.pkl"), "wb") as f:
            pkl.dump(results_accumulator, f)

        summary.add_scalar("test_w2", w2, global_step=epoch * l)
        summary.add_scalar("test_accuracy", accuracy, global_step=epoch * l)
        summary.add_scalar("test_recall", recall, global_step=epoch * l)
        summary.add_scalar("test_precision", precision, global_step=epoch * l)
        summary.add_scalar("test_f1", f1, global_step=epoch * l)
        summary.add_scalar("test_auc", auc, global_step=epoch * l)
        summary.add_scalar("test_alpha", denoiser.alpha, global_step=epoch * l)
        summary.add_scalar("test_alignment", alignment, global_step=epoch * l)
        denoiser.train()
    return average_loss, accuracy, recall, precision, f1, auc, w2, average_first_loss, average_second_loss, \
        denoiser.alpha, alignment


def train(diffuser, denoiser, optimizer, train_loader, test_loader, transition_matrix,
          process_model, init_marking, final_marking, cfg, summary, logger):
    train_losses, test_losses, test_dist, test_acc, test_precision, test_recall, test_f1, test_auc = \
        [], [], [], [], [], [], [], []
    train_dist, train_acc, train_precision, train_recall, train_f1, train_auc = [], [], [], [], [], []
    train_seq_loss, train_matrix_loss, test_seq_loss, test_matrix_loss = [], [], [], []
    train_alpha, test_alpha = [], []
    train_alignment, test_alignment = [], []
    l = len(train_loader)
    transition_matrix = transition_matrix.unsqueeze(0)
    best_loss = float('inf')
    denoiser.train()
    for epoch in tqdm(range(cfg.num_epochs)):
        l_matrix = 0  # how many times matrix loss was calculated because it's dropped out sometimes
        epoch_loss = 0.0
        epoch_first_loss = 0.0
        epoch_second_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.permute(0, 2, 1).to(cfg.device).float()
            y = y.permute(0, 2, 1).to(cfg.device).float()
            t = diffuser.sample_timesteps(x.shape[0]).to(cfg.device)
            drop_matrix = False
            x_t, eps = diffuser.noise_data(x, t)  # each item in batch gets different level of noise based on timestep
            if np.random.random() < cfg.conditional_dropout:
                y = None
            if np.random.random() < cfg.matrix_dropout:
                drop_matrix = True
            output, matrix_hat, loss, seq_loss, mat_loss = denoiser(x_t, t, x, transition_matrix, y, drop_matrix)
            loss.backward()
            optimizer.step()

            summary.add_scalar("train_loss", loss.item(), global_step=epoch * l + i)
            epoch_loss += loss.item()
            epoch_first_loss += seq_loss
            epoch_second_loss += mat_loss
            summary.add_scalar("train_sequence_loss", seq_loss, global_step=epoch * l + i)
            if mat_loss != 0:
                l_matrix += 1
                summary.add_scalar("train_matrix_loss", mat_loss, global_step=epoch * l + i)
        train_losses.append(epoch_loss / l)
        train_seq_loss.append(epoch_first_loss / l)
        train_matrix_loss.append(epoch_second_loss / max(l_matrix, 1))
        train_alpha.append(denoiser.alpha)

        if epoch % cfg.test_every == 0:
            logger.info("testing epoch")
            if cfg.eval_train:
                denoiser.eval()
                with (torch.no_grad()):
                    sample_index = random.choice(range(len(train_loader)))
                    for i, batch in enumerate(train_loader):
                        if i == sample_index:
                            x, y = batch
                            break
                    x = x.permute(0, 2, 1).to(cfg.device).float()
                    y = y.permute(0, 2, 1).to(cfg.device).float()
                    output, matrix_hat, loss, seq_loss, mat_loss = \
                        diffuser.sample_with_matrix(denoiser, y.shape[0], cfg.num_classes, denoiser.max_input_dim,
                                                    transition_matrix.shape[-1], transition_matrix, x, y,
                                                    cfg.predict_on)
                    x_hat = output.permute(0, 2, 1)
                    x_argmax = torch.argmax(x, dim=1).to('cpu')
                    x_argmax_flat = torch.argmax(x, dim=1).reshape(-1).to('cpu')
                    x_hat_flat = x_hat.reshape(-1, cfg.num_classes).to('cpu')
                    x_hat_prob_flat = torch.softmax(x_hat_flat, dim=1).to('cpu')
                    x_hat_argmax_flat = torch.argmax(x_hat_prob_flat, dim=1).to('cpu')
                    x_hat_prob = torch.softmax(x_hat, dim=2).to('cpu')
                    x_hat_argmax = torch.argmax(x_hat_prob, dim=2)
                    # train_epoch_alignment = np.mean(
                    #    conformance_measure(x_hat_argmax, process_model, init_marking, final_marking,
                    #                        cfg.activity_names, limit=1000, remove_duplicates=True, approximate=False)
                    # )
                    train_epoch_alignment = 0
                    try:
                        auc = roc_auc_score(x_argmax_flat, x_hat_prob_flat, multi_class='ovr', average='micro')
                    except ValueError:
                        logger.info("excuse me wtf")
                        auc = -1
                        with open(os.path.join(cfg.summary_path, f"train_epoch_{epoch}_auc_error.pkl"), "wb") as f:
                            pkl.dump({"x_argmax_flat": x_argmax_flat, "x_hat_prob_flat": x_hat_prob_flat}, f)
                    w2 = np.mean([wasserstein_distance(xi, xhi) for xi, xhi in zip(x_argmax, x_hat_argmax)])
                    accuracy = accuracy_score(x_argmax_flat, x_hat_argmax_flat)
                    precision = precision_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
                    recall = recall_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
                    f1 = f1_score(x_argmax_flat, x_hat_argmax_flat, average='macro', zero_division=0)
                    with open(os.path.join(cfg.summary_path, f"epoch_{epoch}_train.pkl"), "wb") as f:
                        pkl.dump({"x": x, y: "y", "x_hat": x_hat}, f)
                    train_acc.append(accuracy)
                    train_recall.append(recall)
                    train_precision.append(precision)
                    train_f1.append(f1)
                    train_auc.append(auc)
                    train_dist.append(w2)
                    train_alignment.append(train_epoch_alignment)
                    summary.add_scalar("train_w2", w2, global_step=epoch * l)
                    summary.add_scalar("train_accuracy", accuracy, global_step=epoch * l)
                    summary.add_scalar("train_recall", recall, global_step=epoch * l)
                    summary.add_scalar("train_precision", precision, global_step=epoch * l)
                    summary.add_scalar("train_f1", f1, global_step=epoch * l)
                    summary.add_scalar("train_auc", auc, global_step=epoch * l)
                    summary.add_scalar("train_alpha", denoiser.alpha, global_step=epoch * l)
                    summary.add_scalar("train_alignment", train_epoch_alignment, global_step=epoch * l)
                denoiser.train()

            test_epoch_loss, test_epoch_acc, test_epoch_recall, test_epoch_precision, test_epoch_f1, test_epoch_auc, \
                test_epoch_dist, test_epoch_seq_loss, test_epoch_mat_loss, test_alpha_clamp, test_epoch_alignment = \
                evaluate(diffuser, denoiser, test_loader, transition_matrix,
                         process_model, init_marking, final_marking, cfg, summary, epoch, logger)
            test_dist.append(test_epoch_dist)
            test_losses.append(test_epoch_loss)
            test_acc.append(test_epoch_acc)
            test_recall.append(test_epoch_recall)
            test_precision.append(test_epoch_precision)
            test_f1.append(test_epoch_f1)
            test_auc.append(test_epoch_auc)
            test_seq_loss.append(test_epoch_seq_loss)
            test_matrix_loss.append(test_epoch_mat_loss)
            test_alpha.append(test_alpha_clamp)
            test_alignment.append(test_epoch_alignment)
            logger.info("saving model")
            save_ckpt(denoiser, optimizer, epoch, cfg, train_losses[-1], test_losses[-1],
                      test_epoch_loss < best_loss)
            best_loss = test_epoch_loss if test_epoch_loss < best_loss else best_loss

    return (train_losses, test_losses, test_dist, test_acc, test_precision, test_recall, test_f1, test_auc,
            train_acc, train_recall, train_precision, train_f1, train_auc, train_dist, train_seq_loss,
            train_matrix_loss, test_seq_loss, test_matrix_loss, train_alpha, test_alpha,
            train_alignment, test_alignment)


def main():
    args, cfg, dataset, logger = initialize()
    logger.info("loading dataset")
    salads_dataset = SaladsDataset(dataset['target'], dataset['stochastic'])
    train_dataset, test_dataset = train_test_split(salads_dataset, train_size=cfg.train_percent, shuffle=True,
                                                   random_state=cfg.seed)
    logger.info(f"train size: {len(train_dataset)} test size: {len(test_dataset)}")
    # random initialization instead of None for compatibility, isn't used in any way if enable_matrix is false
    rg_transition_matrix = torch.randn((cfg.num_classes, 2, 2)).to(cfg.device)
    logger.info("discovering process")
    dk_process_model, dk_init_marking, dk_final_marking = discover_dk_process(train_dataset, cfg,
                                                                              preprocess=remove_duplicates_dataset)
    if cfg.enable_matrix:
        logger.info("creating flow matrix")
        if cfg.matrix_type == "pm":
            rg_nx, rg_transition_matrix = get_process_model_petri_net_transition_matrix(dk_process_model,
                                                                                        dk_init_marking,
                                                                                        dk_final_marking)
            rg_transition_matrix = torch.tensor(rg_transition_matrix, device=cfg.device).unsqueeze(0).float()
        elif cfg.matrix_type == "rg":
            rg_nx, rg_transition_matrix = get_process_model_reachability_graph_transition_multimatrix(dk_process_model,
                                                                                                      dk_init_marking)
            rg_transition_matrix = torch.tensor(rg_transition_matrix, device=cfg.device).float()
        rg_transition_matrix = pad_to_multiple_of_n(rg_transition_matrix)
    if cfg.enable_gnn:
        logger.info("creating process graph")
        if cfg.hetero:
            pm_nx_data = prepare_process_model_for_heterognn(dk_process_model,
                                                             dk_init_marking, dk_final_marking, cfg).to(cfg.device)
        else:
            pm_nx_data = prepare_process_model_for_gnn(dk_process_model,
                                                       dk_init_marking, dk_final_marking, cfg).to(cfg.device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    diffuser = Diffusion(noise_steps=cfg.num_timesteps, device=cfg.device)

    if cfg.enable_matrix:
        denoiser = ConditionalUnetMatrixDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                                 max_input_dim=salads_dataset.sequence_length,
                                                 transition_dim=rg_transition_matrix.shape[-1],
                                                 gamma=cfg.gamma,
                                                 matrix_out_channels=rg_transition_matrix.shape[0],
                                                 device=cfg.device).to(cfg.device).float()
    elif cfg.enable_gnn:
        if cfg.hetero:
            denoiser = ConditionalUnetHeteroGraphDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                                          max_input_dim=salads_dataset.sequence_length,
                                                          graph_data=pm_nx_data, embedding_dim=128,
                                                          hidden_dim=128, pooling=cfg.gnn_pooling,
                                                          device=cfg.device).to(cfg.device).float()
        else:
            denoiser = ConditionalUnetGraphDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                                    max_input_dim=salads_dataset.sequence_length,
                                                    num_nodes=pm_nx_data.num_nodes,
                                                    graph_data=pm_nx_data,
                                                    embedding_dim=128, hidden_dim=128,
                                                    pooling=cfg.gnn_pooling, device=cfg.device).to(cfg.device).float()
    else:
        denoiser = ConditionalUnetDenoiser(in_ch=cfg.num_classes, out_ch=cfg.num_classes,
                                           max_input_dim=salads_dataset.sequence_length,
                                           device=cfg.device).to(cfg.device).float()
    if cfg.parallelize:
        denoiser = nn.DataParallel(denoiser, device_ids=[0, 1])

    optimizer = AdamW(denoiser.parameters(), cfg.learning_rate)
    summary = SummaryWriter(cfg.summary_path)

    start_time = time.perf_counter()

    (train_losses, test_losses, test_dist, test_acc, test_precision, tests_recall, test_f1, test_auc, train_acc,
     train_recall, train_precision, train_f1, train_auc, train_dist, train_seq_loss, train_mat_loss, test_seq_loss,
     test_mat_loss, train_alpha, test_alpha, train_alignment, test_alignment) = \
        train(diffuser, denoiser, optimizer, train_loader, test_loader, rg_transition_matrix,
              dk_process_model, dk_init_marking, dk_final_marking, cfg, summary, logger)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    px.line(train_losses).write_html(os.path.join(cfg.summary_path, "train_loss.html"))
    px.line(test_losses).write_html(os.path.join(cfg.summary_path, "test_loss.html"))
    px.line(test_dist).write_html(os.path.join(cfg.summary_path, "test_dist.html"))
    px.line(test_acc).write_html(os.path.join(cfg.summary_path, "test_acc.html"))
    px.line(test_precision).write_html(os.path.join(cfg.summary_path, "test_precision.html"))
    px.line(tests_recall).write_html(os.path.join(cfg.summary_path, "test_recall.html"))
    px.line(test_f1).write_html(os.path.join(cfg.summary_path, "test_f1.html"))
    px.line(test_auc).write_html(os.path.join(cfg.summary_path, "test_auc.html"))
    px.line(train_acc).write_html(os.path.join(cfg.summary_path, "train_acc.html"))
    px.line(train_recall).write_html(os.path.join(cfg.summary_path, "train_recall.html"))
    px.line(train_precision).write_html(os.path.join(cfg.summary_path, "train_precision.html"))
    px.line(train_f1).write_html(os.path.join(cfg.summary_path, "train_f1.html"))
    px.line(train_auc).write_html(os.path.join(cfg.summary_path, "train_auc.html"))
    px.line(train_dist).write_html(os.path.join(cfg.summary_path, "train_dist.html"))
    px.line(train_seq_loss).write_html(os.path.join(cfg.summary_path, "train_seq_loss.html"))
    px.line(train_mat_loss).write_html(os.path.join(cfg.summary_path, "train_mat_loss.html"))
    px.line(test_seq_loss).write_html(os.path.join(cfg.summary_path, "test_seq_loss.html"))
    px.line(test_mat_loss).write_html(os.path.join(cfg.summary_path, "test_mat_loss.html"))
    # px.line(train_alpha).write_html(os.path.join(cfg.summary_path, "train_alpha.html"))
    # px.line(test_alpha).write_html(os.path.join(cfg.summary_path, "test_alpha.html"))
    px.line(train_alignment).write_html(os.path.join(cfg.summary_path, "train_alignment.html"))
    px.line(test_alignment).write_html(os.path.join(cfg.summary_path, "test_alignment.html"))

    final_results = {
        "elapsed_time": str(timedelta(seconds=elapsed)),
        "train":
            {
                "loss": train_losses[-1],
                "acc": train_acc[-1],
                "precision": train_precision[-1],
                "recall": train_recall[-1],
                "f1": train_f1[-1],
                "auc": train_auc[-1],
                "dist": train_dist[-1],
                "alignment": train_alignment[-1],
            },
        "test":
            {
                "loss": test_losses[-1],
                "acc": test_acc[-1],
                "precision": test_precision[-1],
                "recall": tests_recall[-1],
                "f1": test_f1[-1],
                "auc": test_auc[-1],
                "dist": test_dist[-1],
                "alignment": test_alignment[-1],
            }
    }
    with open(os.path.join(cfg.summary_path, "final_results.json"), "w") as f:
        json.dump(final_results, f)


if __name__ == "__main__":
    main()
