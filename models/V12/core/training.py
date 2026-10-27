"""Entrenamiento e inferencia (v3 / v4).

Cambios respecto a la version anterior:
  * fit(select_metric=...) permite elegir la mejor epoca por una metrica de p
    (p_within_10 / p_mae_mm) en validacion realista, en vez de la perdida total.
  * El log muestra p_ce, topologia conjunta y topology_exact cuando existen.
"""

import copy
import numpy as np
import torch

from .config import N_Y


def split_batch(batch, device):
    x_cat = batch["x_cat"].to(device)
    x_num = batch["x_num"].to(device)
    targets = {
        key: value.to(device)
        for key, value in batch.items()
        if key not in ("x_cat", "x_num")
    }
    return x_cat, x_num, targets


def teacher_forcing_schedule(
    epoch,
    epochs,
    start=1.0,
    end=0.15,
    warmup_fraction=0.15,
):
    warmup_epochs = max(1, int(epochs * warmup_fraction))
    if epoch <= warmup_epochs:
        return float(start)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return float(start + (end - start) * progress)


SELECT_METRICS = ("val_loss", "p_within_10", "p_mae_mm", "topology_exact", "p_and_topology")


def selection_score(select_metric, val_loss, val_parts):
    """Puntuacion a MAXIMIZAR para elegir la mejor epoca."""
    if select_metric == "val_loss":
        return -val_loss
    if select_metric == "p_and_topology":
        # p dentro de +-10 mm (validacion realista) + topologia exacta: ambos en [0, 1].
        return val_parts["p_within_10"] + val_parts["topology_exact"]
    if select_metric not in val_parts:
        raise KeyError(
            f"La metrica '{select_metric}' no esta en las partes de validacion: "
            f"{sorted(val_parts)}"
        )
    value = val_parts[select_metric]
    return -value if select_metric == "p_mae_mm" else value


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scheduler=None,
    use_true_topology=False,
    teacher_forcing_ratio=0.0,
    clip_norm=5.0,
):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    parts_sum = {}
    n_batches = 0

    for batch in loader:
        x_cat, x_num, targets = split_batch(batch, device)

        if use_true_topology:
            transitions = targets["p_transition_idx"]
            active_count = (
                targets["n_alma_idx"]
                + targets["n_cantonera_idx"]
                + 1
            )
        else:
            transitions = None
            active_count = None

        with torch.set_grad_enabled(train):
            outputs = model(
                x_cat,
                x_num,
                p_transition_idx=transitions,
                p_teacher=(
                    targets["masked_value"][:, :N_Y]
                    if train
                    else None
                ),
                active_count=active_count,
                teacher_forcing_ratio=(
                    teacher_forcing_ratio if train else 0.0
                ),
            )
            loss, parts = criterion(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        for key, value in parts.items():
            parts_sum[key] = parts_sum.get(key, 0.0) + value

    if n_batches == 0:
        raise RuntimeError("El DataLoader no contiene batches.")

    return (
        total_loss / n_batches,
        {key: value / n_batches for key, value in parts_sum.items()},
    )


def format_epoch_log(
    epoch,
    epochs,
    train_loss,
    val_loss,
    train_parts,
    val_parts,
    teacher_forcing_ratio,
    learning_rate,
    is_best,
    best_val_loss,
    select_metric="val_loss",
):
    marker = " [MEJOR]" if is_best else ""
    lines = [
        (
            f"Epoca {epoch:03d}/{epochs:03d}"
            f" | LR {learning_rate:.2e}"
            f" | TF {100 * teacher_forcing_ratio:5.1f}%"
            f" | Train {train_loss:8.4f}"
            f" | Val {val_loss:8.4f}"
            f" | Mejor {best_val_loss:8.4f}{marker}"
            f" | sel={select_metric}"
        ),
        (
            "  p       "
            f"| loss train/val "
            f"{train_parts.get('p_value', float('nan')):6.3f}/"
            f"{val_parts.get('p_value', float('nan')):6.3f}"
            f" | CE {train_parts.get('p_ce', float('nan')):6.3f}/"
            f"{val_parts.get('p_ce', float('nan')):6.3f}"
            f" | acumulada "
            f"{train_parts.get('p_cumulative', float('nan')):6.3f}/"
            f"{val_parts.get('p_cumulative', float('nan')):6.3f}"
            f" | MAE {val_parts.get('p_mae_mm', float('nan')):6.2f} mm"
            f" | <=10 mm {100 * val_parts.get('p_within_10', 0.0):5.1f}%"
        ),
        (
            "  geometria"
            f" | MAE acumulado "
            f"{val_parts.get('cumulative_mae_mm', float('nan')):6.2f} mm"
            f" | reg {val_parts.get('reg_always', float('nan')):6.3f}"
            f" | rigidizadores "
            f"{val_parts.get('stiffener_value', float('nan')):6.3f}"
            f" | presencia "
            f"{val_parts.get('masked_presence', float('nan')):6.3f}"
        ),
        (
            "  topologia"
            f" | n_alma {val_parts.get('n_alma', float('nan')):6.3f}"
            f" | n_cantonera "
            f"{val_parts.get('n_cantonera', float('nan')):6.3f}"
            f" | conjunta {val_parts.get('topology', float('nan')):6.3f}"
            f" | exacta {100 * val_parts.get('topology_exact', float('nan')):5.1f}%"
        ),
        (
            "  clases  "
            f"| standard {val_parts.get('standard', float('nan')):6.3f}"
            f" | bolt {val_parts.get('bolt', float('nan')):6.3f}"
            f" | slip {val_parts.get('slip_critical', float('nan')):6.3f}"
        ),
    ]
    return "\n".join(lines)


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    epochs=100,
    lr=2e-3,
    weight_decay=1e-4,
    checkpoint_path=None,
    log_every=5,
    tf_start=1.0,
    tf_end=0.15,
    tf_warmup_fraction=0.15,
    clip_norm=5.0,
    select_metric="val_loss",
    min_select_epoch=1,
    verbose=True,
):
    if select_metric not in SELECT_METRICS:
        raise ValueError(f"select_metric debe ser uno de {SELECT_METRICS}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=epochs * len(train_loader),
        pct_start=0.15,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_parts": [],
        "val_parts": [],
        "teacher_forcing_ratio": [],
        "learning_rate": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "select_metric": select_metric,
        "best_score": None,
    }

    best_state = None
    best_val_loss = float("inf")
    best_score = -float("inf")

    for epoch in range(1, epochs + 1):
        teacher_forcing_ratio = teacher_forcing_schedule(
            epoch,
            epochs,
            start=tf_start,
            end=tf_end,
            warmup_fraction=tf_warmup_fraction,
        )

        train_loss, train_parts = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            use_true_topology=True,
            teacher_forcing_ratio=teacher_forcing_ratio,
            clip_norm=clip_norm,
        )

        # Validacion realista para seleccionar el modelo de produccion.
        val_loss, val_parts = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scheduler=None,
            use_true_topology=False,
            teacher_forcing_ratio=0.0,
            clip_norm=clip_norm,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_parts"].append(train_parts)
        history["val_parts"].append(val_parts)
        history["teacher_forcing_ratio"].append(teacher_forcing_ratio)
        history["learning_rate"].append(current_lr)

        score = selection_score(select_metric, val_loss, val_parts)
        is_best = epoch >= min_select_epoch and score > best_score
        if is_best:
            best_score = score
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            history["best_val_loss"] = val_loss
            history["best_score"] = float(score)

            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "teacher_forcing_ratio": teacher_forcing_ratio,
                    },
                    checkpoint_path,
                )

        if verbose and (
            epoch == 1
            or epoch % log_every == 0
            or is_best
            or epoch == epochs
        ):
            print("=" * 132)
            print(
                format_epoch_log(
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                    train_parts,
                    val_parts,
                    teacher_forcing_ratio,
                    current_lr,
                    is_best,
                    best_val_loss,
                    select_metric,
                )
            )

    if best_state is None:
        raise RuntimeError("No se pudo guardar ningun estado del modelo.")

    model.load_state_dict(best_state)
    if verbose:
        print("\n" + "=" * 132)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 132)
        print(f"Criterio de seleccion : {select_metric}")
        print(f"Mejor epoca           : {history['best_epoch']}")
        print(f"val_loss en esa epoca : {history['best_val_loss']:.6f}")
        if checkpoint_path is not None:
            print(f"Checkpoint            : {checkpoint_path}")
        print("Los pesos de la mejor epoca se han restaurado en model.")

    return history


@torch.no_grad()
def predict_loader(
    model,
    loader,
    device,
    use_true_transitions=False,
):
    model.eval()
    batches = []

    for batch in loader:
        x_cat, x_num, targets = split_batch(batch, device)

        if use_true_transitions:
            transitions = targets["p_transition_idx"]
            active_count = (
                targets["n_alma_idx"]
                + targets["n_cantonera_idx"]
                + 1
            )
        else:
            transitions = None
            active_count = None

        outputs = model(
            x_cat,
            x_num,
            p_transition_idx=transitions,
            active_count=active_count,
            teacher_forcing_ratio=0.0,
        )

        batches.append(
            {
                key: value.detach().cpu().numpy()
                for key, value in outputs.items()
            }
        )

    if not batches:
        raise RuntimeError("El DataLoader no produjo predicciones.")

    return {
        key: np.concatenate(
            [batch[key] for batch in batches],
            axis=0,
        )
        for key in batches[0]
    }
