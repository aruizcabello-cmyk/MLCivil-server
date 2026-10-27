"""Perdida multitarea para la ablacion autorregresiva v3."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_Y, N_CANTONERA_CLASSES


def per_column_mean(values, mask):
    mask = mask.float()
    n_active = mask.sum(dim=0)
    per_column = (values * mask).sum(dim=0) / n_active.clamp_min(1.0)
    available = n_active > 0
    if available.any():
        return per_column[available].mean()
    return values.sum() * 0.0


class ConnectionDesignLoss(nn.Module):
    def __init__(
        self,
        n_alma_weights,
        n_cantonera_weights,
        presence_pos_weight,
        p_mean_mm,
        p_std_mm,
        w_cls=1.0,
        w_counts=1.0,
        w_reg=1.0,
        w_p_value=2.0,
        w_p_cumulative=0.1,
        w_stiffener_value=1.0,
        w_masked_presence=0.5,
        cumulative_scale_mm=20.0,
        p_tolerance_mm=10.0,
    ):
        super().__init__()
        self.w_cls = float(w_cls)
        self.w_counts = float(w_counts)
        self.w_reg = float(w_reg)
        self.w_p_value = float(w_p_value)
        self.w_p_cumulative = float(w_p_cumulative)
        self.w_stiffener_value = float(w_stiffener_value)
        self.w_masked_presence = float(w_masked_presence)
        self.cumulative_scale_mm = float(cumulative_scale_mm)
        self.p_tolerance_mm = float(p_tolerance_mm)

        self.register_buffer(
            "n_alma_weights",
            torch.as_tensor(n_alma_weights, dtype=torch.float32),
        )
        self.register_buffer(
            "n_cantonera_weights",
            torch.as_tensor(n_cantonera_weights, dtype=torch.float32),
        )
        self.register_buffer(
            "presence_pos_weight",
            torch.as_tensor(presence_pos_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "p_mean_mm",
            torch.as_tensor(p_mean_mm, dtype=torch.float32).view(1, N_Y),
        )
        self.register_buffer(
            "p_std_mm",
            torch.as_tensor(p_std_mm, dtype=torch.float32).view(1, N_Y),
        )

        self.huber = nn.SmoothL1Loss(reduction="none", beta=1.0)

    def forward(self, outputs, targets):
        loss_standard = F.cross_entropy(
            outputs["standard_logits"],
            targets["standard_idx"],
        )
        loss_bolt = F.cross_entropy(
            outputs["bolt_logits"],
            targets["bolt_idx"],
        )
        loss_slip = F.binary_cross_entropy_with_logits(
            outputs["slip_critical_logit"],
            targets["slip_critical"],
        )
        loss_classification = loss_standard + loss_bolt + loss_slip

        loss_n_alma = F.cross_entropy(
            outputs["n_alma_logits"],
            targets["n_alma_idx"],
            weight=self.n_alma_weights,
        )
        loss_n_cantonera = F.cross_entropy(
            outputs["n_cantonera_logits"],
            targets["n_cantonera_idx"],
            weight=self.n_cantonera_weights,
        )
        loss_counts = loss_n_alma + loss_n_cantonera

        loss_reg_always = self.huber(
            outputs["reg_always"],
            targets["reg_always"],
        ).mean()

        mask = targets["masked_mask"]
        p_mask = mask[:, :N_Y]
        stiffener_mask = mask[:, N_Y:]

        p_pred = outputs["masked_value"][:, :N_Y]
        p_true = targets["masked_value"][:, :N_Y]
        stiffener_pred = outputs["masked_value"][:, N_Y:]
        stiffener_true = targets["masked_value"][:, N_Y:]

        loss_p_value = per_column_mean(
            self.huber(p_pred, p_true),
            p_mask,
        )
        loss_stiffener_value = per_column_mean(
            self.huber(stiffener_pred, stiffener_true),
            stiffener_mask,
        )

        p_pred_mm = p_pred * self.p_std_mm + self.p_mean_mm
        p_true_mm = p_true * self.p_std_mm + self.p_mean_mm
        cumulative_pred_mm = torch.cumsum(p_pred_mm, dim=1)
        cumulative_true_mm = torch.cumsum(p_true_mm, dim=1)

        loss_p_cumulative = per_column_mean(
            self.huber(
                cumulative_pred_mm / self.cumulative_scale_mm,
                cumulative_true_mm / self.cumulative_scale_mm,
            ),
            p_mask,
        )

        presence_values = F.binary_cross_entropy_with_logits(
            outputs["masked_presence_logit"],
            mask,
            pos_weight=self.presence_pos_weight,
            reduction="none",
        )
        loss_presence = presence_values.mean(dim=0).mean()

        with torch.no_grad():
            p_error_mm = torch.abs(p_pred_mm - p_true_mm)
            n_active = p_mask.sum().clamp_min(1.0)
            p_within_10 = (
                ((p_error_mm <= self.p_tolerance_mm).float() * p_mask).sum()
                / n_active
            )
            p_mae_mm = (p_error_mm * p_mask).sum() / n_active
            cumulative_mae_mm = (
                torch.abs(cumulative_pred_mm - cumulative_true_mm) * p_mask
            ).sum() / n_active

        with torch.no_grad():
            topology_exact = (
                (outputs["n_alma_logits"].argmax(dim=1) == targets["n_alma_idx"])
                & (outputs["n_cantonera_logits"].argmax(dim=1) == targets["n_cantonera_idx"])
            ).float().mean()

        total = (
            self.w_cls * loss_classification
            + self.w_counts * loss_counts
            + self.w_reg * loss_reg_always
            + self.w_p_value * loss_p_value
            + self.w_p_cumulative * loss_p_cumulative
            + self.w_stiffener_value * loss_stiffener_value
            + self.w_masked_presence * loss_presence
        )

        parts = {
            "standard": loss_standard.item(),
            "bolt": loss_bolt.item(),
            "slip_critical": loss_slip.item(),
            "n_alma": loss_n_alma.item(),
            "n_cantonera": loss_n_cantonera.item(),
            "reg_always": loss_reg_always.item(),
            "p_value": loss_p_value.item(),
            "p_cumulative": loss_p_cumulative.item(),
            "p_within_10": p_within_10.item(),
            "p_mae_mm": p_mae_mm.item(),
            "cumulative_mae_mm": cumulative_mae_mm.item(),
            "stiffener_value": loss_stiffener_value.item(),
            "masked_presence": loss_presence.item(),
            "topology_exact": topology_exact.item(),
        }
        return total, parts


# =============================================================================
# V4: perdida para la cabeza de p discretizada (+ topologia conjunta opcional)
# =============================================================================


class ConnectionDesignLossV4(ConnectionDesignLoss):
    """Hereda la perdida v3 (clasificacion, reg_always, rigidizadores y las
    metricas p_within_10 / p_mae_mm / cumulative_mae_mm) y:

    * anula los terminos de regresion de p y los sustituye por una cross-entropy
      ORDINAL sobre la rejilla de 5 mm, promediada por separacion activa
      (coincide con la metrica, a diferencia del promedio por columna);
    * aplica la perdida de presencia solo a los rigidizadores (la presencia de
      p no se usa en inferencia: mandan los conteos);
    * si joint_topology=True, sustituye la perdida de conteos por una unica
      cross-entropy sobre (n_alma, n_cantonera);

    label_scale_bins: escala (en bins) del suavizado Laplace de la etiqueta.
        1.0 -> la probabilidad cae a 1/e a 5 mm del valor real. Penaliza menos
        fallar por 1 bin que por 5. Con ~0.05 las etiquetas son casi duras.
    """

    def __init__(
        self,
        *args,
        n_p_bins,
        p_bin_step_mm=5.0,
        w_p_ce=2.0,
        label_scale_bins=1.0,
        joint_topology=False,
        **kwargs,
    ):
        original_w_counts = float(kwargs.get("w_counts", 1.0))
        original_w_presence = float(kwargs.get("w_masked_presence", 0.5))
        kwargs["w_p_value"] = 0.0
        kwargs["w_p_cumulative"] = 0.0
        kwargs["w_masked_presence"] = 0.0
        if joint_topology:
            kwargs["w_counts"] = 0.0
        super().__init__(*args, **kwargs)

        self.w_p_ce = float(w_p_ce)
        self.p_bin_step_mm = float(p_bin_step_mm)
        self.label_scale_bins = float(label_scale_bins)
        self.joint_topology = bool(joint_topology)
        self.w_topology = original_w_counts if joint_topology else 0.0
        self.w_presence_stiffeners = original_w_presence
        self.register_buffer("bin_index", torch.arange(int(n_p_bins), dtype=torch.float32))

    def forward(self, outputs, targets):
        total, parts = super().forward(outputs, targets)

        # --- p: cross-entropy ordinal sobre la rejilla ---
        p_logits = outputs["p_logits"]  # (B, N_Y, K)
        p_mask = targets["masked_mask"][:, :N_Y]
        p_true_mm = targets["masked_value"][:, :N_Y] * self.p_std_mm + self.p_mean_mm
        n_bins = p_logits.shape[-1]
        target_bin = (p_true_mm / self.p_bin_step_mm).round().clamp(0, n_bins - 1)

        distance = (self.bin_index.view(1, 1, -1) - target_bin.unsqueeze(-1)).abs()
        soft_target = torch.exp(-distance / max(self.label_scale_bins, 1e-3))
        soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True)

        ce = -(soft_target * F.log_softmax(p_logits, dim=-1)).sum(dim=-1)  # (B, N_Y)
        loss_p_ce = (ce * p_mask).sum() / p_mask.sum().clamp_min(1.0)
        total = total + self.w_p_ce * loss_p_ce
        parts["p_ce"] = loss_p_ce.item()

        # --- presencia: solo rigidizadores ---
        mask = targets["masked_mask"]
        presence_values = F.binary_cross_entropy_with_logits(
            outputs["masked_presence_logit"][:, N_Y:],
            mask[:, N_Y:],
            pos_weight=self.presence_pos_weight[N_Y:],
            reduction="none",
        )
        loss_presence = presence_values.mean(dim=0).mean()
        total = total + self.w_presence_stiffeners * loss_presence
        parts["masked_presence"] = loss_presence.item()

        # --- topologia ---
        if self.joint_topology:
            joint_target = (
                targets["n_alma_idx"] * N_CANTONERA_CLASSES + targets["n_cantonera_idx"]
            )
            loss_topology = F.cross_entropy(outputs["topology_logits"], joint_target)
            total = total + self.w_topology * loss_topology
            parts["topology"] = loss_topology.item()

        return total, parts
