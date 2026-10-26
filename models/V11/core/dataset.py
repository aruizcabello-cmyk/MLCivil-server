import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .config import *

BOLT_TO_IDX = {v: i for i, v in enumerate(CATALOGO_BOLT)}


def transition_indices_from_counts(na, nc):
    out = np.full((len(na), N_Y), P_TRANSITION_UNUSED_IDX, np.int64)
    for r, (a, c) in enumerate(zip(np.asarray(na, int), np.asarray(nc, int))):
        for j, x in enumerate(secuencia_transiciones(a, c)):
            out[r, j] = P_TRANSITION_TO_IDX[x]
    return out


def transition_indices_from_columns(df):
    out = np.zeros((len(df), N_Y), np.int64)
    for j, c in enumerate(P_TRANSITION_COLUMNS):
        out[:, j] = [P_TRANSITION_TO_IDX[x] for x in df[c].fillna("UNUSED").astype(str)]
    return out


def validate_transition_labels(df):
    if not all(c in df for c in P_TRANSITION_COLUMNS):
        return None
    return pd.Series(
        np.all(
            transition_indices_from_columns(df)
            == transition_indices_from_counts(df.n_alma, df.n_cantonera),
            1,
        ),
        index=df.index,
    )


def validate_connection_targets(df):
    p = df[P_COLUMNS].fillna(0).to_numpy(float) > 0
    k = df.n_alma.astype(int).to_numpy() + df.n_cantonera.astype(int).to_numpy() + 1
    m = np.arange(N_Y)[None, :] < k[:, None]
    return pd.DataFrame({"valid": np.all(p == m, 1)}, index=df.index)


class Preprocessor:
    def fit(self, df):
        self.cat_encoders = {
            c: LabelEncoder().fit(df[c].astype(str)) for c in CATEGORICAL_FEATURES
        }
        self.num_scaler = StandardScaler().fit(df[NUMERIC_FEATURES])
        self.reg_always_scaler = StandardScaler().fit(df[REGRESSION_ALWAYS])
        self.standard_encoder = LabelEncoder().fit(df.standard.astype(str))
        x = df[MASKED_COLUMNS].fillna(0).to_numpy(float)
        self.masked_mean = np.zeros(x.shape[1])
        self.masked_std = np.ones(x.shape[1])
        frac = np.zeros(x.shape[1])
        for j in range(x.shape[1]):
            a = x[:, j] > 0
            frac[j] = a.mean()
            if a.sum() > 1:
                self.masked_mean[j] = x[a, j].mean()
                self.masked_std[j] = max(x[a, j].std(), 1e-6)
        self.presence_pos_weight = np.maximum(
            (1 - frac) / np.maximum(frac, 1e-6), 1
        ).astype(np.float32)

        def w(col, n):
            v = df[col].astype(int).to_numpy()
            c = np.array([max((v == i).sum(), 1) for i in range(n)])
            q = len(v) / (n * c)
            return (q / q.mean()).astype(np.float32)

        self.count_class_weights = {
            "n_alma": w("n_alma", N_ALMA_CLASSES),
            "n_cantonera": w("n_cantonera", N_CANTONERA_CLASSES),
        }
        return self

    @property
    def cat_cardinalities(self):
        return [len(self.cat_encoders[c].classes_) for c in CATEGORICAL_FEATURES]

    @property
    def n_standard_classes(self):
        return len(self.standard_encoder.classes_)

    def transform_features(self, df):
        return np.stack(
            [
                self.cat_encoders[c].transform(df[c].astype(str))
                for c in CATEGORICAL_FEATURES
            ],
            1,
        ).astype(np.int64), self.num_scaler.transform(df[NUMERIC_FEATURES]).astype(
            np.float32
        )

    def transform_targets(self, df):
        real = df[MASKED_COLUMNS].fillna(0).to_numpy(np.float32)
        na = df.n_alma.astype(int).to_numpy()
        nc = df.n_cantonera.astype(int).to_numpy()
        tr = (
            transition_indices_from_columns(df)
            if all(c in df for c in P_TRANSITION_COLUMNS)
            else transition_indices_from_counts(na, nc)
        )
        return {
            "standard_idx": self.standard_encoder.transform(
                df.standard.astype(str)
            ).astype(np.int64),
            "bolt_idx": df.Bolt.astype(str)
            .str.replace("M", "", regex=False)
            .astype(int)
            .map(BOLT_TO_IDX)
            .to_numpy(np.int64),
            "slip_critical": df.SlipCritical.astype(str)
            .str.upper()
            .isin(["YES", "TRUE", "1"])
            .to_numpy(np.float32),
            "n_alma_idx": na.astype(np.int64),
            "n_cantonera_idx": nc.astype(np.int64),
            "reg_always": self.reg_always_scaler.transform(
                df[REGRESSION_ALWAYS]
            ).astype(np.float32),
            "masked_value": ((real - self.masked_mean) / self.masked_std).astype(
                np.float32
            ),
            "masked_mask": (real > 0).astype(np.float32),
            "p_transition_idx": tr,
            "beam_d_mm": df.BEAM_d_mm.to_numpy(np.float32),
        }

    def inverse_reg_always(self, x):
        return self.reg_always_scaler.inverse_transform(x)

    def inverse_masked(self, x):
        return x * self.masked_std + self.masked_mean


class ConnectionDataset(Dataset):
    def __init__(self, df, pre):
        self.xc, self.xn = pre.transform_features(df)
        self.t = pre.transform_targets(df)

    def __len__(self):
        return len(self.xc)

    def __getitem__(self, i):
        out = {"x_cat": torch.tensor(self.xc[i]), "x_num": torch.tensor(self.xn[i])}
        long = {
            "standard_idx",
            "bolt_idx",
            "n_alma_idx",
            "n_cantonera_idx",
            "p_transition_idx",
        }
        out.update(
            {
                k: torch.tensor(v[i], dtype=torch.long if k in long else torch.float32)
                for k, v in self.t.items()
            }
        )
        return out
