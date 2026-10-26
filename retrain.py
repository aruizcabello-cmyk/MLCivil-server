import json, os, random, re, shutil, sys, time, traceback, zlib
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import config, retrain_manager as rm

SEED, EPOCHS, BATCH, LR, WD = 42, 200, 128, 2e-3, 1e-4
HID, NB, DROP, GRU = 192, 3, 0.10, 128
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bucket(u): return zlib.crc32(str(u).encode()) % 100   # <15 test, <30 val, resto train (estable)


def next_name(base="V11"):
    n = [int(m.group(1)) for p in config.MODELS_DIR.iterdir()
         if (m := re.fullmatch(re.escape(base) + r"\.(\d+)", p.name))]
    return f"{base}.{max(n, default=0) + 1}"


def main():
    n_raw = rm.n_rows()
    try:
        parent = config.MODEL_NAME
        pdir = config.MODELS_DIR / parent
        sys.path.insert(0, str(pdir))
        import core.config as C
        from core.dataset import Preprocessor, ConnectionDataset, validate_connection_targets, validate_transition_labels
        from core.model import ConnectionDesignNetV4, n_bins_from_train
        from core.losses import ConnectionDesignLossV4
        from core.training import fit, run_epoch

        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        df = pd.read_parquet(rm.MASTER)
        n_raw = len(df)

        num = list(dict.fromkeys(C.NUMERIC_FEATURES + C.REGRESSION_ALWAYS + C.MASKED_COLUMNS + ["n_alma", "n_cantonera"]))
        for c in num: df[c] = pd.to_numeric(df[c], errors="coerce")
        df[C.MASKED_COLUMNS] = df[C.MASKED_COLUMNS].fillna(0.0)
        df = df.dropna(subset=C.CATEGORICAL_FEATURES + C.NUMERIC_FEATURES + C.REGRESSION_ALWAYS +
                       ["standard", "Bolt", "SlipCritical", "n_alma", "n_cantonera"])
        b = df.Bolt.astype(str).str.replace("M", "", regex=False).astype(float).astype(int)
        df = df[b.isin(C.CATALOGO_BOLT) & (df.n_alma < C.N_ALMA_CLASSES) & (df.n_cantonera < C.N_CANTONERA_CLASSES)]
        df = df[validate_connection_targets(df)["valid"]]
        lv = validate_transition_labels(df)
        if lv is not None: df = df[lv]
        df = df.reset_index(drop=True)

        bk = df["UNICODE"].map(bucket)
        te, va, tr = df[bk < 15], df[(bk >= 15) & (bk < 30)], df[bk >= 30]
        cats = C.CATEGORICAL_FEATURES + ["standard"]

        def seen(d, t):
            m = pd.Series(True, index=d.index)
            for c in cats: m &= d[c].astype(str).isin(set(t[c].astype(str)))
            return m
        for _ in range(2):
            mv, mt = seen(va, tr), seen(te, tr)
            tr = pd.concat([tr, va[~mv], te[~mt]]); va, te = va[mv], te[mt]
        tr, va, te = [d.reset_index(drop=True) for d in (tr, va, te)]
        print(f"train {len(tr)} val {len(va)} test {len(te)}", flush=True)

        def loss_for(pre, nb):
            return ConnectionDesignLossV4(
                n_alma_weights=np.ones(C.N_ALMA_CLASSES, np.float32),
                n_cantonera_weights=np.ones(C.N_CANTONERA_CLASSES, np.float32),
                presence_pos_weight=pre.presence_pos_weight, p_mean_mm=pre.masked_mean[:C.N_Y],
                p_std_mm=pre.masked_std[:C.N_Y], w_cls=1.0, w_counts=1.0, w_reg=1.0, w_p_value=2.0,
                w_p_cumulative=0.1, w_stiffener_value=1.0, w_masked_presence=0.5, cumulative_scale_mm=20.0,
                p_tolerance_mm=10.0, n_p_bins=nb, p_bin_step_mm=C.Y_GRID_STEP, joint_topology=True,
                w_p_ce=2.0, label_scale_bins=1.0).to(DEV)

        def score(model, pre, nb, d):
            ld = DataLoader(ConnectionDataset(d, pre), batch_size=128)
            _, parts = run_epoch(model, ld, loss_for(pre, nb), DEV, use_true_topology=False)
            return parts["p_within_10"] + parts["topology_exact"], parts

        pre = Preprocessor().fit(tr)
        nb = n_bins_from_train(tr, C.P_COLUMNS, step_mm=C.Y_GRID_STEP)
        kw = dict(batch_size=BATCH)
        trl = DataLoader(ConnectionDataset(tr, pre), shuffle=True, **kw)
        val = DataLoader(ConnectionDataset(va, pre), **kw)
        model = ConnectionDesignNetV4(
            cat_cardinalities=pre.cat_cardinalities, num_continuous=len(C.NUMERIC_FEATURES),
            n_standard_classes=pre.n_standard_classes, p_mean_mm=pre.masked_mean[:C.N_Y],
            p_std_mm=pre.masked_std[:C.N_Y], hidden_dim=HID, num_blocks=NB, dropout=DROP, gru_hidden_dim=GRU,
            previous_p_scale_mm=200.0, cumulative_scale_mm=1000.0, n_p_bins=nb,
            p_bin_step_mm=C.Y_GRID_STEP, joint_topology=True).to(DEV)
        hist = fit(model, trl, val, loss_for(pre, nb), DEV, epochs=EPOCHS, lr=LR, weight_decay=WD,
                   log_every=20, select_metric="p_and_topology")

        new_score, new_parts = score(model.eval(), pre, nb, te)
        try:
            ppre = joblib.load(pdir / "preprocessor.joblib")
            ck = torch.load(pdir / "weights.pt", map_location="cpu", weights_only=False)
            pm = ConnectionDesignNetV4(
                cat_cardinalities=ck["cat_cardinalities"], num_continuous=len(ck["numeric_features"]),
                n_standard_classes=ck["n_standard_classes"], p_mean_mm=ck["p_mean_mm"], p_std_mm=ck["p_std_mm"],
                hidden_dim=ck.get("hidden_dim", 192), num_blocks=ck.get("num_blocks", 3),
                dropout=ck.get("dropout", 0.1), gru_hidden_dim=ck.get("gru_hidden_dim", 128),
                n_p_bins=ck["n_p_bins"], p_bin_step_mm=ck["p_bin_step_mm"],
                joint_topology=bool(ck.get("cfg", {}).get("joint_topology", False)))
            pm.load_state_dict(ck["model_state_dict"]); pm.to(DEV).eval()
            old_score, _ = score(pm, ppre, ck["n_p_bins"], te)
        except Exception:
            traceback.print_exc(); old_score = None
        promoted = old_score is not None and len(te) >= 50 and new_score >= old_score
        print(f"score nuevo {new_score:.4f} | padre {old_score} | promoted {promoted}", flush=True)

        name = next_name()
        tmp = config.MODELS_DIR / f"_building_{name}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(pdir / "core", tmp / "core", ignore=shutil.ignore_patterns("__pycache__"))
        ad = tmp / "core" / "adapter.py"
        ad.write_text(re.sub(r"^MODEL_NAME = .*$", "MODEL_NAME = Path(__file__).resolve().parent.parent.name",
                             ad.read_text(encoding="utf-8"), flags=re.M), encoding="utf-8")
        (tmp / "data").mkdir()
        for f in (pdir / "data").glob("tekla_*.csv"): shutil.copy(f, tmp / "data" / f.name)
        df.to_parquet(tmp / "data" / "datos_modelo.parquet", index=False)
        torch.save({"model_state_dict": model.state_dict(), "cfg": {"joint_topology": True},
                    "best_epoch": hist["best_epoch"], "n_p_bins": nb, "p_bin_step_mm": C.Y_GRID_STEP,
                    "cat_cardinalities": pre.cat_cardinalities, "numeric_features": list(C.NUMERIC_FEATURES),
                    "n_standard_classes": pre.n_standard_classes, "hidden_dim": HID, "num_blocks": NB,
                    "dropout": DROP, "gru_hidden_dim": GRU, "p_mean_mm": pre.masked_mean[:C.N_Y],
                    "p_std_mm": pre.masked_std[:C.N_Y]}, tmp / "weights.pt")
        joblib.dump(pre, tmp / "preprocessor.joblib")
        meta = {"name": name, "parent": parent, "date": datetime.now().isoformat(timespec="seconds"),
                "rows_used": len(df), "rows_raw": n_raw, "train": len(tr), "val": len(va), "test": len(te),
                "score_new": new_score, "score_parent": old_score, "promoted": promoted,
                "p_within_10": new_parts["p_within_10"], "topology_exact": new_parts["topology_exact"]}
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        os.replace(tmp, config.MODELS_DIR / name)
        if promoted:
            (config.MODELS_DIR / "active.json").write_text(json.dumps({"name": name}), encoding="utf-8")
        rm.update(status="done", new_version=name, promoted=promoted, activated=False,
                  rows_at_last_train=n_raw, pid=None, last_error=None)
    except Exception as ex:
        traceback.print_exc()
        rm.update(status="error", last_error=str(ex)[:500], rows_at_last_train=n_raw, pid=None)


if __name__ == "__main__":
    main()