import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightning.pytorch.utilities.warnings import PossibleUserWarning
import warnings
warnings.filterwarnings("ignore", category=PossibleUserWarning)

THIS = Path(__file__).resolve()
PROJ = THIS.parents[1]
PY = (PROJ / ".venv" / "Scripts" / "python.exe") if sys.platform.startswith("win") else sys.executable
RUN = PROJ / "tests" / "run_mnist.py"
LOGDIR = PROJ / "lightning_logs"


def _latest_metrics_csv(exp_name: str) -> Path:
    base = LOGDIR / exp_name
    if not base.exists():
        raise FileNotFoundError(f"Nie ma katalogu z logami: {base}")
    versions = sorted(base.glob("version_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not versions:
        raise FileNotFoundError(f"Brak version_* w {base}")
    csv = versions[0] / "metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Brak metrics.csv w {versions[0]}")
    return csv


def _read_first_n_batches(csv_path: Path, n: int = 100) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    # zostaw tylko interesujące kolumny
    cols = [c for c in df.columns if c in ("step", "global_step", "train_loss", "policy_loss")]
    if not cols:
        raise ValueError("Brak oczekiwanych kolumn w metrics.csv")
    df = df[cols]

    # ustandaryzuj nazwę kroku
    if "global_step" in df.columns and "step" not in df.columns:
        df.rename(columns={"global_step": "step"}, inplace=True)

    if "train_loss" not in df.columns:
        raise ValueError("Brak kolumny train_loss w metrics.csv")

    # zostaw tylko wiersze z wartościami train_loss (kroki treningowe)
    df = df[df["train_loss"].notna()].copy()

    # pierwsze n po sortowaniu
    return df.sort_values("step").head(n).reset_index(drop=True)


def _summarize(df: pd.DataFrame, label: str) -> str:
    tl = df["train_loss"].dropna()
    pl = df["policy_loss"].dropna() if "policy_loss" in df.columns else None

    def slope_per_100(series):
        if series is None or series.empty:
            return None
        steps = df.loc[series.index, "step"]
        if len(series) < 2 or steps.iloc[-1] == steps.iloc[0]:
            return None
        return (series.iloc[-1] - series.iloc[0]) / (steps.iloc[-1] - steps.iloc[0]) * 100.0

    def line(name, series):
        if series is None or series.empty:
            return f"{name}: brak"
        s100 = slope_per_100(series)
        s100_txt = f"{s100:.4g}" if s100 is not None else "n/a"
        return (f"{name}: start={series.iloc[0]:.4g}, "
                f"koniec={series.iloc[-1]:.4g}, "
                f"min={series.min():.4g}, "
                f"mean={series.mean():.4g}, "
                f"slope/100steps={s100_txt}")

    return "\n".join([
        f"=== {label} (N={len(tl)}) ===",
        line("train_loss", tl),
        line("policy_loss", pl),
    ])


def analyze(exp_a: str, exp_b: str, n: int = 100) -> None:
    csv_a = _latest_metrics_csv(exp_a)
    csv_b = _latest_metrics_csv(exp_b)
    df_a = _read_first_n_batches(csv_a, n=n)
    df_b = _read_first_n_batches(csv_b, n=n)

    # wydruk podsumowania
    print(_summarize(df_a, f"{exp_a} (first {n})"))
    print(_summarize(df_b, f"{exp_b} (first {n})"))

    # metryka "czas do progu"
    def time_to_threshold(df, col="train_loss", thr=0.30):
        hit = df[df[col] <= thr]
        return int(hit.iloc[0]["step"]) if not hit.empty else None

    tta = time_to_threshold(df_a)
    ttb = time_to_threshold(df_b)
    print(f"[extra] time_to_0.30: {exp_a}={tta}, {exp_b}={ttb}")

    # wykres: smoothing + twin y
    out_dir = PROJ / "tests" / "_ab_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{exp_a}_vs_{exp_b}_first_{n}.png"

    def smooth(s, w=5):
        return s.rolling(w, min_periods=1).mean()

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()

    # train_loss (oś lewa)
    l1, = ax1.plot(df_a["step"], smooth(df_a["train_loss"]), label=f"{exp_a} / train_loss")
    l2, = ax1.plot(df_b["step"], smooth(df_b["train_loss"]), label=f"{exp_b} / train_loss")

    # policy_loss (oś prawa) – tylko jeśli jest w CSV
    lines = [l1, l2]
    if "policy_loss" in df_a.columns and df_a["policy_loss"].notna().any():
        l3, = ax2.plot(df_a["step"], smooth(df_a["policy_loss"]), linestyle="--", label=f"{exp_a} / policy_loss")
        lines.append(l3)
    if "policy_loss" in df_b.columns and df_b["policy_loss"].notna().any():
        l4, = ax2.plot(df_b["step"], smooth(df_b["policy_loss"]), linestyle="--", label=f"{exp_b} / policy_loss")
        lines.append(l4)

    ax1.set_xlabel("step")
    ax1.set_ylabel("train_loss")
    ax2.set_ylabel("policy_loss")
    ax2.grid(False)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    plt.title(f"First {n} batches – {exp_a} vs {exp_b} (smoothed)")
    plt.tight_layout()
    plt.savefig(png, dpi=160)
    print(f"[OK] Zapisano wykres: {png}")


def run_one(init_as_zeros: int, max_steps: int, logger_name: str) -> int:
    cmd = [
        str(PY), str(RUN),
        "--init_as_zeros", str(init_as_zeros),
        "--max_steps", str(max_steps),
        "--logger_name", logger_name,
    ]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--skip_runs", action="store_true", help="tylko analiza istniejących logów")
    parser.add_argument("--n", type=int, default=100, help="ile batchy porównać")
    parser.add_argument("--exp_a", type=str, default="init_zeros")
    parser.add_argument("--exp_b", type=str, default="init_warm")
    args = parser.parse_args()

    if not args.skip_runs:
        rc = run_one(1, args.max_steps, args.exp_a)
        if rc != 0:
            sys.exit(rc)
        rc = run_one(0, args.max_steps, args.exp_b)
        if rc != 0:
            sys.exit(rc)

    analyze(args.exp_a, args.exp_b, n=args.n)


if __name__ == "__main__":
    main()
