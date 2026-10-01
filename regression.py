# -*- coding: utf-8 -*-
"""
키(cm)로 몸무게(kg)를 예측하는 회귀분석

모델
  1) 선형회귀             y = b0 + b1·x            (최소제곱법, 경사하강법으로 검증)
  2) 2차 다항회귀          y = c0 + c1·x + c2·x²    (곡선이지만 계수에 대해선 선형)
  3) 비선형회귀 - 거듭제곱  y = a · x^b              (비선형 최소제곱, scipy curve_fit)
  4) 비선형회귀 - 지수      y = a · e^(b·x)          (비선형 최소제곱, scipy curve_fit)

사용법
  python regression.py                 # 학습 + 결과 출력 + 그래프 저장(images/)
  python regression.py --height 175    # 175cm일 때 네 모델의 예측 몸무게
  python regression.py --show          # 그래프 창 띄우기
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "height_weight_only.xlsx"
IMAGE_DIR = ROOT / "images"


# ── 데이터 ───────────────────────────────────────────
def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    x = df["키 (cm)"].to_numpy(dtype=float)
    y = df["몸무게 (kg)"].to_numpy(dtype=float)
    return x, y


# ── 평가 지표 ─────────────────────────────────────────
def r2_score(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


# ── 1. 선형회귀 ───────────────────────────────────────
def fit_linear(x, y):
    """최소제곱법 공식: b1 = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²,  b0 = ȳ - b1·x̄"""
    xm, ym = x.mean(), y.mean()
    b1 = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    b0 = ym - b1 * xm
    return float(b0), float(b1)


def fit_linear_gd(x, y, lr=0.1, epochs=1000):
    """같은 선형회귀를 경사하강법으로 학습 (MSE 최소화)"""
    n = len(x)
    xm, xsd = x.mean(), x.std()
    xs = (x - xm) / xsd                   # 표준화 → 학습이 빠르고 안정적
    w, b = 0.0, 0.0
    for _ in range(epochs):
        err = (w * xs + b) - y
        w -= lr * (2 / n) * np.sum(err * xs)   # ∂MSE/∂w
        b -= lr * (2 / n) * np.sum(err)        # ∂MSE/∂b
    b1 = w / xsd
    b0 = b - b1 * xm
    return float(b0), float(b1)


# ── 2. 2차 다항회귀 ───────────────────────────────────
def fit_poly2(x, y):
    c2, c1, c0 = np.polyfit(x, y, 2)      # 최고차항부터 반환
    return float(c0), float(c1), float(c2)


# ── 3. 비선형회귀: 거듭제곱 ───────────────────────────
def power_model(x, a, b):
    return a * np.power(x, b)


def fit_power(x, y):
    # 초기값: log y = log a + b·log x 직선의 계수
    b_init, loga_init = np.polyfit(np.log(x), np.log(y), 1)
    (a, b), _ = curve_fit(power_model, x, y, p0=[np.exp(loga_init), b_init], maxfev=10000)
    return float(a), float(b)


# ── 4. 비선형회귀: 지수 ───────────────────────────────
def exp_model(x, a, b):
    return a * np.exp(b * x)


def fit_exp(x, y):
    # 초기값: log y = log a + b·x 직선의 계수
    b_init, loga_init = np.polyfit(x, np.log(y), 1)
    (a, b), _ = curve_fit(exp_model, x, y, p0=[np.exp(loga_init), b_init], maxfev=10000)
    return float(a), float(b)


# ── 전체 학습 ─────────────────────────────────────────
def fit_all(x, y):
    b0, b1 = fit_linear(x, y)
    c0, c1, c2 = fit_poly2(x, y)
    pa, pb = fit_power(x, y)
    ea, eb = fit_exp(x, y)
    return {
        "선형": {
            "params": [b0, b1],
            "equation": f"y = {b0:.4f} + {b1:.4f}·x",
            "predict": lambda h: b0 + b1 * np.asarray(h, float),
        },
        "2차 다항": {
            "params": [c0, c1, c2],
            "equation": f"y = {c0:.4f} {c1:+.4f}·x {c2:+.6f}·x²",
            "predict": lambda h: c0 + c1 * np.asarray(h, float) + c2 * np.asarray(h, float) ** 2,
        },
        "비선형(거듭제곱)": {
            "params": [pa, pb],
            "equation": f"y = {pa:.6e} · x^{pb:.4f}",
            "predict": lambda h: power_model(np.asarray(h, float), pa, pb),
        },
        "비선형(지수)": {
            "params": [ea, eb],
            "equation": f"y = {ea:.4f} · e^({eb:.5f}·x)",
            "predict": lambda h: exp_model(np.asarray(h, float), ea, eb),
        },
    }


def predict_all(models, height):
    return {name: float(m["predict"](height)) for name, m in models.items()}


# ── 그래프 ────────────────────────────────────────────
def save_plots(x, y, models, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMAGE_DIR.mkdir(exist_ok=True)
    labels = {"선형": "Linear", "2차 다항": "Polynomial (deg 2)",
              "비선형(거듭제곱)": "Nonlinear: power", "비선형(지수)": "Nonlinear: exponential"}
    grid = np.linspace(x.min(), x.max(), 300)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x, y, s=18, alpha=0.45, color="gray", label="Data (n=%d)" % len(x))
    for name, m in models.items():
        ax.plot(grid, m["predict"](grid), lw=2, label=labels[name])
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title("Height vs Weight: regression fits")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "fit.png", dpi=150)

    fig2, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (name, m) in zip(axes.ravel(), models.items()):
        yhat = m["predict"](x)
        ax.scatter(yhat, y - yhat, s=14, alpha=0.5)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(labels[name])
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Predicted weight (kg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Residual (kg)")
    fig2.suptitle("Residual analysis")
    fig2.tight_layout()
    fig2.savefig(IMAGE_DIR / "residuals.png", dpi=150)

    if show:
        plt.show()


# ── 실행 ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="키로 몸무게를 예측하는 회귀분석")
    parser.add_argument("--height", type=float, help="예측할 키(cm)")
    parser.add_argument("--show", action="store_true", help="그래프 창 띄우기")
    args = parser.parse_args()

    x, y = load_data()
    models = fit_all(x, y)

    if args.height is not None:
        h = args.height
        if h < x.min() or h > x.max():
            print(f"주의: 학습 데이터 범위({x.min():.0f}~{x.max():.0f}cm) 밖이라 예측을 믿기 어렵습니다.")
        print(f"키 {h:g} cm의 예측 몸무게")
        for name, w in predict_all(models, h).items():
            print(f"  {name:<12} {w:6.1f} kg")
        return

    print(f"데이터: {len(x)}명, 키 {x.min():.0f}~{x.max():.0f}cm\n")
    print(f"{'모델':<14}{'R²':>8}{'RMSE(kg)':>11}   식")
    for name, m in models.items():
        yhat = m["predict"](x)
        print(f"{name:<14}{r2_score(y, yhat):8.4f}{rmse(y, yhat):11.3f}   {m['equation']}")

    gd_b0, gd_b1 = fit_linear_gd(x, y)
    print(f"\n경사하강법 선형회귀: y = {gd_b0:.4f} + {gd_b1:.4f}·x  (최소제곱법과 같아야 함)")

    print("\n예측 예시")
    for h in (160, 170, 180):
        preds = predict_all(models, h)
        print(f"  {h}cm → " + ", ".join(f"{k} {v:.1f}kg" for k, v in preds.items()))

    save_plots(x, y, models, show=args.show)
    print(f"\n그래프 저장: {IMAGE_DIR / 'fit.png'}, {IMAGE_DIR / 'residuals.png'}")


if __name__ == "__main__":
    main()
