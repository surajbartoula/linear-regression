"""Measure the precision of the model (MSE, RMSE, MAER^2)."""

import sys
from train import load_data
from predict import load_thetas, estimate_price


def main():
    """Calculate MSE, RMSE, MAE and R^2 to measure precision
    of the model."""
    path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    km, price = load_data(path)
    theta0, theta1 = load_thetas()
    m = len(km)
    preds = [estimate_price(k, theta0, theta1) for k in km]
    errors = [preds[i] - price[i] for i in range(m)]
    mse = sum(e * e for e in errors) / m
    mae = sum(abs(e) for e in errors) / m
    mean_p = sum(price) / m
    ss_tot = sum((p - mean_p) ** 2 for p in price)
    ss_res = sum(e * e for e in errors)
    r2 = 1 - ss_res / ss_tot
    print(f"MSE : {mse: .2f}")
    print(f"RMSE: {mse ** 0.5:.2f}")
    print(f"MAE : {mae: .2f}")
    print(f"R^2 : {r2:.4f}")


if __name__ == "__main__":
    main()
