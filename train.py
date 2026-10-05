"""Train a linear regression
(price = theta0 + theta1 * mileage) by gradient descent."""

import csv
import json
import sys

LEARNING_RATE = 0.1
ITERATIONS = 5000


def train(km: list[float], price: list[float]) -> tuple[float, float]:
    """Train the data using gradient descent."""
    m = len(km)
    # Normalize mileage to [0, 1] so gradient descent converges
    # with a reasonalbe learning curve
    km_min, km_max = min(km), max(km)
    scale = km_max - km_min
    x = [(k - km_min) / scale for k in km]
    theta0, theta1 = 0.0, 0.0
    for _ in range(ITERATIONS):
        errors = [(theta0 + theta1 * x[i]) - price[i] for i in range(m)]
        tmp0 = LEARNING_RATE * sum(errors) / m
        tmp1 = LEARNING_RATE * sum(errors[i] * x[i] for i in range(m)) / m
        theta0 -= tmp0
        theta1 -= tmp1
    real_theta1 = theta1 / scale
    real_theta0 = theta0 - real_theta1 * km_min
    return real_theta0, real_theta1


def load_data(path: str) -> tuple[list[float], list[float]]:
    """Load data from csv file and list km and price"""
    km = []
    price = []
    # newline = "" means don't translate newline, csv.reader handles it.
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            km.append(float(row[0]))
            price.append(float(row[1]))
    return km, price


def main():
    """Main function to load data, train with data and save
    theta0 & theta1 inot theta.json to use on predict.py"""
    path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    try:
        km, price = load_data(path)
    except (OSError, ValueError) as e:
        print(f"Error reading {path}: {e}")
        sys.exit(1)
    theta0, theta1 = train(km, price)
    with open("theta.json", "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)
    print(f"Training done: theta0 = {theta0:.4f}, theta1 = {theta1:.6f}")


if __name__ == "__main__":
    main()
