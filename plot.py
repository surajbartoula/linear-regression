"""Plot the dataset and the regression line."""

import sys
import matplotlib.pyplot as plt
from train import load_data
from predict import load_thetas, estimate_price


def main():
    """Plot the csv file and the line from trained data."""
    path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    km, price = load_data(path)
    theta0, theta1 = load_thetas()
    xs = [min(km), max(km)]
    ys = [estimate_price(x, theta0, theta1) for x in xs]
    plt.scatter(km, price, label="data")
    plt.plot(xs, ys, color="red", label="regression line")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
