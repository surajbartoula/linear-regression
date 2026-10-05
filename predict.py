"""Predict the price of a car for a given mileage."""

import json


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """Our linear model estimatePrice(mileage) = theta0 + (theta1 * mileage)"""
    return theta0 + theta1 * mileage


def load_thetas(path: str = "thetas.json") -> tuple[float, float]:
    """Load thetas from json file if not initialize as 0."""
    try:
        with open(path) as f:
            d = json.load(f)
        return d["theta0"], d["theta1"]
    except (OSError, KeyError, ValueError):
        return 0.0, 0.0  # Model not trained yet


def main():
    """Main function to load thetas and estimate price."""
    theta0, theta1 = load_thetas()
    while True:
        try:
            mileage = float(input("Enter a mileage (km): "))
        except ValueError:
            print("Please enter a valid number.")
            continue
        except EOFError:
            return
        if mileage < 0:
            print("Mileage must be positive.")
            continue
        break
    print(f"Estimated price: {estimate_price(mileage, theta0, theta1):.2f}")


if __name__ == "__main__":
    main()
