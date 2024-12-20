import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from scipy import special

ACT_FUNCS = {
    "swish": lambda x: x / (1.0 + np.exp(-x)),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1 + special.erf(x / np.sqrt(2))),
    "hswish": lambda x: x * np.clip(x + 3, 0, 6) / 6,
    "exp": lambda x: np.exp(x),
    "reci": lambda x: np.reciprocal(x),
    "sqrt_reci": lambda x: np.reciprocal(np.sqrt(x)),
    "silu": lambda x: x / (1 + np.exp(-x)),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot an activation function")
    parser.add_argument(
        "--func", type=str, choices=ACT_FUNCS.keys(), required=False, 
        help="Activation function to plot (e.g., 'swish', 'sigmoid', etc.)"
    )
    parser.add_argument(
        "--xmin", type=float, default=-5.0, help="Minimum x value for plot"
    )
    parser.add_argument(
        "--xmax", type=float, default=5.0, help="Maximum x value for plot"
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples for plotting"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the plot images"
    )
    parser.add_argument(
        "--json_file", type=str, default=None, help="Path to the JSON file with parameterized functions"
    )
    return parser.parse_args()

def load_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def plot_and_save_func(
    original_func, pwl_func, xmin, xmax, samples, output_dir, filename, 
    breakpoints=None, slopes=None, intercepts=None
):
    x = np.linspace(xmin, xmax, samples)
    y_original = original_func(x)
    y_fitted = pwl_func(x, xmin, xmax, breakpoints, slopes, intercepts) if breakpoints is not None else None

    plt.figure(figsize=(6, 4))
    plt.plot(x, y_original, label="Original Function", color="blue")
    if y_fitted is not None:
        plt.plot(x, y_fitted, label="PWL Function", linestyle="--", color="red")
    plt.title(f"Activation Function: {filename}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/{filename}.png")
    plt.close()

def pwl_func(x, xmin, xmax, breakpoints, slopes, intercepts):
    y = np.zeros_like(x)
    breakpoints = np.insert(breakpoints, 0, xmin)
    breakpoints = np.append(breakpoints, xmax)

    for i in range(0, len(breakpoints)-1):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i+1])
        y[mask] = slopes[i] * x[mask] + intercepts[i]

    y[x == xmax] = slopes[-1] * x[x == xmax] + intercepts[-1]
    return y

def plot_json_functions(json_data, xmin, xmax, samples, output_dir):
    for func_name, params in json_data.items():
        if func_name not in ACT_FUNCS:
            print(f"Warning: No original function found for {func_name}. Skipping.")
            continue

        original_func = ACT_FUNCS[func_name]  

        for key, func_params in params.items():
            breakpoints = np.array(func_params["breakpoints"])
            slopes = np.array(func_params["slopes"])
            intercepts = np.array(func_params["intercepts"])

            filename = f"{func_name}_{key}"
            plot_and_save_func(
                original_func, pwl_func, xmin, xmax, samples, output_dir, filename, 
                breakpoints, slopes, intercepts
            )

def main():
    args = parse_args()

    if args.func:
        func = ACT_FUNCS[args.func]
        plot_and_save_func(func, lambda x, *_: None, args.xmin, args.xmax, args.samples, args.output_dir, args.func)
    if args.json_file:
        json_data = load_json(args.json_file)
        plot_json_functions(json_data, args.xmin, args.xmax, args.samples, args.output_dir)

if __name__ == "__main__":
    main()
