import numpy as np
import random
random.seed(42)
np.random.seed(42)
from scipy import special
import os
from deap import base, creator, tools, algorithms
import argparse
import json

ACT_FUNCS = {
    "swish": lambda x: x / (1.0 + np.exp(-x)),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1 + special.erf(x / np.sqrt(2))),
    "hswish": lambda x: x * np.clip(x + 3, 0, 6) / 6,
    "exp": lambda x: np.exp(x),
    "reci": lambda x: np.reciprocal(x),
    "sqrt_reci": lambda x: np.reciprocal(np.sqrt(x)),
}

def round_to_nearest_bits(x, decimal_bits):
    """

    :param x: floating input
    :param decimal_bits: bits that the input should reserve
    :return: the formatted input with specific decimal bits
    """
    scaled_value = x * (2 ** decimal_bits)
    rounded_value = np.round(scaled_value)  # very important
    result = rounded_value / (2 ** decimal_bits)
    return result

def save_to_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def load_from_file(filename):
    with open(filename, "r") as file:
        return json.load(file)


def calculate_coeff_bias(a1, a2, act_type, bit, pr=False):
    func = ACT_FUNCS[act_type]
    if pr: print(a1, a2)
    if a2 != a1:
        coeff = (func(a2) - func(a1)) / (a2 - a1)
        if pr:print(func(a2),func(a1), a2-a1, coeff)
        bias = -a1 * coeff + func(a1)
    else:
        # Handle the case where a2 is equal to a1
        coeff = 0  # or some other appropriate value or handling method
        bias = 0
    resize = 2 ** (bit - 2)  # coeff is directly converted to fixed point format with the specified bit
    return (np.round(coeff * resize) / resize, np.round(bias * resize) / resize)


def get_list(split_point, act_type, a_bit, pr=False):
    coeff_bias_pairs = [calculate_coeff_bias(a1, a2, act_type, a_bit, pr=pr) for a1, a2 in
                        zip(split_point[:-1], split_point[1:])]
    coeff, bias = zip(*coeff_bias_pairs)
    return coeff, bias


def piecewise_linear_approximation(x, split_points, slopes, biases):
    index = np.digitize(x, split_points) - 1
    index = min(index, len(slopes) - 1)
    return slopes[index] * x + biases[index]


def compute_errors(func_original, func_approx, x_values):
    y_original = np.array([func_original(x) for x in x_values])
    y_approx = np.array([func_approx(x) for x in x_values])

    l1_loss = np.mean(np.abs(y_original - y_approx))
    l2_loss = np.sqrt(np.mean((y_original - y_approx) ** 2))
    mse = np.mean((y_original - y_approx) ** 2)

    return l1_loss, l2_loss, mse


def create_fixed_point_attr(decimal_bits, sp_range):
    rand_val = np.random.uniform(sp_range[0], sp_range[1])
    return rand_val

def create_float_point_attr(sp_range):
    rand_val = np.random.uniform(sp_range[0], sp_range[1])
    return rand_val


def genetic_find_best_split_points_one_shot(func_name, x_range, sp_range, num_splits, total_iters=1000, decimal_bits_range=(0,6),
                                   pop_size=50, crossover_prob=0.7, mutation_prob=0.2, offset=0, neg_inf=-4, pos_inf=4, w_b_bit=8, mutate=True):
    print("processing range:", decimal_bits_range)
    func = ACT_FUNCS[func_name]
    if "FitnessMin" not in creator.__dict__:
        # -1 represents to minimize the error
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        # Create the individual class
        creator.create("Individual", list, fitness=creator.FitnessMin)
    # Create the toolbox for different functions
    toolbox = base.Toolbox()
    # Generates values within the desired range with the specified decimal bits
    toolbox.register("attr_float", create_float_point_attr, sp_range)
    # Initialize the individual class with the attr_float function
    # The initial values are all in fixed point format decided by the decimal_bits
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float,
                     num_splits)  # Individual is a person
    # Initialize the population class with the individual class
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Population is a group of people

    def mutate_fixed_point(individual, mu, sigma, indpb, offset=0, mutate=True):
        if mutate:
            for i in range(len(individual)):
                for j in range(decimal_bits_range[0]+offset, decimal_bits_range[1] + 1): # decimal_bits_range[1] + 1
                    p = np.random.random()
                    if p >= 0.05 * j and p < 0.05 * (j + 1):
                        scale_factor = 2 ** (decimal_bits_range[0] + j)
                        individual[i] = round(individual[i] * scale_factor) / scale_factor
                        individual[i] = min(max(individual[i], sp_range[0]), sp_range[1])
                    elif p >= 0.9:
                        individual[i] += np.random.normal(mu, sigma)
                individual[i] = min(max(individual[i], sp_range[0]), sp_range[1])
        else:
            for i in range(len(individual)):
                if np.random.random() >= 0.9:
                    individual[i] += np.random.normal(mu, sigma)
                    individual[i] = min(max(individual[i], sp_range[0]), sp_range[1])
        return individual,

    # Define the evaluation function
    def evaluate(individual):
        individual.sort()
        split_points = [neg_inf] + individual + [pos_inf]
        coeff, bias = get_list(split_points, func_name, w_b_bit)
        error = 0.0
        x_values = np.arange(x_range[0], x_range[1], 0.01)
        y_values = np.array([func(x) for x in x_values])
        split_points = [neg_inf if split_point < neg_inf else pos_inf if split_point > pos_inf else split_point for
                        split_point in split_points]
        approx_values = [piecewise_linear_approximation(x, split_points, coeff, bias) for x in x_values]
        error += np.mean((y_values - approx_values) ** 2)
        return error,

    # cxTwoPoint: crossover function for every two individuals whose probability > crossover_prob
    toolbox.register("mate", tools.cxTwoPoint)  # cross probility is crossover_prob
    # mutation function
    toolbox.register("mutate", mutate_fixed_point, mu=0, sigma=0.2, indpb=0.1, offset=offset, mutate=mutate)
    # selection function
    toolbox.register("select", tools.selTournament,
                     tournsize=3)  # Randomly select 3 individuals and choose the best one
    # evaluation function
    toolbox.register("evaluate", evaluate)
    # construct the population with pop_size individuals
    population = toolbox.population(n=pop_size)
    # Imitate the evolution process
    algorithms.eaSimple(population, toolbox, crossover_prob, mutation_prob, total_iters)
    # Select the best individual
    best_individual = tools.selBest(population, 1)[0]
    best_splits = [neg_inf] + best_individual + [pos_inf]
    best_splits.sort()
    return best_splits


def autopwl(activation_function_name, x_range=(-4, 4), sp_range=(-4, 4), num_splits=10, total_iters=100, decimal_bit=5, offset=0, neg_inf=-4, pos_inf=4, w_b_bit=8, mutate=True):
    if activation_function_name not in ACT_FUNCS:
        print("Invalid activation function name. Valid names are:", ", ".join(ACT_FUNCS.keys()))
        return
    print("x_range:", x_range, "sp_range:", sp_range)
    split_points = genetic_find_best_split_points_one_shot(activation_function_name, x_range, sp_range, num_splits, total_iters, decimal_bit, offset=offset, neg_inf=neg_inf, pos_inf=pos_inf, w_b_bit=w_b_bit, mutate=mutate)
    coeff, bias = get_list(split_points, activation_function_name, w_b_bit, pr=True)
    return split_points, coeff, bias


def offline_pwlstore_one_shot(act_func='hswish', x_range=(-3.5, 3.5), sp_range=(-3, 3), decimal_bit_range=(0, 6), num_splits=7,
                     total_iters=100, mutate=True):
    if act_func =='gelu' and num_splits == 7: offset = 2
    elif act_func =='hswish' and num_splits==15:offset = 2
    else: offset = 0
    print("offset:", offset)

    if act_func == 'gelu' or act_func == 'hswish':
        neg_inf = -10000.0
        pos_inf = 10000.0
        w_b_bit = 8
    elif act_func == 'exp':
        neg_inf = -16.0
        pos_inf = 0.0
        w_b_bit = 8
    elif act_func == 'reci':
        neg_inf = 0.5
        pos_inf = 4.0
        w_b_bit = 8
    elif act_func == 'sqrt_reci':
        neg_inf = 0.25
        pos_inf = 4.0
        w_b_bit = 8
    else:
        raise NotImplementedError('Not support')
    results = {}
    results[act_func] = {}
    split_points, coeff, bias = autopwl(act_func, x_range=x_range, sp_range=sp_range, num_splits=num_splits, total_iters=total_iters, decimal_bit=decimal_bit_range, offset=offset, neg_inf=neg_inf, pos_inf=pos_inf, w_b_bit=w_b_bit, mutate=mutate)
    for bit in range(decimal_bit_range[0], decimal_bit_range[1]+1):
        split_points_tmp = [round_to_nearest_bits(split_point, bit) for split_point in split_points]
        print(f"Start for activation_function: {act_func} and decimal_bit: {bit}")
        results[act_func][bit] = {
            "breakpoints": split_points_tmp[1:-1],
            "slopes": coeff,
            "intercepts": bias
        }
    save_to_file(results, f"./pretrained/{act_func}_pwl_{num_splits}.json")


def config_parser():
    parser = argparse.ArgumentParser(description='GQA-LUT')
    parser.add_argument("--act_func", type=str, default='hswish', help="Activation function name")
    parser.add_argument("--num_splits", type=int, default=7, help="Number of split points")
    parser.add_argument("--total_iters", type=int, default=500, help="Total iterations")
    parser.add_argument("--decimal_bit", type=int, default=6, help="Decimal bit precision")
    parser.add_argument("--decimal_bit_range", nargs='+', type=int, default=(0, 6), help="Decimal bit range")
    parser.add_argument("--x_range", nargs='+', type=float, help="List of split points")
    parser.add_argument("--sp_range", nargs='+', type=float, help="List of split points")
    parser.add_argument("--mutate", action='store_true', default=False, help="Whether using mutation")
    return parser

def main():
    parser = config_parser()
    args = parser.parse_args()
    offline_pwlstore_one_shot(act_func=args.act_func, x_range=args.x_range, sp_range=args.sp_range,
                         decimal_bit_range=args.decimal_bit_range, num_splits=args.num_splits,
                         total_iters=args.total_iters, mutate=args.mutate)


if __name__ == "__main__":
    main()
