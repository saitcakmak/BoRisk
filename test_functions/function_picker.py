"""
The function picker.
Find most in: https://www.sfu.ca/~ssurjano/optimization.html
"""
from test_functions.simple_test_functions import SineQuadratic, SimpleQuadratic
from test_functions.standardized_function import StandardizedFunction
from test_functions.cont_newsvendor import ContinuousNewsvendor
from test_functions.prod_line import ProductionLine
from test_functions.branin_williams import BraninWilliams
from test_functions.robust_synthetic import Marzat6
from test_functions.cvx_portfolio_simulator import CVXPortfolioSimulator
from test_functions.covid_exp_class import CovidSim
from botorch.test_functions import (Ackley,
                                    Beale,
                                    Branin,
                                    DixonPrice,
                                    EggHolder,
                                    Levy,
                                    Rastrigin,
                                    SixHumpCamel,
                                    ThreeHumpCamel,
                                    Powell,
                                    Hartmann)
from botorch.test_functions import SyntheticTestFunction

function_dict = {"simplequad": SimpleQuadratic,
                 "sinequad": SineQuadratic,
                 "powell": Powell,
                 "beale": Beale,
                 "dixonprice": DixonPrice,
                 "eggholder": EggHolder,
                 "levy": Levy,
                 "rastrigin": Rastrigin,
                 "branin": Branin,
                 "ackley": Ackley,
                 "hartmann3": Hartmann,
                 "hartmann4": Hartmann,
                 "hartmann6": Hartmann,
                 "sixhumpcamel": SixHumpCamel,
                 "threehumpcamel": ThreeHumpCamel,
                 "braninwilliams": BraninWilliams,
                 "marzat": Marzat6,
                 "portfolio": CVXPortfolioSimulator,
                 "covid": CovidSim}


def function_picker(function_name: str, noise_std: float = 0.1,
                    negate: bool = False) -> SyntheticTestFunction:
    """
    Returns the appropriate function callable
    If adding new BoTorch test functions, run them through StandardizedFunction.
    StandardizedFunction and all others listed here allow for a seed to be specified.
    If adding something else, make sure the forward (or __call__) takes a seed argument.
    :param function_name: Function to be used
    :param noise_std: observation noise level
    :param negate: In most cases, should be true for maximization
    :return: Function callable
    """
    if function_name == 'newsvendor':
        function = ContinuousNewsvendor()
    elif function_name == 'prod_line':
        function = ProductionLine()
    elif function_name in function_dict.keys():
        if function_name[-1].isdigit():
            function = StandardizedFunction(function_dict[function_name](dim=int(function_name[-1]),
                                                                         noise_std=noise_std, negate=negate))
        else:
            function = StandardizedFunction(function_dict[function_name](noise_std=noise_std, negate=negate))
    else:
        raise ValueError("Function name was not found!")

    return function
