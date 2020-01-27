import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction


class ContinuousNewsvendor(SyntheticTestFunction):
    """
    This is the continuous news-vendor problem adopted from SimOpt.
    See the write-up for implementation details.
    """
    dim = 3
    _bounds = [(0, 1) for _ in range(dim)]
    _optimizers = None  # is given by a formula in notes for given alpha beta

    cost = 5
    sell_price = 9
    salvage = 1
    alpha_lb = 1  # lower bound of alpha range
    alpha_range = 2  # spread of alpha range, i.e. alpha is in [alpha_lb, alpha_lb + alpha_range], uniformly distributed
    beta_lb = 10  # lower bound of beta range
    beta_range = 20  # spread of beta range, same as alpha
    x_lb = 0  # same
    x_range = 1  # same

    def __init__(self, run_length: int = 10, crn: bool = False):
        """
        Initialize the problem
        :param run_length: Number of days of demand to simulate. Reduces the observation error.
        :param crn: If True, CRN will be employed, i.e. same demands are used to evaluate each solution.
        """
        super().__init__()
        if run_length < 1:
            raise ValueError("run_length must be a positive integer.")
        self.run_length = run_length
        self.crn = crn

    def forward(self, X: Tensor, noise: bool = True, seed: int = None) -> Tensor:
        """
        Simulates the demand for run_length days and returns the objective value.
        The result is negated to get a minimization problem
        :param X: First dimension is x, last two dimensions are alpha and beta of the distribution.
                    Should be standardized to unit-hypercube, see bounds defined above.
                    Tensor of size(-1) = 3. Return is of appropriate batch shape.
        :param noise: Noise free evaluation is not available, leave as True.
        :param seed: If given, this is the seed for random number generation.
        :return: - Profit after run_length days, same batch shape as X
        """
        if not noise:
            raise ValueError("Noise free evaluation is not available.")
        if X.size(-1) != self.dim:
            raise ValueError("X must have size(-1) = 3.")
        x, alpha, beta = torch.split(X, 1, dim=-1)
        # store the old state and set the seed
        old_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed=seed)
        except TypeError:
            torch.random.seed()
        # project the variables back from the unit domain
        x = self.x_lb + x * self.x_range
        alpha = self.alpha_lb + alpha * self.alpha_range
        beta = self.beta_lb + beta * self.beta_range

        # generate the demands
        demand = torch.empty((*x.size()[:-1], self.run_length))
        if self.crn:
            for i in range(self.run_length):
                demand[..., i] = torch.pow(torch.pow(torch.rand(1).repeat(*x.size()), -1/beta.double()) - 1,
                                           1/alpha.double()).squeeze(-1)
        else:
            for i in range(self.run_length):
                demand[..., i] = torch.pow(torch.pow(torch.rand(*x.size()), -1/beta.double()) - 1,
                                           1/alpha.double()).squeeze(-1)

        # compute profit
        period_cost = self.cost * x
        sales = torch.min(demand, x.expand(*x.size()[:-1], self.run_length))
        sales_revenue = self.sell_price * sales
        salvage_revenue = self.salvage * (x.expand(*x.size()[:-1], self.run_length) - sales)
        profit = sales_revenue + salvage_revenue - period_cost

        # restore old random state
        torch.random.set_rng_state(old_state)

        return -torch.mean(profit, dim=-1, keepdim=True)

    def evaluate_true(self, X: Tensor) -> Tensor:
        raise NotImplementedError("True function evaluation is not available.")


if __name__ == "__main__":
    # for testing purposes
    from time import time
    start = time()
    ctnv = ContinuousNewsvendor()
    print(ctnv(torch.tensor([0.9, 0.5, 0.5])))
    print('time: ', time()-start)
