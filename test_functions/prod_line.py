import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction
from numpy.random import exponential
import numpy as np


class ProductionLine(SyntheticTestFunction):
    """
    This is the ProductionLine problem adopted from SimOpt.
    See the write-up for implementation details.
    """
    _optimizers = None
    rate_lb = 0  # lower bound of server rates
    rate_scale = 2  # scale of server rates
    arrival_lb = 0.2  # lower bound of arrival rate
    arrival_scale = 1  # scale of the arrival rate
    # some parameters of the problem
    r = 10000
    c_0 = 1
    c_1 = 400

    def __init__(self, num_servers: int = 3, capacity: int = 10, server_cost: Tensor = torch.tensor([1, 5, 9]),
                 run_length: float = 1000, repetitions: int = 10):
        """
        Initialize the problem
        :param num_servers: Number of servers in the system, also gives the problem dimension (num_servers + 1)
        :param capacity: the queue capacity of each server
        :param server_cost: a vector of server costs, used in revenue calculation
        :param run_length: Length of time to run the system
        :param repetitions: Number of repetitions, controls the observation error
        """
        self.num_servers = num_servers
        self.dim = num_servers + 1
        self._bounds = [(0, 1) for _ in range(self.dim)]
        super().__init__()
        self.capacity = capacity
        if server_cost.reshape(-1).size(0) != num_servers:
            raise ValueError("server_cost must be a tensor of size num_servers")
        self.server_cost = server_cost.reshape(-1)
        self.run_length = run_length
        self.repetitions = repetitions

    def forward(self, X: Tensor, noise: bool = True, seed: int = None) -> Tensor:
        """
        Handles the input processing and returns the average revenue
        Returns the negative revenue for minimization
        :param X: First dimensions are for server rates, the last dimension is for arrival rate lambda.
                    Should be standardized to unit-hypercube, see bounds defined above for scaling.
                    Tensor of size(-1) = dim. Return is of appropriate batch shape.
        :param noise: Noise free evaluation is not available, leave as True.
        :param seed: If given, this is the seed for random number generation
        :return: Average revenue after run_length replications, same batch shape as X, negated
        """
        if not noise:
            raise ValueError("Noise free evaluation is not available.")
        if X.size(-1) != self.dim:
            raise ValueError("X must have size(-1) = num_servers + 1.")
        # store the old random state and set the seed
        old_state = np.random.get_state()
        np.random.seed(seed)

        # process the input and scale it back to true values
        flat_X = X.reshape(-1, self.dim)
        rates, arrival = torch.split(flat_X, [self.num_servers, 1], dim=-1)
        rates = self.rate_lb + rates * self.rate_scale
        arrival = self.arrival_lb + arrival * self.arrival_scale
        # evaluate the solutions within a for loop
        results = torch.empty((rates.size(0), 1))
        for i in range(rates.size(0)):
            inner_results = torch.empty(self.repetitions)
            for j in range(self.repetitions):
                inner_results[j] = self._simulate_system(rates[i], arrival[i])
            results[i] = torch.mean(inner_results)
        # restore the old random state
        np.random.set_state(old_state)

        # return the results in same batch shape as X, negate since we're minimizing a risk function
        return -results.reshape(*X.size()[:-1], 1)

    def _simulate_system(self, rates: Tensor, arrival: Tensor) -> Tensor:
        """
        Simulates the system with given input parameters
        :param rates: The service rates for servers
        :param arrival: The arrival rate to the first server
        :return: Revenue of the simulation run
        """
        # set up the variables and draw a number of random variables for start
        N = 1000  # the size of chunks of random variables to generate
        arrival_stream = Tensor(exponential(1/arrival.detach().numpy(), N))
        server_stream = Tensor(exponential(1/rates.detach().numpy(), (N, self.num_servers)))
        arrival_index = 0
        server_index = torch.zeros(self.num_servers, dtype=torch.long)
        server_queue = torch.zeros(self.num_servers, dtype=torch.long)  # queue accounts for customer in service as well
        blocked = torch.zeros(self.num_servers, dtype=torch.bool)  # 1 if server is blocked, 0 otherwise
        time = 0
        event_list = []  # list of events
        # Each event is a list: [event time, event type, associated server]
        # events have one of two types: arrival (0), service completion(1)

        # generate first arrival
        event = [time + arrival_stream[arrival_index], 0, 0]
        arrival_index += 1
        event_list.append(event)

        next_event = 0
        completed_count = 0

        while True:
            current_event = event_list.pop(next_event)
            # if time is up, return the revenue, otherwise, update time
            if current_event[0] > self.run_length:
                revenue = ((self.r * completed_count / time) / (self.c_0 + torch.sum(self.server_cost * rates))) \
                          - self.c_1
                return revenue
            else:
                time = current_event[0]
            server = current_event[2]

            if current_event[1]:
                # process service completion
                if server == self.num_servers - 1:
                    # if it is the final server
                    server_queue[server] -= 1
                    completed_count += 1
                else:
                    if server_queue[server + 1] == 0:
                        # if the next server is free
                        server_queue[server] -= 1
                        server_queue[server + 1] += 1
                        event = [time + server_stream[server_index[server + 1], server + 1], 1, server + 1]
                        event_list.append(event)
                        server_index[server + 1] += 1
                        # check if the server ran out of random variables
                        if server_index[server + 1] == N:
                            server_index[server + 1] = 0
                            server_stream[:, server + 1] = Tensor(exponential(1 / rates[server + 1].detach().numpy(), N))
                    elif server_queue[server + 1] <= self.capacity:
                        # next server is not free but has space in queue
                        server_queue[server] -= 1
                        server_queue[server + 1] += 1
                    else:
                        # no room to move, block the server
                        blocked[server] = 1

                # if not blocked, serve next customer
                if server_queue[server] > 0 and not blocked[server]:
                    event = [time + server_stream[server_index[server], server], 1, server]
                    event_list.append(event)
                    server_index[server] += 1
                    # check if the server ran out of random variables
                    if server_index[server] == N:
                        server_index[server] = 0
                        server_stream[:, server] = Tensor(exponential(1 / rates[server].detach().numpy(), N))

                # check for chains of blockages
                if server and server_queue[server] <= self.capacity:
                    check = server - 1
                    go = 1  # flag for the loop
                    while go:
                        if blocked[check]:
                            if server_queue[check] > 1:
                                # if there's more in the queue, serve them
                                event = [time + server_stream[server_index[check], check], 1, check]
                                event_list.append(event)
                                server_index[check] += 1
                                # check if the server ran out of random variables
                                if server_index[check] == N:
                                    server_index[check] = 0
                                    server_stream[:, check] = Tensor(exponential(1 / rates[check].detach().numpy(), N))
                            blocked[check] = 0  # unblock the server
                            server_queue[check + 1] += 1
                            server_queue[check] -= 1
                            check -= 1  # see if there's more to check, if not stop
                            if check == -1:
                                go = 0
                        else:
                            # end of blockage chain
                            go = 0

            else:
                # process arrivals
                if server_queue[0] == 0:
                    # server is empty
                    event = [time + server_stream[server_index[0], 0], 1, 0]  # service completion event
                    event_list.append(event)
                    server_index[0] += 1
                    # check if the server ran out of random variables
                    if server_index[0] == N:
                        server_index[0] = 0
                        server_stream[:, 0] = Tensor(exponential(1 / rates[0].detach().numpy(), N))

                server_queue[0] += 1  # increment queue
                event = [time + arrival_stream[arrival_index], 0, 0]  # generate next arrival
                event_list.append(event)
                arrival_index += 1
                # if we ran out of random numbers, generate new ones
                if arrival_index == N:
                    arrival_index = 0
                    arrival_stream = Tensor(exponential(1 / arrival.detach().numpy(), N))

            # select the next event as argmin of event time
            next_event = int(torch.argmin(Tensor(event_list)[:, 0]).reshape(-1)[0])

    def evaluate_true(self, X: Tensor) -> Tensor:
        raise NotImplementedError("True function evaluation is not available.")


if __name__ == "__main__":
    # for testing purposes
    from time import time
    start = time()
    line = ProductionLine(repetitions=10)
    print(line(torch.tensor([0.75, 0.5, 0.25, 0.3])))
    print('time: ', time()-start)
