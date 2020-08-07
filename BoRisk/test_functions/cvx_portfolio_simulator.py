import os
import sys
import subprocess
import warnings
import numpy as np
import pandas as pd
import cvxportfolio as cp
from typing import Optional
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction


class CVXPortfolioSimulator(SyntheticTestFunction):
    r"""
    """

    _optimizers = None

    def __init__(
        self,
        dim: int = 5,
        experiment_id: Optional[str] = "__",
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = dim
        self.experiment_id = experiment_id
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.datadir = self.script_dir + "/portfolio_test_data/"
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        if X.dim() < 3:
            X = X.unsqueeze(-2)
            flag = True
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([1]))
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                gamma_risk = float(999.9 * X[i, j, 0] + 0.1)
                gamma_tcost = float(2.5 * X[i, j, 1] + 5.5)
                gamma_holding = float(99.9 * X[i, j, 2] + 0.1)
                bid_ask_spread = float(0.0099 * X[i, j, 3] + 0.0001)
                borrow_cost = float(0.0009 * X[i, j, 4] + 0.0001)
                output[i, j, 0] = self._evaluate_with_unscaled_input(
                    gamma_risk, gamma_tcost, gamma_holding, bid_ask_spread, borrow_cost
                )
        if flag:
            output = output.squeeze(-2)
        return output

    def _evaluate_with_unscaled_input(
        self, gamma_risk, gamma_tcost, gamma_holding, bid_ask_spread, borrow_cost
    ):

        # Set up requirements for portfolio simulation
        self.experiment_id = str(int(torch.randint(1, 100000, torch.Size([1]))))
        sigmas = pd.read_csv(
            self.datadir + "sigmas.csv.gz", index_col=0, parse_dates=[0]
        ).iloc[:, :-1]
        returns = pd.read_csv(
            self.datadir + "returns.csv.gz", index_col=0, parse_dates=[0]
        )
        volumes = pd.read_csv(
            self.datadir + "volumes.csv.gz", index_col=0, parse_dates=[0]
        ).iloc[:, :-1]

        w_b = pd.Series(index=returns.columns, data=1)
        w_b.USDOLLAR = 0.0
        w_b /= sum(w_b)

        start_t = "2012-01-01"
        end_t = "2016-12-31"

        simulated_tcost = cp.TcostModel(
            half_spread=bid_ask_spread / 2.0,
            nonlin_coeff=1.0,
            sigma=sigmas,
            volume=volumes,
        )
        simulated_hcost = cp.HcostModel(borrow_costs=borrow_cost)
        simulator = cp.MarketSimulator(
            returns,
            costs=[simulated_tcost, simulated_hcost],
            market_volumes=volumes,
            cash_key="USDOLLAR",
        )

        return_estimate = pd.read_csv(
            self.datadir + "return_estimate.csv.gz", index_col=0, parse_dates=[0]
        ).dropna()
        volume_estimate = pd.read_csv(
            self.datadir + "volume_estimate.csv.gz", index_col=0, parse_dates=[0]
        ).dropna()
        sigma_estimate = pd.read_csv(
            self.datadir + "sigma_estimate.csv.gz", index_col=0, parse_dates=[0]
        ).dropna()

        optimization_tcost = cp.TcostModel(
            half_spread=bid_ask_spread / 2.0,
            nonlin_coeff=1.0,
            sigma=sigma_estimate,
            volume=volume_estimate,
        )
        optimization_hcost = cp.HcostModel(borrow_costs=borrow_cost)

        copy_of_risk_model_name = "risk_model_" + self.experiment_id + ".h5"
        subprocess.call(
            [
                "bash",
                self.script_dir + "/make_copy_of_risk_model.sh",
                self.datadir + "risk_model.h5",
                self.datadir + copy_of_risk_model_name,
            ]
        )
        risk_data = pd.HDFStore(self.datadir + copy_of_risk_model_name)
        risk_model = cp.FactorModelSigma(
            risk_data.exposures, risk_data.factor_sigma, risk_data.idyos
        )

        results = {}
        policies = {}
        policies[(gamma_risk, gamma_tcost, gamma_holding)] = cp.SinglePeriodOpt(
            return_estimate,
            [
                gamma_risk * risk_model,
                gamma_tcost * optimization_tcost,
                gamma_holding * optimization_hcost,
            ],
            [cp.LeverageLimit(3)],
        )
        warnings.filterwarnings("ignore")
        results.update(
            dict(
                zip(
                    policies.keys(),
                    simulator.run_multiple_backtest(
                        1e8 * w_b,
                        start_time=start_t,
                        end_time=end_t,
                        policies=policies.values(),
                        parallel=True,
                    ),
                )
            )
        )
        results_df = pd.DataFrame()
        results_df[r"$\gamma^\mathrm{risk}$"] = [el[0] for el in results.keys()]
        results_df[r"$\gamma^\mathrm{trade}$"] = [el[1] for el in results.keys()]
        results_df[r"$\gamma^\mathrm{hold}$"] = ["%g" % el[2] for el in results.keys()]
        results_df["Return"] = [results[k].excess_returns for k in results.keys()]
        for k in results.keys():
            returns = results[k].excess_returns.to_numpy()
        returns = returns[:-1]
        subprocess.call(
            [
                "bash",
                self.script_dir + "/delete_copy_of_risk_model.sh",
                self.datadir + copy_of_risk_model_name,
            ]
        )
        return np.mean(returns) * 100 * 250
