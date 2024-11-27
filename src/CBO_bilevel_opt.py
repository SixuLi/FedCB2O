import math

import numpy as np
import copy
import os
import logging

from objective_func import L, G, L_2d, contrained_OPT_obj

class CBO_bilevel:
    def __init__(self, train_init, args):
        self.train_init = train_init
        self.args = args
        self.N = args.N
        self.M = args.M
        self.T = args.T
        self.d = args.d
        self.Lambda = args.Lambda
        self.Sigma = args.Sigma
        self.Gamma = args.Gamma
        self.Alpha = args.Alpha
        self.Beta = args.Beta
        self.seed = args.seed
        self.thetas_position = []

        self.initialization()

    def initialization(self):
        np.random.seed(self.seed)

        # Initialize the parameter for each agent
        if self.d == 1:
            self.agents = np.random.uniform(-10,10, size=self.N)
        elif self.d > 1:
            self.agents = np.random.uniform(-10,10, (self.N, self.d))
        self.agents_idx = np.arange(0,self.N, 1)

    def run_optimization(self):
        self.t = 0
        self.I_beta_position = []
        self.consensus_point_position = []
        while self.t < self.T:
            self.thetas_position.append(copy.deepcopy(self.agents))
            if self.t % 1000 == 0:
                logging.info('Training epoch {}'.format(self.t))

            # Randomly pick M agents to attend the optimization in round t
            A_t = np.random.choice(self.agents_idx, self.M, replace=False)
            thetas = copy.deepcopy((np.take(self.agents, A_t, axis=0)))
            # print('Shape of thetas:', thetas.shape)

            self.consensus_point = self.calculate_consensus_point(thetas)
            self.consensus_point_position.append(self.consensus_point)

            # Update agent position
            if self.d == 1:
                z_p = np.random.normal(loc=0, scale=1, size=thetas.shape)
                thetas = thetas - self.Lambda * self.Gamma * (thetas - self.consensus_point) \
                            + self.Sigma * math.sqrt(self.Gamma) * (thetas - self.consensus_point) * z_p
            elif self.d > 1:
                z_p = np.zeros((np.size(thetas, 0), self.d, self.d))
                z_p[:, np.arange(self.d), np.arange(self.d)] = np.random.multivariate_normal(mean=np.zeros(self.d),
                                                                                             cov=np.eye(self.d),
                                                                                             size=np.size(thetas,0))
                thetas = thetas - self.Lambda * self.Gamma * (thetas - self.consensus_point) \
                               + self.Sigma * math.sqrt(self.Gamma) \
                               * np.matmul(np.expand_dims(thetas - self.consensus_point, axis=1), z_p).squeeze(axis=1)

            self.agents[A_t] = thetas
            self.t += 1

    def calculate_consensus_point(self, thetas):
        # Get the quantile of participated agents
        I_beta = self.get_quantile(thetas)
        self.I_beta_position.append(I_beta)

        # Calculate weights for agents within the quantile
        if self.args.opt_type == 'unconstrained':
            mu = -self.Alpha * G(I_beta)
        elif self.args.opt_type == 'constrained':
            constrained_opt_obj = contrained_OPT_obj(d=self.d)
            mu = -self.Alpha * constrained_opt_obj.simple_2d(I_beta)
        weights = np.exp(mu.astype(float)) / np.sum(np.exp(mu.astype(float)))

        # print('Shape of I_beta', I_beta.shape)
        # print('Shape of weights', weights.shape)

        return np.matmul(I_beta.T, weights)

    def get_quantile(self, thetas):
        if self.args.opt_type == 'unconstrained':
            if self.d == 1:
                Q_beta = np.quantile(L(thetas), self.Beta)
                return thetas[np.where(L(thetas) <= Q_beta)]
            elif self.d == 2:
                # print('Objective function value=', L_2d(thetas))
                Q_beta = np.quantile(L_2d(thetas), self.Beta)
                return thetas[np.where(L_2d(thetas) <= Q_beta)]
        elif self.args.opt_type == 'constrained':
            constrained_opt_obj = contrained_OPT_obj(d=self.d)
            Q_beta = np.quantile(constrained_opt_obj.ellipse_2d(thetas), self.Beta)
            return thetas[np.where(constrained_opt_obj.ellipse_2d(thetas) <= Q_beta)]





