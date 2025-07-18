# shared_code/chapter2_utils.py

"""
Vlad Prytula, Zooplus SE, 2025
Chapter 2 Utilities for "Foundations of Machine Learning: A Functional Analysis Perspective"

This module contains the implementation for the Disjoint Linear UCB Agent (LinUCBAgent)
from Chapter 2.
"""

import numpy as np
import torch
from typing import List

class LinUCBAgent:
    """
    Implements the Disjoint LinUCB algorithm from Chapter 2.

    This agent maintains a separate set of linear model parameters (A_a, b_a)
    for each arm 'a'. It assumes the expected reward is a linear function of
    the context features.
    """
    def __init__(self, n_arms: int, feature_dim: int, lambda_: float = 1.0, alpha: float = 1.0):
        """
        Initializes the agent's parameters.

        Args:
            n_arms (int): The number of arms (actions) available.
            feature_dim (int): Dimensionality of the input context features.
            lambda_ (float): Regularization parameter for the A matrix (ridge regression).
            alpha (float): Exploration-exploitation trade-off parameter.
        """
        self.n_arms = n_arms
        self.feature_dim = feature_dim
        self.lambda_ = lambda_
        self.alpha = alpha

        # Disjoint models: A list of matrices and vectors, one for each arm.
        # A_a is a (d x d) matrix for arm 'a'.
        self.A = [self.lambda_ * np.identity(self.feature_dim) for _ in range(self.n_arms)]
        # b_a is a (d x 1) vector for arm 'a'.
        self.b = [np.zeros((self.feature_dim, 1)) for _ in range(self.n_arms)]

    def predict(self, feature_vectors: torch.Tensor) -> int:
        """
        Selects an arm based on the LinUCB scoring rule.

        Note: This method accepts a PyTorch tensor for compatibility with the
        simulation loop but performs its calculations in NumPy.

        Args:
            feature_vectors (torch.Tensor): A tensor of shape (n_arms, feature_dim)
                                            containing the context for each arm.
        Returns:
            int: The index of the chosen arm.
        """
        # Convert torch tensor to numpy for internal calculations
        feature_vectors_np = feature_vectors.numpy()
        
        scores = np.zeros(self.n_arms)
        
        for arm_idx in range(self.n_arms):
            x = feature_vectors_np[arm_idx].reshape(-1, 1) # Shape (d, 1)
            
            # Solve for theta_hat = A_inv * b using np.linalg.solve for
            # better numerical stability than direct inversion.
            # A_inv = np.linalg.inv(self.A[arm_idx])
            # theta_hat = A_inv @ self.b[arm_idx]
            theta_hat = np.linalg.solve(self.A[arm_idx], self.b[arm_idx])
            
            # Exploitation term: x^T * theta_hat
            exploit_term = (x.T @ theta_hat).item()
            
            # Exploration term: alpha * sqrt(x^T * A_inv * x)
            A_inv = np.linalg.inv(self.A[arm_idx]) # Need inverse for the confidence bound
            explore_term = self.alpha * np.sqrt(x.T @ A_inv @ x).item()
            
            scores[arm_idx] = exploit_term + explore_term
            
        # Return the arm with the highest score
        return np.argmax(scores)

    def update(self, chosen_arm_idx: int, feature_vector: torch.Tensor, reward: float):
        """
        Updates the parameters for the chosen arm.

        Args:
            chosen_arm_idx (int): The index of the arm that was played.
            feature_vector (torch.Tensor): The feature vector for the CHOSEN arm.
            reward (float): The observed reward.
        """
        # Convert torch tensor to numpy
        x = feature_vector.numpy().reshape(-1, 1) # Shape (d, 1)
        
        # Update only the parameters for the arm that was chosen
        self.A[chosen_arm_idx] += x @ x.T
        self.b[chosen_arm_idx] += reward * x