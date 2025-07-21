# shared_code/chapter3_utils.py

"""
Vlad Prytula, Zooplus SE, 2025
Chapter 3 Utilities for "Foundations of Machine Learning: A Functional Analysis Perspective"

This module contains the components specific to the NeuralUCB agent. It relies
on common_utils.py for the simulation environment and other shared functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap, functional_call
from typing import List

# Import shared components
from .common_utils import FEATURE_DIM

# --- Neural Network and Agent Implementation ---
class NeuralBanditNetwork(nn.Module):
    """
    A simple MLP to act as the function approximator for the NeuralUCB agent.
    It takes a feature vector and outputs a single scalar value (predicted reward).
    """
    def __init__(self, feature_dim: int, hidden_dims: List[int] = [100, 100]):
        super(NeuralBanditNetwork, self).__init__()
        self.layers = nn.ModuleList()
        input_dim = feature_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass. Note the absence of a final sigmoid."""
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

class NeuralUCBAgent:
    """
    Implements the NeuralUCB algorithm using efficient, vectorized operations.
    """
    def __init__(self,
                 feature_dim: int,
                 hidden_dims: List[int] = [100, 100],
                 lambda_: float = 1.0,
                 alpha: float = 1.0,
                 lr: float = 0.01):
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_ = lambda_
        self.model = NeuralBanditNetwork(feature_dim, hidden_dims)
        self.p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.A = torch.eye(self.p) * self.lambda_
        self.b = torch.zeros((self.p, 1))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.last_jacobian = None

    def predict(self, feature_vectors: torch.Tensor) -> int:
        self.model.eval()

        # --- THE FIX ---
        # Perform the exploitation-term calculation in a no-grad context
        # to prevent interference with the Jacobian calculation's graph.
        with torch.no_grad():
            exploit_scores = self.model(feature_vectors).squeeze()

        A_inv = torch.inverse(self.A)
        
        # This part is the same as our previous fix.
        # We create the parameter dictionary without detaching.
        params = {k: v for k, v in self.model.named_parameters()}
        buffers = {k: v for k, v in self.model.named_buffers()}

        def compute_grad(x_single):
            # functional_call will now use parameters that require gradients
            # within a clean autograd context.
            pred = functional_call(self.model, (params, buffers), args=(x_single.unsqueeze(0),))
            grads = torch.autograd.grad(pred, params.values())
            return torch.cat([g.view(-1) for g in grads])

        J = vmap(compute_grad)(feature_vectors)
        self.last_jacobian = J

        bonus_squared = torch.einsum('ij,ji->i', J @ A_inv, J.T)
        explore_bonuses = self.alpha * torch.sqrt(torch.clamp(bonus_squared, min=0))

        ucb_scores = exploit_scores + explore_bonuses
        return torch.argmax(ucb_scores).item()

    def update(self, chosen_arm_idx: int, feature_vector: torch.Tensor, reward: float):
        # This method was correct in the previous iteration and remains unchanged.
        self.model.train()
        if self.last_jacobian is None:
            raise ValueError("predict() must be called before update() to populate the Jacobian.")
        
        g_t = self.last_jacobian[chosen_arm_idx].detach().view(-1, 1)
        self.A += g_t @ g_t.T
        self.b += reward * g_t

        self.optimizer.zero_grad()
        prediction = self.model(feature_vector.detach().unsqueeze(0))
        loss = self.loss_fn(prediction, torch.tensor([[reward]], dtype=torch.float32))
        loss.backward()
        self.optimizer.step()