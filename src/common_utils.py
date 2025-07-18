# shared_code/common_utils.py

"""
Vlad Prytula, Zooplus SE, 2025
Common Utilities for "Foundations of Machine Learning: A Functional Analysis Perspective"

This module contains components shared across multiple chapters and experiments, including:
1.  ZooplusSimulator: The simulated environment.
2.  Feature Engineering: The create_feature_vectors function.
3.  Plotting: The plot_results function.
4.  Simulation Runner: The generic run_simulation loop.
5.  Constants: Shared constants like feature dimensions.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# --- Global Constants ---
PERSONA_COLS = ['new_puppy_parent', 'cat_connoisseur', 'fish_hobbyist', 'senior_dog_owner']
PRODUCT_COLS = ['Fish Supplies', 'Cat Food', 'Dog Food', 'Dog Toy', 'Cat Toy']
FEATURE_DIM = len(PERSONA_COLS) + len(PRODUCT_COLS)

# --- Class Definitions (ZooplusSimulator) ---
class ZooplusSimulator:
    """
    A simulated environment for the Zooplus recommendation problem.

    This class manages:
    1. A product catalog with features (category).
    2. A set of user personas with distinct preferences.
    3. A stochastic reward function to simulate user clicks (CTR).
    """
    def __init__(self, n_products=50, n_users=1000, seed=42):
        """
        Initializes the simulation environment.
        
        Args:
            n_products (int): The total number of products in the catalog.
            n_users (int): The total number of unique users in the simulation.
            seed (int): Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.n_products = n_products
        self.n_users = n_users
        
        # 1. Create the Product Catalog
        self.products = self._create_product_catalog()
        
        # 2. Create User Personas and assign each of the n_users to a persona
        self.personas = self._create_user_personas()
        self.user_to_persona_map = self._assign_users_to_personas()

    def _create_product_catalog(self):
        """Creates a pandas DataFrame of products."""
        product_ids = range(self.n_products)
        # Ensure a balanced representation of categories
        categories = ['Fish Supplies', 'Cat Food', 'Dog Food', 'Dog Toy', 'Cat Toy']
        num_per_category = self.n_products // len(categories)
        cat_list = []
        for cat in categories:
            cat_list.extend([cat] * num_per_category)
        # Fill the remainder, if any
        cat_list.extend(self.rng.choice(categories, self.n_products - len(cat_list)))
        
        product_data = {
            'product_id': product_ids,
            'category': self.rng.permutation(cat_list) # Shuffle categories
        }
        return pd.DataFrame(product_data).set_index('product_id')

    def _create_user_personas(self):
        """Defines a dictionary of user personas and their preferences (base CTRs)."""
        return {
            'new_puppy_parent': {'Dog Food': 0.40, 'Dog Toy': 0.50, 'Cat Food': 0.10, 'Cat Toy': 0.05, 'Fish Supplies': 0.02},
            'cat_connoisseur':  {'Dog Food': 0.05, 'Dog Toy': 0.02, 'Cat Food': 0.55, 'Cat Toy': 0.45, 'Fish Supplies': 0.05},
            'budget_shopper':   {'Dog Food': 0.25, 'Dog Toy': 0.15, 'Cat Food': 0.40, 'Cat Toy': 0.20, 'Fish Supplies': 0.20},
            'fish_hobbyist':    {'Dog Food': 0.02, 'Dog Toy': 0.02, 'Cat Food': 0.10, 'Cat Toy': 0.08, 'Fish Supplies': 0.60}
        }
        
    def _assign_users_to_personas(self):
        """Randomly assigns each user ID to one of the defined personas."""
        persona_names = list(self.personas.keys())
        return {user_id: self.rng.choice(persona_names) for user_id in range(self.n_users)}

    def get_true_ctr(self, user_id, product_id):
        """Returns the ground-truth, noise-free click probability."""
        if user_id not in self.user_to_persona_map or product_id not in self.products.index:
            return 0.0
            
        persona_name = self.user_to_persona_map[user_id]
        persona_prefs = self.personas[persona_name]
        
        product_category = self.products.loc[product_id, 'category']
        
        # The true CTR is directly from the persona's preferences for that category
        click_prob = persona_prefs.get(product_category, 0.01) # Default for unknown categories
        return click_prob

    def get_reward(self, user_id, product_id):
        """
        Simulates a user-item interaction and returns a stochastic reward (1 for click, 0 for no-click).
        """
        click_prob = self.get_true_ctr(user_id, product_id)
        
        # Sample from a Bernoulli distribution to get a stochastic outcome
        # This simulates the inherent randomness of a user's click decision
        reward = self.rng.binomial(1, click_prob)
        return reward

    def get_random_user(self):
        """Returns a random user_id from the population."""
        return self.rng.integers(0, self.n_users)

# --- Helper Functions ---
def create_feature_vectors(user_persona: str, products_df: pd.DataFrame) -> torch.Tensor:
    """Creates one-hot encoded feature vectors for all products for a given user."""
    # This is a simplified example. A real system would have richer features.
    persona_features = pd.get_dummies([user_persona])
    product_features = pd.get_dummies(products_df['category'])
    
    # Ensure all possible columns are present
    all_persona_cols = ['new_puppy_parent', 'cat_connoisseur', 'fish_hobbyist', 'senior_dog_owner']
    all_product_cols = ['Fish Supplies', 'Cat Food', 'Dog Food', 'Dog Toy', 'Cat Toy']
    
    persona_features = persona_features.reindex(columns=all_persona_cols, fill_value=0)
    product_features = product_features.reindex(columns=all_product_cols, fill_value=0)
    
    n_products = len(products_df)
    # Repeat the user features for each product
    user_part = np.tile(persona_features.values, (n_products, 1))
    product_part = product_features.values
    
    # Concatenate to form interaction features
    interaction_features = np.concatenate([user_part, product_part], axis=1)
    
    return torch.from_numpy(interaction_features).float()

def run_simulation(agent: Any, simulator: ZooplusSimulator, num_interactions: int) -> List[int]:
    """
    Runs a generic online learning simulation loop.
    
    This function is designed to work with any agent that has a `predict` method
    that takes a tensor of feature vectors and an `update` method.
    """
    print(f"Running simulation for {agent.__class__.__name__}...")
    rewards_history = []
    
    for _ in tqdm(range(num_interactions)):
        user_id = simulator.get_random_user()
        user_persona = simulator.user_to_persona_map[user_id]
        
        all_feature_vectors = create_feature_vectors(user_persona, simulator.products)
        chosen_arm_idx = agent.predict(all_feature_vectors)
        reward = simulator.get_reward(user_id, product_id=chosen_arm_idx)
        
        chosen_feature_vector = all_feature_vectors[chosen_arm_idx]
        agent.update(chosen_arm_idx, chosen_feature_vector, reward)
        
        rewards_history.append(reward)
        
    print(f"Simulation complete. Average reward: {np.mean(rewards_history):.4f}")
    return rewards_history

def plot_results(results: Dict[str, List[int]], window_size: int = 500):
    """
    Plots the moving average of rewards for different agents.
    
    Args:
        results (Dict[str, List[int]]): A dictionary mapping agent names to their reward histories.
        window_size (int): The size of the window for the moving average.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, rewards in results.items():
        # Calculate the moving average to smooth the plot
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        ax.plot(moving_avg, label=f'{name} (Moving Avg)')

    ax.set_title('Agent Performance: Cumulative Click-Through Rate', fontsize=16)
    ax.set_xlabel('Number of Interactions (Time Steps)', fontsize=12)
    ax.set_ylabel(f'Click-Through Rate (Moving Average over {window_size} steps)', fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()