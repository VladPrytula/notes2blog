# Modern Recommendation Systems: Notes and Code

This repository contains my personal notes and code explorations on modern recommendation systems. It's an ongoing effort to break down and organize the core concepts and advanced techniques in the field. Think of it as a working document rather than a finished textbook.

### Motivation and Scope

Recommendation systems are fascinating, blending insights into human behavior with statistical learning and high-performance computing. My goal with these notes is to methodically explore their mathematical foundations—from linear algebra and optimization to probability—and connect them directly to the sophisticated systems used today.

### Approach

Each chapter follows a structured approach designed to build a solid understanding:

1.  **Intuition First:** Every topic starts with *why* it's important, using real-world examples and analogies to make abstract concepts clear.

2.  **Rigor in Detail:** Once the intuition is established, we delve into the formal mathematical definitions, theorems, and derivations. All terms are defined, and derivations are shown step-by-step.

3.  **Application Through Code:** Finally, theory is applied through code. PyTorch examples bridge the gap between abstract math and practical implementation. Each code block is a key part of the explanation, not just an appendix.


### Chapter Overview: The Journey So Far

Our exploration is structured as a logical progression, with each chapter building upon the insights and limitations of the last.

*   **Chapter 1: Personalization in Practice—A Batched Learning Baseline**
    We begin by establishing a strong, yet flawed, foundation. We construct a realistic e-commerce simulation and use it to train a standard deep learning recommender based on the "Embedding + MLP" architecture. A critical analysis reveals the inherent weaknesses of this static, batched learning approach, namely its inability to adapt to new information or intelligently explore its environment. This provides the crucial motivation for moving toward online learning.

*   **Chapter 2: The Adaptive Recommender—Contextual Bandits in Action**
    Here, we address the static nature of our baseline by introducing the **explore-exploit dilemma**. We formalize and implement the **Linear Upper Confidence Bound (LinUCB)** algorithm, a classic contextual bandit approach. Through a head-to-head simulation, particularly one involving a "market shock," we demonstrate the profound superiority of online, adaptive learning in a non-stationary world. However, we also identify its architectural limitation: its use of *disjoint models* prevents it from generalizing across the action space.

*   **Chapter 3: Neural Contextual Bandits—Generalization Through Shared Representation**
    This chapter represents a necessary synthesis, combining the adaptive nature of bandits with the powerful representation learning of deep networks. We introduce the **NeuralUCB** algorithm, which uses a single, shared neural network to generalize across all items. A significant portion of this chapter is dedicated to a rigorous analysis of computational efficiency, dissecting the "K-Pass Problem" of naive gradient computation and developing a vastly more efficient, vectorized solution via the Jacobian matrix. This serves as a deep dive into the practical marriage of theory and high-performance computing.

