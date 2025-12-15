# DRL MDP Specification for Hybrid Strategy Selection

This document defines a Markov Decision Process (MDP) for the decision module that chooses per-client strategies (GENERALIZE / PERSONALIZE / HYBRID) in the hybrid federated learning trainer.

## Entities
- **Agent:** The server-side controller (DRL variant) deciding a strategy for each participating client in a round.
- **Environment:** The federated training process, client datasets, and global model.

## State \(s_t\)
For each participating client \(i\) in round \(t\):
- `delta_loss` : pre\-train loss − post\-train loss (local val minibatches)
- `grad_norm` : average gradient norm over local steps
- `cos_sim` : cosine similarity between client update and global delta (prev→current)
- `entropy` : label entropy of the client dataset subset
- `participation` : count of how many rounds the client has trained
- `cluster_id` : optional unsupervised cluster label (set to \(-1\) for DRL in current code)

Server context (implicit / shared):
- Current global model parameters (used by env to simulate outcomes)
- Previous global model parameters (for cosine computation)
- Round index \(t\)

The controller’s per-client observation is the feature vector above; no raw data is exposed.

## Action \(a_t\)
One of three discrete strategies per client:
- GENERALIZE
- PERSONALIZE
- HYBRID

## Transition \(P(s_{t+1} | s_t, a_t)\)
- Environment applies chosen strategies to derive client\-specific start states (or eval states) and runs local training (or simulation). 
- Aggregation updates the global model. 
- Client features for the next round are recomputed from the new global / local outcomes.

## Reward \(r_t\)
Current implementation uses a **scalarized score** per client/strategy (contextual bandit style):
\[
\text{score} = w_\text{global} \cdot (\Delta \text{global acc}) + w_\text{local} \cdot (\Delta \text{local acc}) - w_\text{var} \cdot \text{variance}
\]
- Default weights: `(0.6, 0.4, 0.1)`; variance is set to 0 in current code.
- \(\Delta \text{global acc}\): simulated global accuracy change if only this client’s simulated update were applied (mixed with global at ~1/clients_per_round weight).
- \(\Delta \text{local acc}\): client validation accuracy improvement from simulate\_strategy.

In the DRLController path, these per\-strategy scores are distilled into a policy via KL to soft targets (temperatured softmax over scores). This is a contextual bandit formulation (single\-step; no delayed credit).

## Episode / Horizon
- Each training round is a decision step for the currently selected clients. 
- The horizon is the total number of FL rounds (e.g., 105). The current DRL variant does not propagate multi\-step returns; it treats steps independently (bandit).

## Policy \(\pi(a|s)\)
- Parameterized by a small MLP (StrategyControllerMLP reused): logits over the 3 actions given the client feature vector. 
- Training objective: minimize KL divergence between policy logits (softmax) and the soft targets derived from per\-strategy scores.

## Notes on Design Choices
- **Clustering:** In the DRL path, clustering is skipped; `cluster_id = -1`. For the MLP path, clustering can provide an unsupervised grouping signal.
- **Exploration:** Not explicit; temperature in score distillation provides some softness. Additional exploration (epsilon or entropy regularization) could be added.
- **Novelty:** Reward is a straightforward weighted blend of local/global improvements; no claim of novelty. The DRL path uses score distillation (contextual bandit) rather than full RL credit assignment over multiple rounds.

## Potential Extensions
- Multi\-step RL: define episode return as smoothed global accuracy improvement; use replay and advantage estimates.
- Exploration: epsilon\-greedy over actions per client; entropy regularization on policy.
- Safety: penalize actions that reduce global accuracy beyond a tolerance; or add variance term over client accuracies.
- Clustering for DRL: re\-enable clustering as an additional feature once stable.
