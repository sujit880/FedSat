import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict, namedtuple, deque
import random
from typing import List
from torch.distributions import RelaxedOneHotCategorical, Categorical
from copy import deepcopy
# ============================== Environment (unchanged) ==============================

class AggregationEnvn:    
    def __init__(self, eval_fn, build_class_prototypes, num_classes, num_clients_per_round, device, reward_case: str = "acc_align"):
        self.device = device
        self.num_clients_per_round = num_clients_per_round
        self.num_classes = num_classes
        self.evaluate_fn = eval_fn
        self.build_class_prototypes = build_class_prototypes
        self.reward_case = reward_case
        self.highest_accuracy = -100.0
        self.reset_internal()
        self.weights_vector = None
        self.best_accuracy = None
        self.best_action = None
        self.best_params = None
        self.verbose = False
        
    def reset_internal(self):
        self.global_parameters = None
        self.global_accuracy = None
        self.current_global_parameters = None
        self.clients_parameters_vectors_dict = OrderedDict()
        self.clients_parameters_vectors_list = None
        self.client_evaluation_matrices_flat = None
        self.weights_vector = None
        self.current_accuracy = None
        self.is_done = False
        self.best_accuracy = None
        self.best_params = None
        self.new_best = False
        self.client_prototypes = None
        self.best_action = None

    @torch.no_grad()
    def aggregate(self, client_solutions: List[tuple]):
        total_weight = 0.0
        model_state_dict: OrderedDict = client_solutions[0][1]
        base = [torch.zeros_like(param) for param in model_state_dict.values()]
        
        for weight, client_state_dict in client_solutions:
            total_weight += weight
            for i, param in enumerate(client_state_dict.values()):
                base_dtype = base[i].dtype
                base[i] += (weight * param).to(base_dtype)
        
        averaged_params = [param / (total_weight + 1e-12) for param in base]
        return OrderedDict(zip(model_state_dict.keys(), averaged_params))

    def soft_update(self, tau=0.2, epsilon=1e-6):
        acc_diff = self.best_accuracy - self.global_accuracy
        norm_diff = (acc_diff + 1) / 2
        alpha = tau * norm_diff / (norm_diff + epsilon)

        updated_params = {}
        for key in self.best_params.keys():
            updated_params[key] = (
                (1 - alpha) * self.global_parameters.get(key, self.best_params[key]) +
                alpha * self.best_params[key]
            )

        print(f"Returning soft updated model with accs: best={self.best_accuracy:.3f}, global={self.global_accuracy:.3f}")
        return updated_params

    def step(self, action):
        # action: Tensor of shape [K], ideally on simplex (>=0, sum=1). We'll normalize anyway for safety.
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device)

        # safety clamp + renormalize
        # action = torch.clamp(action, min=0.0)
        action = action + 1e-3
        action = action / (action.sum())

        for i, client in enumerate(self.weights_vector.keys()):
            self.weights_vector[client] = action[i].item()

        client_solutions = [
            (self.weights_vector[client], self.clients_parameters_vectors_dict[client])
            for client in self.clients_parameters_vectors_dict.keys()
        ]
        self.current_global_parameters = self.aggregate(client_solutions)
        global_proto, _ = self.build_class_prototypes(
                model_params=self.current_global_parameters,
            )
        new_accuracy, new_eval_mx = self.evaluate_fn(model_params=self.current_global_parameters)
        # denom = self.current_accuracy if self.current_accuracy and self.current_accuracy > 1e-6 else 1e-6
        # reward = (new_accuracy - (self.current_accuracy if self.current_accuracy is not None else 0.0)) / denom
        # if new_accuracy > (self.global_accuracy if self.global_accuracy is not None else -1e9):
        #     reward *= 2

        reward = compute_reward(
            new_accuracy=new_accuracy,
            new_eval_mx=new_eval_mx,
            current_accuracy=self.current_accuracy,
            global_proto=global_proto,
            client_proto_avg=self.client_prototypes,
            reward_case=self.reward_case,
        )

        self.current_accuracy = new_accuracy
        if (self.best_accuracy is None) or (self.best_accuracy < new_accuracy):
            self.new_best = True
            if self.verbose: print(f"Got new best accuracy: {new_accuracy}!\nAction:", ', '.join(f'{a:.3f}' for a in action.tolist()))
            self.best_accuracy = new_accuracy
            self.best_action = deepcopy(action)
            self.best_params = OrderedDict({k: v.clone() for k, v in self.current_global_parameters.items()})
            for i, key in enumerate(self.weights_vector):
                self.weights_vector[key] = self.best_action[i].clone().detach()

        if self.global_accuracy is None:
            self.global_accuracy = -1e9
        self.is_done = new_accuracy > min(0.9999, self.global_accuracy * 3)
        new_eval_mx_flat = new_eval_mx.flatten().unsqueeze(0)
        next_state = torch.cat([self.client_evaluation_matrices_flat, new_eval_mx_flat], dim=0).unsqueeze(0)
        
        if self.highest_accuracy < new_accuracy:
            self.highest_accuracy = new_accuracy
        return next_state, reward, new_accuracy, self.is_done

    def reset(self, parameters_vectors_dict):
        self.reset_internal()
        client_accuracies, client_eval_matrices = [], []
        global_proto_sums = {}  # class_id -> Tensor sum
        global_counts = {}      # class_id -> int count

        for client, param in parameters_vectors_dict.items():
            acc, eval_matrix = self.evaluate_fn(model_params=param, num_batches=65, name=f'client({client}): ', render=False)
            client_accuracies.append(acc)
            client_eval_matrices.append(eval_matrix)
            # 4) Class-wise prototypes for this client (mean features per class)
            client_proto, proto_count = self.build_class_prototypes( model_params=param)

            for cid, proto in client_proto.items():
                if cid not in global_proto_sums:
                    global_proto_sums[cid] = proto * proto_count[cid]
                    global_counts[cid] = proto_count[cid]
                else:
                    global_proto_sums[cid] += proto * proto_count[cid]
                    global_counts[cid] += proto_count[cid]
        self.client_prototypes = {cid: (global_proto_sums[cid] / global_counts[cid])
                                for cid in global_proto_sums.keys() if global_counts[cid] > 0}

        if self.verbose: print('Clients Accuracy:', ', '.join(f'{acc:.3f}' for acc in client_accuracies))

        self.clients_parameters_vectors_dict = parameters_vectors_dict
        self.weights_vector = {client: 1 for client in parameters_vectors_dict}
        self.best_action = torch.tensor(list(self.weights_vector.values()))
        client_solutions = [(1, param) for param in parameters_vectors_dict.values()]
        self.current_global_parameters = self.aggregate(client_solutions)
        self.current_accuracy, new_eval_mx = self.evaluate_fn(model_params=self.current_global_parameters)
        if self.highest_accuracy < self.current_accuracy:
            self.highest_accuracy = self.current_accuracy

        if self.verbose: print(f"Reset global accuracy: {self.current_accuracy}! for Action: all 1")
        self.global_parameters = OrderedDict({k: v.clone() for k, v in self.current_global_parameters.items()})
        self.global_accuracy = self.current_accuracy
        self.best_accuracy = self.current_accuracy
        self.best_params = OrderedDict({k: v.clone() for k, v in self.current_global_parameters.items()})
        
        self.client_evaluation_matrices_flat = torch.stack([em.flatten() for em in client_eval_matrices])
        new_eval_mx_flat = new_eval_mx.flatten().unsqueeze(0)
        self.is_done = self.global_accuracy > 0.9999
        initial_state = torch.cat([self.client_evaluation_matrices_flat, new_eval_mx_flat], dim=0).unsqueeze(0)
        return initial_state, self.is_done


# ============================== Networks (shared) ==============================

class ActorSAC(nn.Module):
    """
    SAC Policy on the simplex using Gumbel-Softmax (Concrete) distribution.
    The network outputs logits; actions are sampled as a ~ RelaxedOneHotCategorical(temperature, logits),
    which yields a differentiable point on the probability simplex.

    Theory (SAC):
      Maximize:  J_pi = E_{s~D, a~pi}[ alpha * log pi(a|s) - min_i Q_i(s,a) ]
      with temperature alpha balancing reward and entropy.

      We use the reparameterization trick via Concrete/Gumbel-Softmax:
        g_k ~ Gumbel(0,1);  y_k = softmax((logits_k + g_k)/tau)
      which provides rsample() and log_prob().
    """
    def __init__(self, input_channels, input_dim, action_dim, hidden_size=128, temperature=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_dim)
        self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.float32))

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)
        return logits  # raw logits; distribution built outside

    def dist(self, state):
        logits = self.forward(state)
        return RelaxedOneHotCategorical(self.temperature, logits=logits, validate_args=False), logits


class CriticSAC(nn.Module):
    """
    Twin Q network (use two copies of this class). Each takes (state, action) and outputs Q(s,a).
    """
    def __init__(self, input_channels, input_dim, action_dim, hidden_size=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * input_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


# ============================== Replay Buffer ==============================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        return Experience(*zip(*experiences))

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


# ============================== SAC Agent ==============================

class SACAgentR:
    """
    Soft Actor-Critic (SAC) with a Concrete (Gumbel-Softmax) policy over a K-simplex.

    Theoretical core:
      Target for critics:
        y = r + gamma * (1 - d) * [ min_i Q_target_i(s', a') - alpha * log pi(a'|s') ]
      Critic loss:
        L_Q = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
      Actor loss:
        L_pi = E_s[ alpha * log pi(a|s) - min(Q1,Q2)(s,a) ]
      Temperature (entropy) loss (auto-tuning):
        L_alpha = E_s,a[ - alpha * (log pi(a|s) + H_target) ]
        with H_target = -K
    """
    def __init__(self,
                 eval_fn,
                 build_class_prototypes,
                 num_classes,
                 device,
                 num_clients_per_round,
                 input_channels,
                 input_dim,
                 action_dim,
                 hidden_size=128,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 policy_temperature=0.5,
                 target_entropy=None,
                 reward_case: str = "acc_align"):
        self.device = device
        self.env = AggregationEnvn(
            eval_fn,
            build_class_prototypes,
            num_classes,
            num_clients_per_round,
            device,
            reward_case=reward_case,
        )

        # Actor (Concrete/Gumbel-Softmax policy)
        self.actor = ActorSAC(input_channels, input_dim, action_dim, hidden_size, temperature=policy_temperature).to(device)

        # Twin critics + targets
        self.critic1 = CriticSAC(input_channels, input_dim, action_dim, hidden_size).to(device)
        self.critic2 = CriticSAC(input_channels, input_dim, action_dim, hidden_size).to(device)
        self.critic1_target = CriticSAC(input_channels, input_dim, action_dim, hidden_size).to(device)
        self.critic2_target = CriticSAC(input_channels, input_dim, action_dim, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optims
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Entropy temperature alpha (auto-tuned)
        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)

        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(500, self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def eval_action(self, state):
        """
        Deterministic eval: use categorical probs (softmax of logits), which lie on the simplex.
        """
        state = state.to(self.device)
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        # Return first (batch size=1)
        return probs[0]

    def get_action(self, state):
        """
        Stochastic action for data collection: rsample from Concrete distribution, already on simplex.
        """
        state = state.to(self.device)
        dist, logits = self.actor.dist(state)
        a = dist.rsample()                    # shape [B, K], simplex
        # For env, we pass a[0]
        return a[0].detach()

    def step(self, action):
        return self.env.step(action)

    def reset(self, parameters_vectors_dict):
        return self.env.reset(parameters_vectors_dict)

    def soft_update(self, source, target):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)

        states = torch.cat(batch.state).to(self.device)          # [B, C, L]
        actions = torch.stack(batch.action).to(self.device)      # [B, K] on simplex
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        next_states = torch.cat(batch.next_state).to(self.device)                                   # [B, C, L]
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)      # [B,1]

        # -------- Target Q computation --------
        with torch.no_grad():
            next_dist, _ = self.actor.dist(next_states)
            next_actions = next_dist.rsample()                                # [B,K], simplex
            # numeric guard
            eps = 1e-8
            next_actions = next_actions.clamp_min(eps)
            next_actions = next_actions / next_actions.sum(dim=-1, keepdim=True).clamp_min(eps)
            next_log_pi = next_dist.log_prob(next_actions).unsqueeze(1)       # [B,1]
            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            target_q = rewards + (1 - dones) * self.gamma * (min_q_target - self.alpha.detach() * next_log_pi)
        # -------- Critic update --------
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # -------- Actor update --------
        dist, _ = self.actor.dist(states)
        new_actions = dist.rsample()
        # numeric guard
        eps = 1e-8
        new_actions = new_actions.clamp_min(eps)
        new_actions = new_actions / new_actions.sum(dim=-1, keepdim=True).clamp_min(eps)
        log_pi = dist.log_prob(new_actions).unsqueeze(1)         # [B,1]
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # -------- Temperature (alpha) update --------
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # -------- Soft update target critics --------
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        # (Optional) return losses for logging
        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

# ============================== Usage Notes ==============================
# - Instantiate SACAgent instead of DDPGAgent with the same env args.
# - The agent produces simplex actions compatible with your aggregation weights.
# - If you prefer less “peaky” actions initially, increase policy_temperature (e.g., 0.7–1.0).
# - Target entropy defaults to -K; you can tune it (e.g., -0.5*K for lower entropy).


def _per_class_recall_from_confmat(confmat: torch.Tensor) -> torch.Tensor:
    """
    confmat: (C, C) with rows = true class, cols = predicted class.
    Returns per-class recall vector of shape (C,).
    """
    if isinstance(confmat, (list, tuple)):
        confmat = torch.tensor(confmat, dtype=torch.float32)
    elif not torch.is_tensor(confmat):
        confmat = torch.as_tensor(confmat, dtype=torch.float32)

    tp = torch.diag(confmat)                               # (C,)
    per_class_total = confmat.sum(dim=1).clamp_min(1e-6)   # avoid div-by-zero
    recall = (tp / per_class_total).to(torch.float32)
    return recall

def _proto_alignment_score(global_proto_dict, client_proto_dict):
    """
    Mean cosine similarity across classes present in BOTH dicts.
    Returns a scalar in [0, 1] by mapping cosine from [-1, 1] -> [0, 1].
    """
    common = sorted(set(global_proto_dict.keys()) & set(client_proto_dict.keys()))
    if len(common) == 0:
        return 0.5  # neutral if no overlap

    sims = []
    for c in common:
        if c not in global_proto_dict or c not in client_proto_dict: continue
        g = global_proto_dict[c]
        p = client_proto_dict[c]
        # ensure tensors on same device & 1D
        if g.dim() > 1:
            g = g.view(-1)
        if p.dim() > 1:
            p = p.view(-1)

        # move to same device for cosine
        dev = g.device if g.is_cuda else p.device
        g = g.to(dev).float()
        p = p.to(dev).float()

        cos = F.cosine_similarity(g.unsqueeze(0), p.unsqueeze(0), dim=1).item()
        sims.append(cos)

    mean_cos = float(sum(sims) / len(sims))
    # map from [-1, 1] → [0, 1]
    return 0.5 * (mean_cos + 1.0)

def compute_reward(
    new_accuracy: float,
    new_eval_mx,                       # confusion matrix or anything convertible to (C,C)
    current_accuracy: float | None,
    global_proto: dict,                # {class_id: tensor(D,)}
    client_proto_avg: dict,            # {class_id: tensor(D,)}
    w_acc: float = 1.0,
    w_align: float = 0.5,
    w_fair: float = 0.5,
    reward_case: str = "acc_align",
):
    """
    Returns scalar reward combining:
      - accuracy improvement (normalized by current_accuracy),
      - prototype alignment (mean cosine similarity),
      - fairness (1 - std_dev of per-class recall).
    """
    # --- Accuracy improvement term (your original logic, cleaned) ---
    acc_score = -2 if (new_accuracy - current_accuracy)<0 else 100*(new_accuracy - current_accuracy) # in [-2, 100]

    # --- Prototype alignment term ---
    align = _proto_alignment_score(global_proto, client_proto_avg)  # in [0, 1]

    # --- Fairness term: 1 - std(recall) (higher is better, ∈ (0,1]) ---
    try:
        recalls = _per_class_recall_from_confmat(new_eval_mx)
        if recalls.numel() > 1:
            fairness = float(1.0 - torch.std(recalls).clamp_max(1.0).item())
        else:
            fairness = 0.5  # neutral when only one class
    except Exception:
        fairness = 0.5      # neutral if eval matrix is not a confmat

    # --- Combine (weights are tunable hyperparams) ---
    case_key = (reward_case or "acc_align").lower()

    if case_key in {"acc_align_fair", "all", "full"}:
        reward = w_acc * acc_score + w_align * align + w_fair * fairness
    elif case_key in {"acc_align", "acc_align_only"}:
        reward = w_acc * acc_score + w_align * align
    elif case_key in {"acc_fair", "acc_fairness"}:
        reward = w_acc * acc_score + w_fair * fairness
    elif case_key in {"align_fair", "align_fairness"}:
        reward = w_align * align + w_fair * fairness
    elif case_key in {"align", "alignment"}:
        reward = align
    elif case_key in {"fair", "fairness"}:
        reward = fairness
    elif case_key in {"acc", "accuracy", "acc_score"}:
        reward = acc_score
    elif case_key in {"align_plus_fair", "alignfair", "align_plus_fairness"}:
        reward = align + fairness
    else:
        raise ValueError(f"Unknown reward_case '{reward_case}'.")

    return float(reward)
