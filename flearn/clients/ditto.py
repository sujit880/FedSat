import torch
from copy import deepcopy
from typing import Tuple

from flearn.clients.client import BaseClient
from flearn.utils.torch_utils import graph_size


class DittoClient(BaseClient):
    """Client implementation for **Ditto** (Li *et al.*, NeurIPS 2021).

    Ditto maintains **two** sets of parameters on every client:

    1. A *personalised* model `p_i` that is regularised towards the global
       model `w` via a proximal term (Moreau envelope):

           min_{p_i}\; f_i(p_i) + (λ/2)‖p_i − w‖².

    2. A *global* model `w_i` that is updated *without* the proximal term and
       returned to the server for FedAvg aggregation.

    Only the *global* model parameters are sent upstream; the personalised
    parameters are cached locally and used for evaluation or fine‑tuning in
    the next round.  This differs from **FedProx**, where the prox‑regularised
    model itself is uploaded to the server.
    """

    # ------------------------------------------------------------------
    # Initialisation ----------------------------------------------------
    # ------------------------------------------------------------------
    def init_client_specific_params(self, lam: float, **kwargs):
        """Called once by the server to configure Ditto‑specific hyper‑params."""
        self.lam: float = lam
        # cache for personalised weights across rounds
        self._personalised_state = None  # type: dict | None

    # ------------------------------------------------------------------
    # Public helpers ----------------------------------------------------
    # ------------------------------------------------------------------
    def get_personalised_params(self):
        """Return the latest personalised weights (for evaluation/debug)."""
        return self._personalised_state

    # ------------------------------------------------------------------
    # Ditto local training ---------------------------------------------
    # ------------------------------------------------------------------
    def solve_inner_ditto(
        self,
        global_model: torch.nn.Module,
        num_epochs: int = 1,
        batch_size: int = 10,
    ) -> Tuple[Tuple[int, dict], Tuple[int, int, int]]:
        """Run one communication round of Ditto on this client.

        1. **Personalised pass** – update `p_i` with proximal term.
        2. **Global pass** – fine‑tune a *fresh* copy of the global weights
           `w` on local data (no prox).  These weights are uploaded.
        """
        if global_model is None:
            raise RuntimeError("Global model required for Ditto")

        # Book‑keeping --------------------------------------------------
        bytes_w = graph_size(self.model)  # size of weights downloaded
        train_sample_size = 0

        # ==============================================================
        # 1. PERSONALISED UPDATE  (prox‑regularised)                    
        # ==============================================================
        if self._personalised_state is None:
            # first round -> initialise personalised weights with global
            self._personalised_state = deepcopy(global_model.state_dict())

        personalised_model = deepcopy(global_model)
        personalised_model.load_state_dict(self._personalised_state)
        personalised_model.to(self.device)
        personalised_model.train()

        # Build an optimiser that mirrors the global one (SGD/Adam, etc.)
        opt_cls = type(self.optimizer)
        lr = self.optimizer.param_groups[0]["lr"]
        optimiser_p = opt_cls(personalised_model.parameters(), lr=lr)

        for _ in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if len(labels) <= 1:
                    continue  # avoid BN/contrastive issues on tiny batches

                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level

                # Supervised loss
                logits = personalised_model(inputs)
                loss_sup = self.criterion(logits, labels)

                # Proximal regulariser  (λ/2)||p_i − w||²
                reg_loss = 0.0
                with torch.no_grad():
                    global_params = [p.clone() for p in global_model.parameters()]
                for p_loc, p_glob in zip(personalised_model.parameters(), global_params):
                    reg_loss += torch.sum((p_loc - p_glob) ** 2)

                loss = loss_sup + 0.5 * self.lam * reg_loss

                optimiser_p.zero_grad()
                loss.backward()
                optimiser_p.step()

                train_sample_size += len(labels)

        # cache personalised weights for next round & evaluation
        self._personalised_state = deepcopy(personalised_model.state_dict())

        # ==============================================================
        # 2. GLOBAL UPDATE  (FedAvg‑style, no proximal term)            
        # ==============================================================
        self.model.load_state_dict(global_model.state_dict())  # reset to w
        self.model.train()

        for _ in range(num_epochs):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if len(labels) <= 1:
                    continue

                if self.noisy:
                    inputs = inputs + torch.randn_like(inputs) * self.noise_level

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_sample_size += len(labels)

        # ==============================================================
        # 3. PACK‑UP & RETURN                                           
        # ==============================================================
        soln = self.get_model_params()  # global weights after local SGD
        comp = num_epochs * (train_sample_size // batch_size) * batch_size
        bytes_r = graph_size(self.model)  # size of weights uploaded
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
