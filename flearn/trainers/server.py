import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from rich.console import Console
from flearn.utils.model_utils import Metrics
from flearn.utils.torch_utils import process_grad
from flearn.utils.trainer_utils import list_overlapping
from flearn.models.model import get_model, get_model_by_name, SimplexModel, SimplexcModel
from flearn.data.data_utils import get_participants_stat
from flearn.config.config_main import TRAINERS
from flearn.config.config_paths import (
    DUMP_JSON_RESULT_PATH,
    TEST_STATS_FILE_NAME,
    TEST_ALL_STATS_FILE_NAME,
    FIGURES_PATH,
    RLAGG_STATS_FILE_NAME,
)
import json
import pickle
import os
import time
import ot
import importlib
import random
from copy import deepcopy
from typing import Callable, Any
from flearn.utils.plotting import plot_data_dict_in_pdf, visualize_embeddings
from flearn.utils.tools import get_optimal_cuda_device
from flearn.data.data_utils import get_testloader
from flearn.data.test_subset import get_representative_subset
from flearn.utils.losses import get_loss_fun
from flearn.data.dataset import CLASSES
from flearn.utils.trainer_utils import normalize_dict
from flearn.utils.file_writer import FileWriter
from flearn.clients.client import BaseClient
from flearn.models.sac_aggregator import SACAgentR


class BaseServer(object):
    def __init__(self, params):
        # Transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)
        self.get_representative_subset = get_representative_subset
        self.count = 0
        if not hasattr(self, "reward_case"):
            self.reward_case = "acc_align"
        # Create worker nodes
        torch.manual_seed(self.seed)
        print(f"setting client_model")
        if self.cuda >= 0:
            # os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.cuda}"  # only expose GPU 2
            torch.cuda.set_device(self.cuda)   # e.g. self.cuda = 2
            self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        elif self.gpu >= 0:
            self.device = torch.device(
                get_optimal_cuda_device(True) if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = "cpu"
        if self.model == None:
            self.client_model:torch.nn.Module = get_model(self.dataset, self.device)
        elif hasattr(self, "trainer") and self.trainer == "floco":
            self.client_model:torch.nn.Module = SimplexModel(
                {
                    'model': {
                        'name': self.model,
                    },
                    'floco': {
                        'endpoints': self.endpoints,
                    },
                    'dataset': {
                        'name': self.dataset,
                    },
                    'common': {
                        'seed': self.seed,
                    }
                }
            ).to(self.device)
        else:
            self.client_model:torch.nn.Module = get_model_by_name(self.dataset, self.device, self.model)

        if self.num_rounds%100==7 and not self.trainer == "floco": #self.num_rounds%100==7 for simplex cls
            if not hasattr(self, "endpoints"):
                self.endpoints = 20
            self.client_model:torch.nn.Module = SimplexcModel(
                dataset_name=self.dataset,
                model_name=self.model,
                device=self.device,
                endpoints=self.endpoints,
                seed=self.seed
            ).to(self.device)

        self.loss_converged: bool = False
        self.personalized: bool = None
        self.ddpg_aggregation: bool = None
        self.robust_test: bool = False
        self.global_test: bool = False
        self.test_loader, self.num_test_samples = get_testloader(self.dataset, self.data_settings_name, self.n_class, 1024)
        self.test_subset = None
        self.plot_features = False
        print("num_test_samples: ", self.num_test_samples)
        print(f"client_model set")
        
        self.sumed_cost_matrix = torch.zeros(
            CLASSES[self.dataset], CLASSES[self.dataset], device=self.device
        )
        self.inner_opt = optim.SGD(
            self.client_model.parameters(), lr=params["learning_rate"]
        )

        if not hasattr(self, "img_float_val_range"):
            self.img_float_val_range = (0, 1)

        self.clients = self.setup_clients(self.dataset, self.client_model)
        print("{} Clients in Total".format(len(self.clients)))
        num_clients = len(self.clients)
        #####
        self.testing_mode = "Local"
        if self.num_rounds%100==1:
            self.global_test = True # To test the model across different test data
            print(f"Testing Global..")
            self.testing_mode = "Global"
        if self.num_rounds%100==3:
            self.robust_test = True # To test the model across different test data
            print(f"Testing Robustness..")
            self.testing_mode = "Robust"
        if self.num_rounds%100==5:
            self.robust_test = True
            self.global_test = True
            print(f"Testing Both Robust and Global..")
            self.testing_mode = "Global"
        if self.num_rounds%100==7: #self.num_rounds%100==7 for simplex cls  and self.num_rounds%100==2 for normal cls
            self.robust_test = True # To test the model across different test data
            self.global_test = True
            print(f"Testing Both Robust and Global with simplex cls..")
            self.testing_mode = "Global"
        if self.num_rounds%100==9:   
            self.noisy = True            # Check if self has noise_level, else ask user
            if not hasattr(self, "noise_level"):
                self.noise_level = float(input("Please enter noise level: "))
            str_noise_level = str(self.noise_level).replace('.', '_')
            print(f"DP noise..")
            for c in self.clients:
                c.noisy = True
                c.noise_level = self.noise_level
            self.robust_test = True
            self.global_test = True
            print(f"Testing Both Robust and Global..")
            self.testing_mode = f"noisy_{str_noise_level}"
        if self.num_rounds%100==11: 
            self.plot_features = True
            self.testing_mode = "Plot"
        if self.num_rounds%100==13: 
            self.plot_features = True
            self.robust_test = True # To test the model across different test data
            self.global_test = True
            print(f"Testing Both Robust and Global..")
            self.testing_mode = "Plot"

        ######
        self.latest_model = self.client_model.state_dict()
        Lr_ = (
            f"0_" + str(self.learning_rate).split(".")[1]
        )  # '0_01' if self.learning_rate == 0.01 else '0_001'
        self.le = f"" # Loss name extension
        if self.loss is not None:
            if self.loss == "CAPA":
                # client_ = self.clients[0]
                # a = str(client_.criterion.prior_beta).split(".")
                self.le += f"(P_{0}_{5}AB_{0}_{0}{"1"}_m_{0}_{"2"}K_{6}S)"
            if self.loss == "CL":
                client_ = self.clients[0]
                a = str(client_.criterion.tau).split(".")
                if len(a)>1:
                    tau = f"{a[0]}_{a[1]}"
                else: tau = f"{a[0]}"
                self.le += f"(tau_{tau})"
            if self.loss == "DB":
                client_ = self.clients[0]
                a = str(client_.criterion.alpha).split(".")
                if len(a)>1:
                    alpha = f"{a[0]}_{a[1]}"
                else: alpha = f"{a[0]}"
                a = str(client_.criterion.ema_m).split(".")
                if len(a)>1:
                    ema_m = f"{a[0]}_{a[1]}"
                else: ema_m = f"{a[0]}"
                self.le += f"(v2_alpha_{alpha}_ema_{ema_m})"
            if self.loss == "CACS":
                client_ = self.clients[0]
                # a = str(client_.criterion.m0).split(".")
                # if len(a)>1:
                #     m0 = f"{a[0]}_{a[1]}"
                # else: m0 = f"{a[0]}"
                # a = str(client_.criterion.alpha).split(".")
                # if len(a)>1:
                #     alpha = f"{a[0]}_{a[1]}"
                # else: alpha = f"{a[0]}"
                a = str(client_.criterion.prior_beta).split(".")
                if len(a)>1:
                    pb = f"{a[0]}_{a[1]}"
                else: pb = f"{a[0]}"
                a = str(client_.criterion.conf_beta).split(".")
                if len(a)>1:
                    cb = f"{a[0]}_{a[1]}"
                else: cb = f"{a[0]}"
                a = str(client_.criterion.lmu).split(".")
                if len(a)>1:
                    lmu = f"{a[0]}_{a[1]}"
                else: lmu = f"{a[0]}"
                a = str(client_.criterion.cmu).split(".")
                if len(a)>1:
                    cmu = f"{a[0]}_{a[1]}"
                else: cmu = f"{a[0]}"
                a = str(client_.criterion.ema_m).split(".")
                if len(a)>1:
                    ema_m = f"{a[0]}_{a[1]}"
                else: ema_m = f"{a[0]}"
                # self.le += f"(m_{m0}_a_{alpha}_c{cb}_p{pb}_lmu_{lmu}_cmu_{cmu}_ema_{ema_m})" #v1 to v8
                # self.le += f"(v9_c{cb}_p{pb}_lmu_{lmu}_cmu_{cmu}_ema_{ema_m})" #v9
                self.le += f"(p{pb}_c{cb}_lmu_{lmu}_cmu_{cmu}_ema_{ema_m})" #v10
            if self.loss == "CALC":
                client_ = self.clients[0]
                # CACS_LC: include prior_beta, conf_beta, lmu, cmu, ema_m and tau
                a = str(getattr(client_.criterion, 'prior_beta', 0.0)).split('.')
                if len(a) > 1:
                    pb = f"{a[0]}_{a[1]}"
                else:
                    pb = f"{a[0]}"
                a = str(getattr(client_.criterion, 'conf_beta', 0.0)).split('.')
                if len(a) > 1:
                    cb = f"{a[0]}_{a[1]}"
                else:
                    cb = f"{a[0]}"
                a = str(getattr(client_.criterion, 'lmu', 0.9)).split('.')
                if len(a) > 1:
                    lmu = f"{a[0]}_{a[1]}"
                else:
                    lmu = f"{a[0]}"
                a = str(getattr(client_.criterion, 'cmu', 0.01)).split('.')
                if len(a) > 1:
                    cmu = f"{a[0]}_{a[1]}"
                else:
                    cmu = f"{a[0]}"
                a = str(getattr(client_.criterion, 'ema_m', getattr(client_.criterion, 'ema_m', 0.995))).split('.')
                if len(a) > 1:
                    ema_m = f"{a[0]}_{a[1]}"
                else:
                    ema_m = f"{a[0]}"
                # tau comes from the client criterion for label calibration
                a = str(getattr(client_.criterion, 'tau', 0.1)).split('.')
                if len(a) > 1:
                    tau = f"{a[0]}_{a[1]}"
                else:
                    tau = f"{a[0]}"
                self.le += f"(p{pb}_c{cb}_lmu_{lmu}_cmu_{cmu}_ema_{ema_m}_tau_{tau})"
            if self.loss == "LCCA":
                client_ = self.clients[0]
                # LCCA: include tau, lambda_conf and ema_m
                a = str(getattr(client_.criterion, 'tau', 0.1)).split('.')
                if len(a) > 1:
                    tau = f"{a[0]}_{a[1]}"
                else:
                    tau = f"{a[0]}"
                a = str(getattr(client_.criterion, 'lambda_conf', 0.5)).split('.')
                if len(a) > 1:
                    lconf = f"{a[0]}_{a[1]}"
                else:
                    lconf = f"{a[0]}"
                a = str(getattr(client_.criterion, 'ema_m', 0.995)).split('.')
                if len(a) > 1:
                    ema_m = f"{a[0]}_{a[1]}"
                else:
                    ema_m = f"{a[0]}"
                self.le += f"(tau_{tau}_lconf_{lconf}_ema_{ema_m})"
        print(f"Loss extension: {self.le}")       
        self.experiment_name = f"{self.trainer}_M_{self.model}_{self.dataset}_nc_{self.n_class}_{self.dataset_type}_L_{self.loss}_lr_{Lr_}_B_{self.batch_size}_C_{self.clients_per_round}_E_{self.num_epochs}_{self.num_rounds}"
        
        m_name = self.model if self.model is not None else "MLP"
        self.experiment_short_name = f"{self.trainer}_{m_name}_{self.loss}{self.le}_nc_{self.n_class}_lr_{Lr_}_B_{self.batch_size}_E_{self.num_epochs}_{self.num_rounds}_C_{num_clients}_{self.clients_per_round}"
        self.accuracy_global = []
        self.loss_global = []
        self.accuracy_clients = {}
        self.desc = f"Algo: {self.trainer}, M-{self.model}, D-{self.dataset}, N-{self.n_class}, T-{self.dataset_type}, LR-{self.learning_rate}, E-{self.num_epochs}, L-{self.loss}, B-{self.batch_size}, C-{self.clients_per_round}, G-{self.device}, Test-{self.testing_mode}"

        if self.num_rounds%100==4 or self.num_rounds%100==5: 
            if self.trainer == "fedblo" or self.agg == "drl":
                self.ddpg_aggregation = True
                self.num_classes = CLASSES[self.dataset]
                self.criterion = get_loss_fun("CE")()
                self.test_subset, _ = get_representative_subset(test_loader=self.test_loader, num_samples=1024)
                if self.use_prev_global_model:
                    input_channels = self.clients_per_round + 2  # Clients + prev global + current global model
                    num_clients_per_round=self.clients_per_round + 1 , #+1 for prev global params
                    action_dim=self.clients_per_round +1  #+1 for prev global params
                else:
                    input_channels = self.clients_per_round + 1  # Clients + current global model
                    num_clients_per_round=self.clients_per_round
                    action_dim=self.clients_per_round
                self.rl_agent = SACAgentR(
                                        eval_fn=self.test_model_params, 
                                        build_class_prototypes=self.build_class_prototypes,
                                        num_classes=self.num_classes, 
                                        device=self.device,
                                        num_clients_per_round=num_clients_per_round,
                                        input_channels=input_channels,
                                        input_dim=self.num_classes * self.num_classes,  # Size of each evaluation matrix    
                                        action_dim=action_dim,
                                        reward_case=self.reward_case,
                                    )
                # print("action dim:", self.rl_agent.action_dim)

        if self.agg is not None:
            self.ae = f""
            if self.agg == "fedsat" or self.agg == "fedsatc" or self.agg == "prawgs" or self.agg == "prawgcs":
                a = str(self.top_p).split(".")
                if len(a)>1:
                    top_p = f"{a[0]}_{a[1]}"
                else: top_p = f"{a[0]}"
                self.ae += f"(topP_{top_p})"
            if self.agg == "fedadam":
                a = str(self.server_learning_rate).split(".")
                if len(a)>1:
                    s_lr = f"{a[0]}_{a[1]}"
                else: s_lr = f"{a[0]}"
                self.ae += f"(slr_{s_lr})"
            if self.agg == "fedyogi":
                a = str(self.server_learning_rate).split(".")
                if len(a)>1:
                    s_lr = f"{a[0]}_{a[1]}"
                else: s_lr = f"{a[0]}"
                self.ae += f"(slr_{s_lr})"
            if self.agg == "fedavgm":
                a = str(self.server_learning_rate).split(".")
                if len(a)>1:
                    s_lr = f"{a[0]}_{a[1]}"
                else: s_lr = f"{a[0]}"
                a = str(self.server_momentum).split(".")
                if len(a)>1:
                    s_m = f"{a[0]}_{a[1]}"
                else: s_m = f"{a[0]}"
                self.ae += f"(slr_{s_lr}_sm_{s_m})" # default server_momentum = 0.9, default server_learning_rate = 1.0
            if self.agg == "elastic":
                self.ae += f"(v2_mu_0_95_tau_0_5)" # default elastic_momentum = 0.95, default tau = 0.5
            if self.agg == "drl":
                safe_reward_case = str(getattr(self, "reward_case", "acc_align")).replace(" ", "_").replace("-", "_")
                self.ae += f"(rw_{safe_reward_case})"

            agg_extension = f"_A_{self.agg}{self.ae}"            
            if self.use_prev_global_model:
                agg_extension = f"_A_{self.agg}_pg{self.ae}"   
            print(f"agg_extension: {agg_extension}")
            self.experiment_name = self.experiment_name + agg_extension
            self.experiment_short_name = self.experiment_short_name + agg_extension
            self.desc = self.desc + f", Agg-{self.agg}"

        if self.trainer == "fedadam" or self.trainer == "fedyogi":
            a = str(self.server_learning_rate).split(".")
            if len(a)>1:
                s_lr = f"{a[0]}_{a[1]}"
            else: s_lr = f"{a[0]}"
            self.ae = f"(slr_{s_lr})"
            agg_extension = f"{self.ae}"
            print(f"agg_extension: {agg_extension}")
            self.experiment_name = self.experiment_name + agg_extension
            self.experiment_short_name = self.experiment_short_name + agg_extension
            self.desc = self.desc + f"{self.ae}"

        if self.trainer == "floco":
            self.experiment_name = self.experiment_name + f"_tau_{self.tau}"
            self.experiment_short_name = self.experiment_short_name + f"_tau_{self.tau}"
            self.desc = self.desc + f", Tau-{self.tau}"

        if self.trainer == "fedmrl":
            suffix_parts = []
            desc_bits = []

            if hasattr(self, "tau") and self.tau is not None:
                tau_str = str(self.tau).replace('.', '_')
                suffix_parts.append(f"tau_{tau_str}")
                # desc_bits.append(f"Tau-{self.tau}")

            if hasattr(self, "mu") and self.mu is not None:
                mu_str = str(self.mu).replace('.', '_')
                suffix_parts.append(f"mu_{mu_str}")
                # desc_bits.append(f"Mu-{self.mu}")

            adv_gain = getattr(self, "adv_gain", None)
            if adv_gain is not None:
                adv_str = str(adv_gain).replace('.', '_')
                suffix_parts.append(f"adv_{adv_str}")
                # desc_bits.append(f"AdvGain-{adv_gain}")

            if hasattr(self, "version") and self.version is not None:
                suffix_parts.append(f"{self.version}")
                desc_bits.append(f"Version-{self.version}")
            if hasattr(self, "max_rl_steps") and self.max_rl_steps is not None:
                suffix_parts.append(f"steps_{self.max_rl_steps}")
                # desc_bits.append(f"MaxRLSteps-{self.max_rl_steps}")

            if suffix_parts:
                suffix = "_".join(suffix_parts)
                self.experiment_name = f"{self.experiment_name}_{suffix}"
                self.experiment_short_name = f"{self.experiment_short_name}_{suffix}"
            if desc_bits:
                self.desc = self.desc + ", " + ", ".join(desc_bits)

        if self.trainer == "moon":
            extension = f"(mu_{str(self.mu).replace('.', '_')}_tau_{str(self.tau).replace('.', '_')}_{self.prev_model_version})"
            self.experiment_name = self.experiment_name + f"_{extension}"
            self.experiment_short_name = self.experiment_short_name + f"_{extension}"
            print(f"extension: {extension}")

        if self.num_rounds%100==7:
            self.experiment_name = self.experiment_name + f"_sm"
            self.experiment_short_name = self.experiment_short_name + f"_sm"
            self.desc = self.desc + f", cls-Simplex"

        if self.trainer == "fedprotocvae":
            b = str(self.lamda_malg).split(".")
            if len(b)>1:
                lamda_malg = f"{b[0]}_{b[1]}"
            else: lamda_malg = f"{b[0]}"
            c = str(self.lamda_gcls).split(".")
            if len(c)>1:
                lamda_gcls = f"{c[0]}_{c[1]}"
            else: lamda_gcls = f"{c[0]}"
            extension = f"Alg_{lamda_malg}_CT_{lamda_gcls}v16"
            self.experiment_name = self.experiment_name + f"_{extension}"
            self.experiment_short_name = self.experiment_short_name + f"_{extension}"
            print(f"extension: {extension}")

        self.figures_path = (
            FIGURES_PATH
            + f"/{self.dataset}_{self.dataset_type}_{self.testing_mode}/{self.experiment_short_name}/"
        )
        # Initialize system metrics
        self.metrics = Metrics(self.clients, params)

        if not hasattr(self, "metadata"):
            self.metadata = {}
        # Basic test mode
        self.metadata["testing_mode"] = self.testing_mode

        # Enrich metadata with reproducible run settings and provenance
        try:
            # Basic identifiers
            self.metadata.setdefault("experiment_short_name", self.experiment_short_name)
            self.metadata.setdefault("experiment_name", self.experiment_name)
            self.metadata.setdefault("description", self.desc)

            # Device and seed
            self.metadata.setdefault("device", {})
            self.metadata["device"].setdefault("device_str", str(self.device))
            self.metadata["device"].setdefault("cuda_index", getattr(self, "cuda", None))
            self.metadata.setdefault("seed", getattr(self, "seed", None))

            # Optimizer / hyperparameters
            self.metadata.setdefault("optimizer", {})
            self.metadata["optimizer"].setdefault("name", getattr(self, "optm", None) or "SGD")
            self.metadata["optimizer"].setdefault("learning_rate", getattr(self, "learning_rate", None))
            self.metadata["optimizer"].setdefault("momentum", getattr(self, "momentum", None))
            self.metadata["optimizer"].setdefault("weight_decay", getattr(self, "weight_decay", None))

            # Core experiment config
            self.metadata.setdefault("core", {})
            self.metadata["core"].setdefault("batch_size", getattr(self, "batch_size", None))
            self.metadata["core"].setdefault("client_epochs", getattr(self, "num_epochs", None))
            self.metadata["core"].setdefault("server_rounds", getattr(self, "num_rounds", None))
            self.metadata["core"].setdefault("num_clients", num_clients)
            self.metadata["core"].setdefault("clients_per_round", getattr(self, "clients_per_round", None))

            # Aggregation/server-specific params
            if hasattr(self, "agg") and self.agg is not None:
                self.metadata.setdefault("aggregation", {})
                self.metadata["aggregation"].setdefault("name", self.agg)
                # common agg-specific params
                if hasattr(self, "top_p"):
                    self.metadata["aggregation"].setdefault("top_p", getattr(self, "top_p"))
                if hasattr(self, "server_learning_rate"):
                    self.metadata["aggregation"].setdefault("server_learning_rate", getattr(self, "server_learning_rate"))
                if hasattr(self, "server_momentum"):
                    self.metadata["aggregation"].setdefault("server_momentum", getattr(self, "server_momentum"))

            # Loss-specific parameters (try to inspect client criterion)
            try:
                client0 = self.clients[0]
                crit = getattr(client0, "criterion", None)
                if crit is not None:
                    loss_params = {}
                    # common candidate attrs
                    candidates = [
                        "prior_beta",
                        "conf_beta",
                        "lmu",
                        "cmu",
                        "ema_m",
                        "tau",
                        "rho",
                        "alpha",
                        "m0",
                        "p0",
                        "c0",
                    ]
                    for a in candidates:
                        if hasattr(crit, a):
                            try:
                                loss_params[a] = getattr(crit, a)
                            except Exception:
                                loss_params[a] = str(getattr(crit, a))
                    if len(loss_params) > 0:
                        self.metadata.setdefault("loss_params", {})
                        self.metadata["loss_params"].update(loss_params)
            except Exception:
                pass

            # provenance
            try:
                self.metadata.setdefault("provenance", {})
                self.metadata["provenance"].setdefault("generated_at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                self.metadata["provenance"].setdefault("generated_by", "BaseServer:FileWriter")
                self.metadata["provenance"].setdefault("results_path", getattr(self.filewriter, "path", None) if hasattr(self, "filewriter") else None)
            except Exception:
                pass
        except Exception:
            # Fallback: leave metadata as-is if enrichment fails
            pass
        self.filewriter = FileWriter(
            dataset_name=self.dataset,
            # dataset_type=self.dataset_type,
            dataset_type=self.data_settings_name,
            trainer=self.trainer,
            model=self.model,
            n_class=self.n_class,
            loss=self.loss,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            rounds=self.num_rounds,
            num_clients=num_clients,
            clients_per_round=self.clients_per_round,
            experiment_name=self.experiment_short_name,
            test_mode=self.testing_mode,
            metadata=self.metadata,
            buffer_limit=self.file_buffer_limit,
        )

        self.filewriter.add_csv_file(
            filename=TEST_STATS_FILE_NAME,
            # headers=["round", "training_loss", "training_accuracy", "test_accuracy"],
            headers=["round", "training_loss", "training_accuracy", "local_accuracy", "robust_accuracy", "global_accuracy"],
        )

        self.filewriter.add_csv_file(
            filename=TEST_ALL_STATS_FILE_NAME,
            headers=["round", "cost_matrix", "all_client_accuracy"],
        )

        self.filewriter.add_csv_file(
            filename=RLAGG_STATS_FILE_NAME,
            headers=["round", "steps", "initial_acc", "improved_acc", "improvement", "improve_percentage", "pocessing_time"],
        )

    def __del__(self):

        self.filewriter.flush_csv_buffers()

        if (
            hasattr(self, "client_model")
            and getattr(self, "client_model", None) is not None
        ):
            print(f"Training Stopped")
            # self.client_model.close()

    def setup_clients(self, dataset, model=None) -> list[BaseClient]:
        """Instantiates clients based on given train and test data directories

        Return:
            List of Clients
        """
        clients_stats = get_participants_stat(
            # self.dataset, self.dataset_type, self.n_class
            self.dataset, self.data_settings_name, self.n_class
        )

        if self.num_clients != None:
            clients_stats = clients_stats[0 : self.num_clients]

        # load corresponding client
        client_class_name = TRAINERS[self.trainer]["client"]
        # Try trainer-specific client module first (flearn.clients.<trainer>),
        # then fall back to deriving module name from client class (FedAvgClient->fedavg).
        ClientClass: type[BaseClient] = None
        tried_paths = []
        # First attempt: module named after trainer (preferred when trainer has its own client implementation)
        trainer_module_path = f"flearn.clients.{self.trainer}"
        try:
            tried_paths.append(trainer_module_path)
            mod = importlib.import_module(trainer_module_path)
            if hasattr(mod, client_class_name):
                ClientClass = getattr(mod, client_class_name)
        except Exception:
            ClientClass = None

        if ClientClass is None:
            # Fallback: derive module name from client class name (e.g., FedAvgClient -> fedavg)
            client_module_name = client_class_name.replace("Client", "").lower()
            client_module_path = f"flearn.clients.{client_module_name}"
            tried_paths.append(client_module_path)
            try:
                mod = importlib.import_module(client_module_path)
                ClientClass = getattr(mod, client_class_name)
            except Exception as e:
                raise ImportError(f"Could not import client class {client_class_name} from tried paths {tried_paths}: {e}")

        all_clients = [
            ClientClass(
                user_id=user_id,
                device=self.device,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                loss=self.loss,
                batch_size=self.batch_size,
                dataset=self.dataset,
                valset_ratio=self.valset_ratio,
                logger=Console(record=self.log),
                gpu=self.gpu,
                dataset_type=dataset_type,
                n_class=n_class,
                optm=self.optm,
                group=None,
                model=self.client_model,
                num_workers=self.num_workers,
                img_float_val_range=self.img_float_val_range,
            )
            for user_id, dataset_type, n_class in clients_stats
        ]
        return all_clients

    def eval(
        self,
        round: int,
        setClientModelFunc: Callable,
        **ClientModelFuncKwargs,
    ):
        if round % self.eval_every == 0:
            train_acc, local_acc, robust_acc, global_acc = None, None, None, None
            stats_train = self.train_error_and_loss()
            if self.plot_features:
                self.eval_client_features(round=round, split_name=f"train")
                self.eval_client_features(round=round, split_name=f"val")
            for c in self.clients:
                setClientModelFunc(c, **ClientModelFuncKwargs)
            # stats = self.test()
            if self.plot_features:
                self.eval_client_features(round=round, split_name=f"test")
                self.eval_global_features(round=round)
            if self.robust_test:
                stats = self.test_model(selected_clients=self.select_clients(round=round, num_clients=int(len(self.clients)*0.1)))
                robust_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
                if not self.global_test: self.accuracy_global.append(robust_acc)
            if self.global_test:
                stats = self.test_global()
                global_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
                self.accuracy_global.append(global_acc)
            stats = self.test()
            local_acc= np.sum(stats[3]) * 1.0 / np.sum(stats[2])
            if not (self.global_test or self.robust_test):
                self.accuracy_global.append(local_acc)
            train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
            train_loss = np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])            
            self.loss_global.append(train_loss)
            tqdm.write(
                "At round {} local accuracy: {}, robust accuracy: {}, global accuracy: {}".format(round, local_acc, robust_acc, global_acc)
            )  # testing accuracy
            tqdm.write("At round {} training accuracy: {}".format(round, train_acc))
            tqdm.write("At round {} training loss: {}".format(round, train_loss))

            self.filewriter.writerow(
                filename=TEST_STATS_FILE_NAME,
                # row=[round, loss, train_acc, acc],
                row=[round, train_loss, train_acc, local_acc, robust_acc, global_acc],
            )
            if train_acc>0.999: self.loss_converged=True

        if self.all_test and ((round + 1) % 10 == 0):
            tqdm.write("Testing all clients for round: {}".format(round))
            self.test_all(server_round=round)

    def eval_end(self):
        if self.all_test:
            self.test_all(server_round=self.num_rounds)
        stats = self.test()
        stats_train = self.train_error_and_loss()
        acc, loss = np.sum(stats[3]) * 1.0 / np.sum(stats[2]), np.dot(
            stats_train[4], stats_train[2]
        ) * 1.0 / np.sum(stats_train[2])
        # self.accuracy_global.append(acc)
        # self.loss_global.append(loss)
        train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])

        self.filewriter.writerow(
                filename=TEST_STATS_FILE_NAME,
                row=[self.num_rounds, loss, train_acc, acc, None, None],
            )
        
        tqdm.write(
            "At last round {} accuracy: {}".format(self.num_rounds, acc)
        )  # testing accuracy
        tqdm.write("At last round {} training accuracy: {}".format(self.num_rounds, train_acc))
        tqdm.write("At last round {} training loss: {}".format(self.num_rounds, loss))
        self.dumping_json()  # Save experiment history in json file

    def train_error_and_loss(self, modelInCPU: bool = False):
        num_samples = []
        tot_correct = []
        losses = []
        # print(f'\nBefore: clients-> {self.clients}\nnum_samples: {num_samples}, total_correct: {tot_correct}, losses: {losses}')
        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss(modelInCPU=modelInCPU)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        # print(f'\nAfter:\nnum_samples: {num_samples}, total_correct: {tot_correct}, losses: {losses}')

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses

    def show_grads(self):
        """
        Return:
            Gradients on all workers and the global gradient
        """
        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)

        intermediate_grads = []
        samples = []

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model)
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads.append(global_grads)

        return intermediate_grads

    def test(self, modelInCPU: bool = False):
        """Tests self.latest_model on given clients"""
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test(modelInCPU=modelInCPU)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct
    
    def test_global(self, modelInCPU: bool = False):
        """Tests self.latest_model on given clients"""
        num_samples = []
        tot_correct = []
        for c in self.clients:
            c.model.eval()
            if modelInCPU:
                c.model = c.model.to(self.device)
            ct, ns = 0, 0
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if len(labels)<=1: continue
                if c.use_custom_classifier:
                    features = c.model.get_representation_features(inputs)
                    outputs = c.classifier(features)
                else: outputs = c.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                ct += correct
                ns += len(labels)

            c.model.train()
            if modelInCPU:
                c.model = c.model.cpu()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct
    
    def test_model(self, selected_clients: list[BaseClient], modelInCPU: bool = False):
        """Tests self.latest_model on given clients"""
        num_samples = []
        tot_correct = []
        for c1 in self.clients:
            for c in selected_clients:
                ct, ns = c.test_model(client=c1, modelInCPU=modelInCPU)
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    @torch.compiler.disable
    def select_clients(self, round, num_clients=20):
        """Selects num_clients clients weighted by the number of samples from possible_clients

        Args:
            num_clients: Number of clients to select; default 20
                Note that within the function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            List of selected clients objects"""

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(
            round
        )  # Make sure for each comparison, we are selecting the same clients each round
        return np.random.choice(
            self.clients, num_clients, replace=False
        )  # Uniform sampling

    def select_workers(self, round, selected_clients, overlap_ratio, client_ratio=1.0):
        # Check if the list of selected clients is empty
        if len(selected_clients) == 0:
            return f"Error: The list of selected clients cannot be empty"

        # Check if there are enough workers available for selected clients
        if len(self.clients) - len(selected_clients) < len(selected_clients):
            return f"Error: Insufficient workers available; reduce the number of clients selected for training"

        # Create a list of remaining clients not in the selected clients
        remaining_clients = [
            client for client in self.clients if client not in selected_clients
        ]

        # Calculate the worker ratio based on the number of remaining clients and selected clients
        worker_ratio = int(
            1 + client_ratio * (len(remaining_clients) / len(selected_clients))
        )
        workers_set = {}
        for i, client in enumerate(selected_clients):
            log = f"Round {round}: Remaining clients: {remaining_clients}"
            np.random.seed(round + i)  # Make sure for each comparison
            selected_workers = np.random.choice(
                remaining_clients, int(worker_ratio), replace=True
            ).tolist()
            workers_set[client] = selected_workers
            log += f"\n Selected workers for client {selected_clients[i]}: {selected_workers}"
        return list_overlapping(overlap_ratio, workers_set)

    def test_all(self, server_round: int, render=False, modelInCPU: bool = False):
        # self.count += 1
        all_client_acc = []
        all_cost_matrix = torch.ones(
            CLASSES[self.dataset], CLASSES[self.dataset], device=self.device
        )
        for index, c in enumerate(self.clients):
            loss, acc, test_sample, record_stats, pred, target = c.test_stats(
                modelInCPU=modelInCPU
            )
            if render:
                tqdm.write("Client Id: {}, Targ-> {}".format(c.id, target))
                tqdm.write(
                    "Client Id: {}, Pred-> {}\n**** Loss: {}, Acc: {} ****".format(
                        c.id, pred, loss, acc
                    )
                )
            all_client_acc.append(np.round(acc, 2))
            if c.id not in self.accuracy_clients:
                self.accuracy_clients[c.id] = {
                    "acc": [acc],
                    "loss": [loss],
                    "pred": [pred.tolist()],
                    "target": [target.tolist()],
                }
            else:
                self.accuracy_clients[c.id]["acc"].append(acc)
                self.accuracy_clients[c.id]["loss"].append(loss)
                self.accuracy_clients[c.id]["pred"].append(pred.tolist)
                self.accuracy_clients[c.id]["target"].append(target.tolist())
            all_cost_matrix += c.test_and_cost_matrix_stats_t(modelInCPU=modelInCPU)
        self.sumed_cost_matrix += all_cost_matrix
        cost_acc = torch.diag(all_cost_matrix).sum() / all_cost_matrix.sum()

        cost_matrix = self.sumed_cost_matrix.cpu().tolist()

        # Write to results csv file
        self.filewriter.writerow(
            filename=TEST_ALL_STATS_FILE_NAME,
            row=[server_round, cost_matrix, all_client_acc],
        )

        mean_acc = np.mean(all_client_acc)
        sum_acc = np.sum(all_client_acc)
        print(
            f"\nMean {mean_acc}, Sum: {sum_acc}, Cost_acc: {cost_acc}, Len: {len(all_client_acc)} \nAll_acc: {all_client_acc}"
        )

    def dumping_json(self, path: str | None = None, if_pdf=True):

        if path is None:
            path = DUMP_JSON_RESULT_PATH
        if not os.path.exists(path):
            os.makedirs(path)
        data = dict()
        data["name"] = self.experiment_name
        data["short_name"] = self.experiment_short_name
        data["x"] = [i for i in range(len(self.accuracy_global))]
        data["dual_axis"] = True
        data["y"] = [[self.accuracy_global], [self.loss_global]]
        data["legends"] = [[f"{self.trainer}-A"], [f"{self.trainer}-L"]]
        data["labels"] = ["Rounds", ["Accuracy", "Loss"]]
        data["max_acc_g"] = max(self.accuracy_global)
        data1 = dict()
        data1["name"] = self.experiment_name
        data1["clients_accuracy"] = self.accuracy_clients
        # print("Data->>", data)
        file_name = f"{self.experiment_name}.json"
        # Write the dictionary to the file in JSON format
        with open(path + file_name, "w") as file:
            json.dump(data, file)
        pickle_name = f"All_clients{self.experiment_name}.pickle"
        # Save dictionary data to a pickle file
        with open(path + pickle_name, "wb") as file:
            pickle.dump(data1, file)
        if if_pdf:
            plot_data_dict_in_pdf(data, path=self.figures_path)

    @torch.no_grad()
    def test_model_params(self, model_params, num_batches=1, name='',  render=False):
        num_classes = CLASSES[self.dataset]      
        if self.test_subset is not None:
            cost_matrix = torch.ones(num_classes, num_classes, device=self.device)
            tot_correct, loss, test_sample = 0, 0.0, 0
            self.client_model.load_state_dict(model_params, strict=False)
            self.client_model.eval()
            
            for inputs, labels in self.test_subset:     
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.client_model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss += self.criterion(outputs, labels).item()
                for x,y in zip(labels, predicted):
                    cost_matrix[x][y] = cost_matrix[x][y] + 1
                    if x==y: tot_correct +=1
                test_sample += len(labels)
            acc = tot_correct/test_sample
            cost_matrix = cost_matrix - 1
            if render:
                print('\n**** Loss: {}, Acc: {} ****'.format(loss, acc))# Subtract 1 from each element in the tensor
                # cost_matrix = cost_matrix - 1
                sum_diagonal = torch.diag(cost_matrix).sum() #Correct
                sum_all = cost_matrix.sum() #Total sample
                accuracy = sum_diagonal / sum_all
                print(f"Accuracy of {name}model is: {accuracy}")
                for i in range(cost_matrix.size(0)):  # Assuming cost_matrix is a square tensor
                    num_samples = cost_matrix[i].sum().item()
                    correct_predictions = cost_matrix[i][i].item()
                    print(f"Class {i}: Total Samples = {num_samples}, Correct Predictions = {correct_predictions}")
            return acc, cost_matrix
        else: 
            print(f'Test loader not available')
            raise RuntimeError
        

    @torch.no_grad()
    def build_class_prototypes(
        self,
        model_params,
    ):
        """
        Compute class-wise prototypes (mean feature vectors) for a model.

        Args:
            model: A model with `get_representation_features(inputs)` and,
                if only_correct=True, a `classifier` head producing logits.

        Returns:
            If return_tensor=False:
                dict: {class_id: prototype (D,)}
            If return_tensor=True:
                (protos, classes)
                - If num_classes is None: protos shape is (C_found, D) ordered by `classes` (sorted list)
                - If num_classes is given: protos shape is (num_classes, D); classes is [0..num_classes-1]
                and rows for classes not seen will be all zeros.
        """
        model: torch.nn.Module = deepcopy(self.client_model)
        model.eval()
        model.load_state_dict(model_params, strict=False)

        sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}

        for inputs, labels in self.test_subset:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            feats = model.get_representation_features(inputs)  # (B, D)

            logits = model.classifier(feats)               # (B, C)
            preds = logits.argmax(dim=1)
            mask = preds.eq(labels)
            if mask.ndim == 0:  # edge case single element
                mask = mask.unsqueeze(0)
            feats = feats[mask]
            labels = labels[mask]

            # Accumulate sums and counts
            for f, y in zip(feats, labels):
                cid = int(y.item())
                if cid not in sums:
                    sums[cid] = f.detach().clone()
                    counts[cid] = 1
                else:
                    sums[cid] += f.detach()
                    counts[cid] += 1

        if not sums:
            raise ValueError("No samples matched the selection (maybe only_correct=True and no correct preds).")

        # Convert to means
        proto_dict = {cid: sums[cid] / counts[cid] for cid in sums.keys()}

        # Build tensor output
        # feat_dim = next(iter(proto_dict.values())).shape[0]
        # protos = torch.zeros(self.num_classes, feat_dim)
        # for c in range(self.num_classes):
        #     if c in proto_dict:
        #         protos[c] = proto_dict[c].cpu()
        # classes = list(range(self.num_classes))
        return proto_dict, counts

    def compute_ot_barycenter(self, client_flat_params, reg=0.5):
        """
        Compute the Wasserstein barycenter of multiple clients' flattened model parameters.
        
        Args:
            client_flat_params (dict): client -> flattened parameter numpy array
            reg (float): Entropy regularization parameter for Sinkhorn algorithm

        Returns:
            np.ndarray: OT barycenter of client parameters
        """
        client_vectors = list(client_flat_params.values())
        num_clients = len(client_vectors)
        
        # # Make sure all vectors are normalized to form valid probability distributions
        # client_vectors = [v / np.sum(v) for v in client_vectors]
        # for i, v in enumerate(client_vectors):
        #     if not np.all(np.isfinite(v)):
        #         print(f"[Warning] Client {i} has invalid values after normalization.")

        # Create cost matrix (squared Euclidean distance between vector indices)
        dim = client_vectors[0].shape[0]
        x = np.arange(dim).reshape(-1, 1)
        cost_matrix = ot.dist(x, x, metric='euclidean') ** 2

        # Compute Wasserstein barycenter
        # barycenter = ot.bregman.barycenter(
        #     distributions=client_vectors,
        #     M=cost_matrix,
        #     reg=reg,
        #     weights=[1 / num_clients] * num_clients  # Uniform weights
        # )
        client_vectors_d = np.vstack(client_vectors).T
        weights = [1 / num_clients] * num_clients
        # print(len(weights), client_vectors_d.shape, len(client_vectors))
        barycenter = ot.bregman.barycenter(
            client_vectors_d,   # A: shape [n_clients, dim]
            cost_matrix,                 # M
            reg,                         # reg
            weights=weights
        )

        return barycenter
        # barycenter_tensor = torch.tensor(barycenter, dtype=torch.float32, device=self.device)
        # return barycenter_tensor

    def eval_features_(self):
        if self.round%10==0 and self.plot_features:            
            all_protos, all_labels = [], [] 
            for proto, label in self.feature_memory:
                proto_array = proto.clone().detach().cpu().numpy()
                all_protos.extend([proto_array])
                all_labels.extend([label])
            visualize_embeddings(features=np.array(all_protos), 
                                labels=np.array(all_labels),
                                plot_path=f"RESULTS/{'Plot_embeddings'}/{self.dataset}_{self.dataset_type}/{self.trainer_name}/Feature_{self.round}.pdf"
                                )

    def _extract_features(self, model: torch.nn.Module, device, dataloader):
        """
        Args:
            model      : nn.Module with .get_representation_features(inputs)
            device     : torch.device
            dataloader : iter yielding (inputs, labels)

        Returns:
            feats (np.ndarray)  shape = [N, D]
            labs  (np.ndarray)  shape = [N]
        """
        features, labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, lbls in dataloader:
                if len(lbls) <= 1:          # skip degenerate batches
                    continue
                inputs, lbls = inputs.to(device), lbls.to(device)
                feats = model.get_representation_features(inputs)
                features.append(feats.cpu().numpy())
                labels.append(lbls.cpu().numpy())

        if not features:                    # empty dataset
            return None, None
        return np.concatenate(features), np.concatenate(labels)

    def eval_client_features(self, round, split_name):
        """
        Every 10 rounds, generate PDF plots of the embedding space:

        • For each client:
            - train   →  Client<i>_train_round<r>.pdf
            - val     →  Client<i>_val_round<r>.pdf
        • For the global model (server):
            - test    →  Global_test_round<r>.pdf
        """
        if (round % 10) or (not self.plot_features):
            return  # nothing to do this round

        save_dir = (
            f"RESULTS/Plot_embeddings/"
            f"{self.dataset}_{self.dataset_type}/{self.trainer}_{self.num_rounds}/{split_name}/"
        )
        os.makedirs(save_dir, exist_ok=True)

        # ---- 1. Per-client plots ------------------------------------------------
        for client in self.clients:
            cid = client.id
            if split_name=="train":
                loader = client.trainloader
            if split_name=="test":
                loader = client.valloader
            if split_name=="val":
                loader = client.valloader
            feats, labs = self._extract_features(
                client.model, client.device, loader
            )
            if feats is None:  # loader empty
                continue

            visualize_embeddings(
                features=feats,
                labels=labs,
                plot_path=os.path.join(
                    save_dir, f"Client{cid}_{split_name}_round{round}.pdf"
                ),
            )

    def eval_global_features(self, round):
        # ---- 2. Global / aggregated model plot ---------------------------------
        if (round % 10) or (not self.plot_features):
            return  # nothing to do this round

        save_dir = (
            f"RESULTS/Plot_embeddings/"
            f"{self.dataset}_{self.dataset_type}/{self.trainer}_{self.num_rounds}/global/"
        )
        os.makedirs(save_dir, exist_ok=True)
        if self.test_subset is None:
            self.test_subset, _ = get_representative_subset(test_loader=self.test_loader, num_samples=1024)
        feats, labs = self._extract_features(self.client_model, self.device, self.test_subset)
        if feats is not None:
            visualize_embeddings(
                features=feats,
                labels=labs,
                plot_path=os.path.join(
                    save_dir, f"Global_test_round{round}.pdf"
                ),
            )

    def drl_aggregate(self, clients_params_dict):
        """
        SAC-compatible aggregation loop (works with your existing env + replay buffer).
        - Uses self.rl_agent.get_action(state) for stochastic data collection (Concrete policy).
        - Calls self.rl_agent.update(batch_size=32) on a cadence (every 2 steps by default).
        """
        if not hasattr(self, "max_rl_steps"):
            self.max_rl_steps = 100

        # Reset SAC env with the clients' models -> initial state
        state, done = self.rl_agent.reset(parameters_vectors_dict=clients_params_dict)

        steps = 0
        rewards = []
        dones = []
        start_time = time.time()

        # IMPORTANT: add parentheses to preserve the intended logic
        while (not done and steps < self.max_rl_steps):
            steps += 1

            # Ensure state has a batch dimension: expected (B, C, L)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            # Sample a simplex action from the SAC policy (Gumbel-Softmax)
            action = self.rl_agent.get_action(state)  # shape [K], already on simplex

            # Step the FL environment with this aggregation weight vector
            next_state, reward, done = self.rl_agent.step(action)

            # Ensure next_state has a batch dimension too
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)

            # --- Experience storage / update ---
            if not (self.num_rounds % 100 == 4):  # keep your PPO special-case
                # Push to buffer (SAC uses off-policy replay)
                self.rl_agent.replay_buffer.push(state, action.detach(), reward, next_state, done)

                # Update critics/actor/alpha periodically
                if steps % 2 == 0:
                    self.rl_agent.update(batch_size=32)
            else:
                rewards.append(reward)
                dones.append(done)

            state = next_state

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000.0

        # Stats and logging
        diff = self.rl_agent.env.best_accuracy - self.rl_agent.env.global_accuracy
        denom = (self.rl_agent.env.global_accuracy if self.rl_agent.env.global_accuracy and self.rl_agent.env.global_accuracy != 0 else 1e-8)
        improvement_percentage = (diff / denom) * 100.0

        # print(
        #     f"Global Model Accuracy after round {self.round+1}: "
        #     f"{self.rl_agent.env.global_accuracy:.2f}%, Steps: {steps}, "
        #     f"diff: {self.rl_agent.env.highest_accuracy - self.rl_agent.env.best_accuracy:.4f}"
        # )

        self.filewriter.writerow(
            filename=RLAGG_STATS_FILE_NAME,
            row=[self.round, steps, self.rl_agent.env.global_accuracy, self.rl_agent.env.best_accuracy,
                diff, improvement_percentage, elapsed_time_ms],
        )

        self.rl_client_weights = deepcopy(self.rl_agent.env.weights_vector)
        # Return best or current globals as before
        if self.rl_agent.env.new_best:
            self.last_acc = self.rl_agent.env.best_accuracy
            return self.rl_agent.env.best_params
        else:
            self.last_acc = self.rl_agent.env.global_accuracy
            return self.rl_agent.env.global_parameters

