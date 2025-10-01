import torch
import numpy as np
from tqdm import trange, tqdm
from flearn.trainers.server import BaseServer
from collections import OrderedDict
from flearn.clients.local import LocalClient
from flearn.config.config_paths import (
    DUMP_JSON_RESULT_PATH,
    TEST_STATS_FILE_NAME,
    TEST_ALL_STATS_FILE_NAME,
    FIGURES_PATH,
)


class LocalServer(BaseServer):
    def __init__(self, params):
        print("Using Federated avg to Train")
        super().__init__(params)

        self.clients: list[LocalClient] = self.clients
        # self.global_test = True

    def train(self):
        """Train using Federated Averaging"""
        print("Training with {} workers ---".format(self.clients_per_round))

        for i in trange(self.num_rounds, desc=self.desc):
            # test model
            self.eval(i, self.set_client_model_test)
            if self.loss_converged: break

            selected_clients: list[LocalClient] = self.select_clients(
                i, num_clients=min(self.clients_per_round, len(self.clients))
            )  # uniform sampling

            for _, c in enumerate(selected_clients):  

                # solve minimization locally
                c.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size
                )
        self.eval_end()

    def set_client_model_test(self, client: LocalClient):
        pass # Parameters not updated in this approach

    '''def eval(
        self,
        round: int,
        **ClientModelFuncKwargs,
    ):
        if round % self.eval_every == 0:
            stats = self.test()  # have set the latest model for all clients
            stats_train = self.train_error_and_loss()
            acc, loss = np.sum(stats[3]) * 1.0 / np.sum(stats[2]), np.dot(
                stats_train[4], stats_train[2]
            ) * 1.0 / np.sum(stats_train[2])
            self.accuracy_global.append(acc)
            self.loss_global.append(loss)
            train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
            tqdm.write(
                "At round {} accuracy: {}".format(round, acc)
            )  # testing accuracy
            tqdm.write("At round {} training accuracy: {}".format(round, train_acc))
            tqdm.write("At round {} training loss: {}".format(round, loss))

            self.filewriter.writerow(
                filename=TEST_STATS_FILE_NAME,
                row=[round, loss, train_acc, acc],
            )

        if self.all_test and ((round + 1) % 10 == 0):
            tqdm.write("Testing all clients for round: {}".format(round))
            self.test_all(server_round=round)

    def test(self, modelInCPU: bool = False):
        """Tests self.latest_model on given clients"""
        num_samples = []
        tot_correct = []
        for c1 in self.clients:
            for c in self.clients:
                ct, ns = c.test_model(model=c1.model, modelInCPU=modelInCPU)
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def train_error_and_loss___(self, modelInCPU: bool = False):
        print(f'Calling Local train and loss')
        num_samples = []
        tot_correct = []
        losses = []
        # print(f'\nBefore: clients-> {self.clients}\nnum_samples: {num_samples}, total_correct: {tot_correct}, losses: {losses}')
        for c1 in self.clients:
            for c in self.clients:
                ct, cl, ns = c.train_error_and_loss_model(model=c1.model,modelInCPU=modelInCPU)
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
                losses.append(cl * 1.0)
        # print(f'\nAfter:\nnum_samples: {num_samples}, total_correct: {tot_correct}, losses: {losses}')

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses'''