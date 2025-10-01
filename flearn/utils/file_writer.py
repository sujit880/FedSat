import os
import json
from typing import Any
from flearn.config.config_paths import RESULTS_DIR
import csv


class FileWriter:
    def __init__(
        self,
        dataset_name: str,
        dataset_type: str,
        trainer: str,
        model: str,
        n_class: int,
        loss: str,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        rounds: int,
        num_clients: int,
        clients_per_round: int,
        experiment_name: str,
        test_mode: str,
        metadata: dict[str, Any] = {},
        buffer_limit: int = 5,
    ) -> None:

        if model is None or model == "None":
            model = "MLP"

        lr = str(learning_rate).replace(".", "_")

        self.experiment_name = experiment_name
        self.path = f"./{RESULTS_DIR}/results/{dataset_name}_{dataset_type}_{test_mode}/{self.experiment_name}/"
        os.makedirs(self.path, exist_ok=True)

        self.metadata: dict[str, Any] = {
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "trainer": trainer,
            "model": model,
            "num_class": n_class,
            "loss_type": loss,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "client_epochs": epochs,
            "server_rounds": rounds,
            "num_clients": num_clients,
            "clients_per_round": clients_per_round,

        }

        self.metadata.update(metadata)

        with open(self.path + "metadata.json", "w") as file:
            json.dump(self.metadata, file)

        self.csv_buffers: dict[str, list[list[Any]]] = {}
        self.buffer_limit = buffer_limit
        self.buffer_size = 0

    def flush_csv_buffers(self) -> None:
        for filename, data in self.csv_buffers.items():
            if len(data) == 0:
                continue

            with open(self.path + filename, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)

            self.csv_buffers[filename] = []
            self.buffer_size = 0

    def add_csv_file(self, filename: str, headers: list[str]) -> None:
        with open(self.path + filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            self.csv_buffers[filename] = []

    def writerow(self, filename: str, row: list[Any]) -> None:
        if filename not in self.csv_buffers.keys():
            raise RuntimeError("FileWriter Writing to uninitialized file")

        self.csv_buffers[filename].append(row)
        self.buffer_size += 1

        if self.buffer_size >= self.buffer_limit:
            self.flush_csv_buffers()

    def writerows(self, filename: str, rows: list[list[Any]]) -> None:
        if filename not in self.csv_buffers.keys():
            raise RuntimeError("FileWriter Writing to uninitialized file")

        self.csv_buffers[filename].extend(rows)
        self.buffer_size += len(rows)

        if self.buffer_size >= self.buffer_limit:
            self.flush_csv_buffers()
