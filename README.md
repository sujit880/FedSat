# FedVision
State of the art FL work for computer vision task


## Paper Information

### Key Publications Referenced

This project is inspired by and builds upon the concepts discussed in several key publications:

1. **FedProto: Federated Prototype Learning across Heterogeneous Clients**
      [View Paper](papers/20819-Article_Text-24832-1-2-20220628.pdf)


### Generate Dataset 
python generate_clients_dataset.py --dataset cifar --type noiid_lbldir --clients 100 --beta 0.5
python generate_clients_dataset.py --dataset mnist --type qty_lbl_imb --clients 100 --beta 0.5
python generate_clients_dataset.py --dataset mnist --type qty_lbl_imb --classes=6 --clients 100 --beta 0.5
### Run Experiment
1. python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --n_class=None --beta=0.3 --feature_noise=0.0 --domain=None --num_clients=100 --batch_size=64 --learning_rate=0.01 --n_class=10 --trainer=fedavg  --num_rounds=150
2.  python main.py --num_epochs=5 --clients_per_round=10 --dataset=cifar --dataset_type=noiid_lbldir --beta=0.3 --num_clients=100 --batch_size=64 --learning_rate=0.01 --model=tresnet18p --trainer=fedavg  --num_rounds=155 --loss=CE
