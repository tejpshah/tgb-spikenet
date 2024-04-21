import argparse
import time
import os 
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from tgb.nodeproppred.evaluate import Evaluator
import json 
from datetime import datetime 
import itertools 
from spikenet import dataset, neuron
from spikenet.layers import SAGEAggregator
from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed, tab_printer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpikeNet(nn.Module):
    def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5,
                 dropout=0.7, bias=True, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='LIF'):

        super().__init__()

        tau = 1.0
        if sampler == 'rw':
            self.sampler = [RandomWalkSampler(
                add_selfloops(adj_matrix)) for adj_matrix in data.adj]
            self.sampler_t = [RandomWalkSampler(add_selfloops(
                adj_matrix)) for adj_matrix in data.adj_evolve]
        elif sampler == 'sage':
            self.sampler = [Sampler(add_selfloops(adj_matrix))
                            for adj_matrix in data.adj]
            self.sampler_t = [Sampler(add_selfloops(adj_matrix))
                              for adj_matrix in data.adj_evolve]
        else:
            raise ValueError(sampler)

        aggregators, snn = nn.ModuleList(), nn.ModuleList()

        for hid in hids:
            aggregators.append(SAGEAggregator(in_features, hid,
                                              concat=concat, bias=bias,
                                              aggr=aggr))

            if act == "IF":
                snn.append(neuron.IF(alpha=alpha, surrogate=surrogate))
            elif act == 'LIF':
                snn.append(neuron.LIF(tau, alpha=alpha, surrogate=surrogate))
            elif act == 'PLIF':
                snn.append(neuron.PLIF(tau, alpha=alpha, surrogate=surrogate))
            else:
                raise ValueError(act)

            in_features = hid * 2 if concat else hid

        self.aggregators = aggregators
        self.dropout = nn.Dropout(dropout)
        self.snn = snn
        self.sizes = sizes
        self.p = p
        self.pooling = nn.Linear(len(data) * in_features, out_features)

    def encode(self, nodes):
        spikes = []
        sizes = self.sizes
        for time_step in range(len(data)):

            snapshot = data[time_step]
            sampler = self.sampler[time_step]
            sampler_t = self.sampler_t[time_step]

            x = snapshot.x
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes
            for size in sizes:
                size_1 = max(int(size * self.p), 1)
                size_2 = size - size_1

                if size_2 > 0:
                    nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
                    nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
                    nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
                else:
                    nbr = sampler(nbr, size_1).view(-1)

                num_nodes.append(nbr.size(0))
                h.append(x[nbr].to(device))

            for i, aggregator in enumerate(self.aggregators):
                self_x = h[:-1]
                neigh_x = []
                for j, n_x in enumerate(h[1:]):
                    neigh_x.append(n_x.view(-1, sizes[j], h[0].size(-1)))

                out = self.snn[i](aggregator(self_x, neigh_x))
                if i != len(sizes) - 1:
                    out = self.dropout(out)
                    h = torch.split(out, num_nodes[:-(i + 1)])

            spikes.append(out)
        spikes = torch.cat(spikes, dim=1)
        neuron.reset_net(self)
        return spikes

    def forward(self, nodes):
        spikes = self.encode(nodes)
        return self.pooling(spikes)

def train(model, optimizer, loss_fn, train_loader, y, device):
    model.train()
    for nodes in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        loss = loss_fn(model(nodes), y[nodes])
        loss.backward()
        optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, loader, y, device, evaluator, eval_metric):
    model.eval()
    logits = []
    labels = []
    for nodes in loader:
        logits.append(model(nodes))
        labels.append(y[nodes])
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    input_dict = {
        "y_true": labels,
        "y_pred": logits,
        "eval_metric": [eval_metric],
    }
    result_dict = evaluator.eval(input_dict)
    score = result_dict[eval_metric]
    return score

def run_experiment(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y = data.y.to(device)

    train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
                            pin_memory=False, batch_size=200000, shuffle=False)
    test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)

    model = SpikeNet(data.num_features, data.num_classes, alpha=args.alpha,
                     dropout=args.dropout, sampler=args.sampler, p=args.p,
                     aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
                     hids=args.hids, act=args.neuron, bias=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_metric = test_metric = 0
    best_train_loss = float('inf')
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, optimizer, loss_fn, train_loader, y, device)
        val_metric = test(model, val_loader, y, device, evaluator, eval_metric)
        test_metric = test(model, test_loader, y, device, evaluator, eval_metric)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
            best_train_loss = train_loss

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}, Test: {test_metric:.4f}')

    # Save the best metrics and key properties in a JSON file
    metrics_data = {
        'dataset': args.dataset,
        'best_epoch': epoch,
        'best_train_loss': best_train_loss,
        'best_validation_score': best_val_metric,
        'best_test_score': best_test_metric,
        'hidden_units': args.hids,
        'aggregation': args.aggr,
        'sampling_sizes': args.sizes,
        'surrogate': args.surrogate,
        'neuron': args.neuron,
        'alpha': args.alpha,
        'dropout': args.dropout,
        'p': args.p,
        'batch_size': args.batch_size,
        'total_time_elapsed': time.time() - start
    }

    return metrics_data

if __name__ == '__main__':
    # Argument setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="?", default="tgbn",
                        help="Datasets (DBLP, Tmall, Patent). (default: TGBN)")
    parser.add_argument("--tgbn_dataset", nargs="?", default="tgbn-trade",
                        help="Specific dataset for TGBN (default: tgbn-trade)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='Learning rate for training. (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed for model. (default: 2022)')

    args = parser.parse_args()

    # Fixed splits for training, validation, and testing
    args.train_size = 0.7
    args.val_size = 0.15
    args.test_size = 0.15
    args.split_seed = 42  # Random seed for splitting nodes
    args.sampler = 'sage'
    args.aggr = 'mean'
    args.surrogate='sigmoid'

    # Dataset selection
    if args.dataset.lower() == "dblp":
        data = dataset.DBLP()
    elif args.dataset.lower() == "tmall":
        data = dataset.Tmall()
    elif args.dataset.lower() == "patent":
        data = dataset.Patent()
    elif args.dataset.lower() == "tgbn":
        data = dataset.TGBN(name=args.tgbn_dataset)
    else:
        raise ValueError(f"{args.dataset} is invalid. Only datasets (DBLP, Tmall, Patent, TGBN) are available.")

    evaluator = Evaluator(name=args.tgbn_dataset)
    eval_metric = data.eval_metric

    data.split_nodes(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, random_state=args.split_seed)

    # Define the hyperparameter ranges
    alpha_range = [0.6, 0.8, 1.0]
    batch_size_range = [512, 1024]
    dropout_range = [0.1, 0.2, 0.3]
    hidden_units_range = [[128, 10], [256, 20], [128, 128], [10, 10], [256, 256]]
    sizes = [5, 2]  # Constant for all experiments
    p_range = [0.6, 0.8, 1.0]
    neuron_type = 'LIF'  # Constant for all experiments

    # Save the results
    os.makedirs('saved_results', exist_ok=True)

    # Load existing results
    try:
        with open('saved_results/last_run.json', 'r') as f:
            all_metrics_data = json.load(f)
    except FileNotFoundError:
        all_metrics_data = []
    except json.JSONDecodeError:
        print("Error reading the JSON file. Starting with an empty dataset.")
        all_metrics_data = []

    # Helper function to check if the current configuration has been run
    def is_duplicate_experiment(current_config, all_metrics):
        for data in all_metrics:
            if (data['alpha'] == current_config['alpha'] and
                data['batch_size'] == current_config['batch_size'] and
                data['dropout'] == current_config['dropout'] and
                data['hidden_units'] == current_config['hids'] and
                data['p'] == current_config['p'] and
                data['neuron'] == current_config['neuron'] and
                data['concat_flag'] == current_config['concat']):
                return True
        return False

    # Generate and run experiments for each combination
    for alpha in alpha_range:
        for batch_size in batch_size_range:
            for dropout in dropout_range:
                for hidden_units in hidden_units_range:
                    for p in p_range:
                        for concat_flag in [True, False]:
                            # Set up the current configuration
                            current_config = {
                                'concat': concat_flag,
                                'alpha': alpha,
                                'batch_size': batch_size,
                                'dropout': dropout,
                                'hids': hidden_units,
                                'sizes': sizes,
                                'p': p,
                                'neuron': neuron_type,
                                'concat_flag': concat_flag
                            }
                            
                            if not is_duplicate_experiment(current_config, all_metrics_data):
                                args.concat = concat_flag
                                args.alpha = alpha
                                args.batch_size = batch_size
                                args.dropout = dropout
                                args.hids = hidden_units
                                args.sizes = sizes
                                args.p = p
                                args.neuron = neuron_type
                                args.epochs = 250

                                print(f"Running experiment with hyperparameters: {alpha, batch_size, dropout, hidden_units, sizes, p, neuron_type, concat_flag}")
                                metrics_data = run_experiment(args)
                                all_metrics_data.append(metrics_data)

                                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                                with open(f'saved_results/all_metrics_{current_datetime}.json', 'w') as f:
                                    json.dump(all_metrics_data, f, indent=4)
                            else:
                                print(f"Skipping duplicate experiment for hyperparameters: {alpha, batch_size, dropout, hidden_units, sizes, p, neuron_type, concat_flag}")