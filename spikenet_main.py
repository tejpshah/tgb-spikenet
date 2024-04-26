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

from spikenet import dataset, neuron
from spikenet.layers import SAGEAggregator
from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed, tab_printer)

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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="DBLP",
                    help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
parser.add_argument('--tgbn_dataset', nargs="?", default="tgbn-trade")
parser.add_argument('--sizes', type=int, nargs='+', default=[5,2,2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
parser.add_argument('--hids', type=int, nargs='+',
                    default=[128, 128, 128], help='Hidden units for each layer. (default: [128, 10])')
parser.add_argument("--aggr", nargs="?", default="mean",
                    help="Aggregate function ('mean', 'sum'). (default: 'mean')")
parser.add_argument("--sampler", nargs="?", default="sage",
                    help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
parser.add_argument("--surrogate", nargs="?", default="sigmoid",
                    help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
parser.add_argument("--neuron", nargs="?", default="LIF",
                    help="Spiking neuron used for training. (IF, LIF, PLIF). (default: LIF")
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size for training. (default: 1024)')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning rate for training. (default: 5e-3)')
parser.add_argument('--train_size', type=float, default=0.4,
                    help='Ratio of nodes for training. (default: 0.4)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Smooth factor for surrogate learning. (default: 1.0)')
parser.add_argument('--p', type=float, default=0.5,
                    help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout probability. (default: 0.7)')
parser.add_argument('--epochs', type=int, default=250,
                    help='Number of training epochs. (default: 100)')
parser.add_argument('--concat', action='store_true',
                    help='Whether to concat node representation and neighborhood representations. (default: False)')
parser.add_argument('--seed', type=int, default=2022,
                    help='Random seed for model. (default: 2022)')


try:
    args = parser.parse_args()
    args.test_size = 1 - args.train_size
    args.train_size = args.train_size - 0.05
    args.val_size = 0.05
    args.split_seed = 42
    tab_printer(args)
except:
    parser.print_help()
    exit(0)

assert len(args.hids) == len(args.sizes), "must be equal!"

if args.dataset.lower() == "dblp":
    data = dataset.DBLP()
elif args.dataset.lower() == "tmall":
    data = dataset.Tmall()
elif args.dataset.lower() == "patent":
    data = dataset.Patent()
elif args.dataset.lower() == "tgbn":
    data = dataset.TGBN(name=args.tgbn_dataset)
    
else:
    raise ValueError(
        f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")

# TODO Error handling
evaluator = Evaluator(name=args.tgbn_dataset)
eval_metric = data.eval_metric

# train:val:test
data.split_nodes(train_size=args.train_size, val_size=args.val_size,
                 test_size=args.test_size, random_state=args.split_seed)

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


def train():
    model.train()
    for nodes in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        probabilities = torch.nn.functional.softmax(model(nodes), dim=0)
        loss_fn(probabilities, y[nodes]).backward()
        optimizer.step()
    return loss.item()

@torch.no_grad()
def test(loader):
    model.eval()
    logits = []
    labels = []
    for nodes in loader:
        logits.append(model(nodes))
        labels.append(y[nodes])
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()

    probabilities = torch.nn.functional.softmax(logits, dim=0)
    # print(labels)
    input_dict = {
        "y_true": labels,
        "y_pred": probabilities,
        "eval_metric": [eval_metric],
    }
    result_dict = evaluator.eval(input_dict)
    score = result_dict[eval_metric]
    return score

best_val_metric = test_metric = 0
best_train_loss = float('inf')
start = time.time()

for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_metric = test(val_loader)
    test_metric = test(test_loader)
    
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
        best_train_loss = train_loss
        
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}, Test: {test_metric:.4f}')

# Create saved_results folder if it doesn't exist
os.makedirs('saved_results', exist_ok=True)

# Get current datetime
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    'total_time_elapsed': time.time() - start
}

with open(f'saved_results/best_metrics_{current_datetime}.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)