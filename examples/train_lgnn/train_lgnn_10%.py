import pickle, torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from cheatools.lgnn import lGNN
from copy import deepcopy
from sklearn.model_selection import StratifiedShuffleSplit

for s in ['train','val']:
    with open(f'graphs/{s}.graphs', 'rb') as input:
        globals()[f'{s}_graphs'] = pickle.load(input)

def create_stratified_subset(graphs, test_size=0.1, random_state=42):
    y = np.array([graph.y for graph in graphs])
    
    print("\nOriginal energy distribution:")
    print(f"Mean: {np.mean(y):.3f}")
    print(f"Std: {np.std(y):.3f}")
    print(f"Min: {np.min(y):.3f}")
    print(f"Max: {np.max(y):.3f}")
    
    bins = np.percentile(y, [0, 25, 50, 75, 100])
    y_binned = np.digitize(y, bins[:-1]) 
    
    print("\nSamples per bin:")
    for i in range(1, len(bins)):
        count = np.sum(y_binned == i)
        print(f"Bin {i} ({bins[i-1]:.3f} to {bins[i]:.3f} eV): {count} samples")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-test_size, random_state=random_state)
    
    for subset_idx, _ in sss.split(range(len(graphs)), y_binned):
        subset_graphs = [graphs[i] for i in subset_idx]
        
        y_subset = np.array([graph.y for graph in subset_graphs])
        print("\nSubset energy distribution:")
        print(f"Mean: {np.mean(y_subset):.3f}")
        print(f"Std: {np.std(y_subset):.3f}")
        print(f"Min: {np.min(y_subset):.3f}")
        print(f"Max: {np.max(y_subset):.3f}")
        
        return subset_graphs

print(f"Original training set size: {len(train_graphs)}")
train_graphs_subset = create_stratified_subset(train_graphs, test_size=0.3)
print(f"Subset training set size: {len(train_graphs_subset)}")

filename = 'lGNN_stratified_30percent'
torch.manual_seed(42)

batch_size = 64
max_epochs = 1000
learning_rate = 1e-3

roll_val_width = 20
patience = 100
report_every = 25

arch = {
    'n_conv_layers': 3,
    'n_hidden_layers': 0,
    'conv_dim': 18,
    'act': 'relu',
}

model = lGNN(arch=arch)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader = DataLoader(train_graphs_subset, batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=len(val_graphs), drop_last=True, shuffle=False)

train_loss, val_loss = [], []
model_states = []

for epoch in range(max_epochs):
    train_loss.append(model.train4epoch(train_loader, batch_size, opt))
    pred, target, _ = model.test(val_loader, len(val_graphs))
    val_mae = abs(np.array(pred) - np.array(target)).mean()
    val_loss.append(val_mae)
    model_states.append(deepcopy(model.state_dict()))

    if epoch >= roll_val_width+patience:
        roll_val = np.convolve(val_loss, np.ones(int(roll_val_width+1)), 'valid') / int(roll_val_width+1)
        min_roll_val = np.min(roll_val[:-patience+1])
        improv = (roll_val[-1] - min_roll_val) / min_roll_val

        if improv > - 0.01:
            print('Early stopping invoked.')
            best_epoch = np.argmin(val_loss)
            best_state = model_states[best_epoch]
            break

    if epoch % report_every == 0:
        print(f'Epoch {epoch} train and val L1Loss: {train_loss[-1]:.3f} / {val_loss[-1]:.3f} eV')

print(f'Finished training sequence. Best epoch was {best_epoch} with val. L1Loss {np.min(val_loss):.3f} eV')

best_state['onehot_labels'] = train_graphs_subset[0].onehot_labels
best_state['arch'] = arch

with open(f'{filename}.state', 'wb') as output:
    pickle.dump(best_state, output)

fig, main_ax = plt.subplots(1, 1, figsize=(8, 5))
color = ['steelblue','green']
label = [r'Training set  L1Loss',r'Validation set L1Loss']

for i, results in enumerate([train_loss, val_loss]):
    main_ax.plot(range(len(results)), results, color=color[i], label=label[i])
    if i == 1:
        main_ax.scatter(best_epoch, val_loss[best_epoch], facecolors='none', edgecolors='maroon', label='Best epoch', s=50, zorder=10)

main_ax.set_xlabel(r'Epoch', fontsize=16)
main_ax.set_ylabel(r'L1Loss [eV]', fontsize=16)
main_ax.set(ylim=(0.025,0.125))
main_ax.legend()

plt.savefig(f'{filename}_curve.png')