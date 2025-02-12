import pickle, torch, json
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from cheatools.lgnn import lGNN
from copy import deepcopy
from datetime import datetime

metal_to_change = 'Ru'   

# Load metal properties to get feature dimension
with open('metal_properties.json', 'r') as f:
    metal_data = json.load(f)['metal_properties_integrated']
n_features = len(metal_data[list(metal_data.keys())[0]])  # Number of properties per atom

# load train, validation and metal test sets
test_set_name = f'{metal_to_change.lower()}_test'
for s in ['train', 'val', test_set_name]:
    with open(f'graphs/{s}_properties.graphs', 'rb') as input:
        globals()[f'{s}_graphs'] = pickle.load(input)

print(f"Loaded {len(train_graphs)} training graphs (no {metal_to_change})")
print(f"Loaded {len(val_graphs)} validation graphs (no {metal_to_change})")
print(f"Loaded {len(globals()[f'{test_set_name}_graphs'])} {metal_to_change} test graphs")

filename = f'lGNN_{metal_to_change.lower()}_properties_extrapolation'
torch.manual_seed(42)

# set Dataloader batch size, learning rate and max epochs
batch_size = 64
max_epochs = 1000
learning_rate = 1e-3

# early stopping parameters
roll_val_width = 20  
patience = 100
report_every = 25

# set lGNN architecture
# Modified to account for property-based input features
arch = {
    'n_conv_layers': 3,
    'n_hidden_layers': 1,  # Added one hidden layer
    'conv_dim': 32,
    'act': 'relu',
    'input_dim': n_features
}

# load model, optimizer, and dataloaders
model = lGNN(arch=arch)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader = DataLoader(train_graphs, batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=len(val_graphs), drop_last=True, shuffle=False)
metal_test_loader = DataLoader(globals()[f'{test_set_name}_graphs'], 
                             batch_size=len(globals()[f'{test_set_name}_graphs']), 
                             drop_last=True, shuffle=False)

# Ensure all graphs have tensor features
def ensure_tensor_features(graph_list):
    for graph in graph_list:
        if isinstance(graph.x, list):
            graph.x = torch.tensor(graph.x, dtype=torch.float32)
    return graph_list

train_graphs = ensure_tensor_features(train_graphs)
val_graphs = ensure_tensor_features(val_graphs)
globals()[f'{test_set_name}_graphs'] = ensure_tensor_features(globals()[f'{test_set_name}_graphs'])


# initialize arrays for training, validation and test error
train_loss, val_loss = [], []
model_states = []

# Create timestamp for saving results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# epoch loop
best_val_loss = float('inf')
for epoch in range(max_epochs):
    train_loss.append(model.train4epoch(train_loader, batch_size, opt))
    
    # Validation on non-metal structures
    pred, target, _ = model.test(val_loader, len(val_graphs))
    val_mae = abs(np.array(pred) - np.array(target)).mean()
    val_loss.append(val_mae)
    model_states.append(deepcopy(model.state_dict()))

    if epoch >= roll_val_width + patience:
        roll_val = np.convolve(val_loss, np.ones(int(roll_val_width+1)), 'valid') / int(roll_val_width+1)
        min_roll_val = np.min(roll_val[:-patience+1])
        improv = (roll_val[-1] - min_roll_val) / min_roll_val

        if improv > -0.01:
            print('Early stopping invoked.')
            best_epoch = np.argmin(val_loss)
            best_state = model_states[best_epoch]
            break

    if val_mae < best_val_loss:
        best_val_loss = val_mae
        best_epoch = epoch
        best_state = deepcopy(model.state_dict())

    if epoch % report_every == 0:
        print(f'Epoch {epoch} train/val L1Loss: {train_loss[-1]:.3f} / {val_loss[-1]:.3f} eV')

# Load best model and evaluate on metal test set
model.load_state_dict(best_state)
metal_pred, metal_target, _ = model.test(metal_test_loader, len(globals()[f'{test_set_name}_graphs']))
metal_mae = abs(np.array(metal_pred) - np.array(metal_target)).mean()

print(f'Training finished:')
print(f'Best epoch: {best_epoch}')
print(f'Best validation L1Loss: {np.min(val_loss):.3f} eV')
print(f'{metal_to_change} test set L1Loss: {metal_mae:.3f} eV')

# Save training results
best_state['properties_info'] = {
    'n_features': n_features,
    'property_names': list(metal_data[list(metal_data.keys())[0]].keys())
}
best_state['arch'] = arch

with open(f'{filename}_{timestamp}.state', 'wb') as output:
    pickle.dump(best_state, output)

# Plot training curves and results
plt.figure(figsize=(8, 12))

# Training and validation curves
plt.subplot(3, 1, 1)
plt.plot(train_loss, 'steelblue', label='Training L1Loss')
plt.plot(val_loss, 'green', label='Validation L1Loss')
plt.scatter(best_epoch, val_loss[best_epoch], facecolors='none', edgecolors='maroon', 
           label='Best epoch', s=50, zorder=10)
plt.xlabel('Epoch')
plt.ylabel('L1Loss [eV]')
plt.legend()
plt.title('Training Progress')

# Parity plot for metal predictions
plt.subplot(3, 1, 2)
plt.scatter(metal_target, metal_pred, alpha=0.5, color='silver', 
           label=f'{metal_to_change} predictions')
plt.plot([min(metal_target), max(metal_target)], [min(metal_target), max(metal_target)], 
         'k--', label='Perfect prediction')
plt.xlabel('DFT Energy [eV]')
plt.ylabel('Predicted Energy [eV]')
plt.legend()
plt.title(f'{metal_to_change} Test Set Predictions\nMAE: {metal_mae:.3f} eV')

# Property correlation plot
plt.subplot(3, 1, 3)
errors = np.abs(np.array(metal_pred) - np.array(metal_target))
plt.hist(errors, bins=20, color='skyblue', alpha=0.7)
plt.xlabel('Absolute Error [eV]')
plt.ylabel('Count')
plt.title('Error Distribution on Test Set')

plt.tight_layout()
plt.savefig(f'{filename}_results_{timestamp}.png')
plt.close()

# Save numerical results
results = {
    'best_epoch': best_epoch,
    'best_val_loss': np.min(val_loss),
    f'{metal_to_change.lower()}_test_mae': metal_mae,
    'train_loss': train_loss,
    'val_loss': val_loss,
    f'{metal_to_change.lower()}_predictions': {'true': metal_target, 'pred': metal_pred},
    'property_info': best_state['properties_info']
}

with open(f'{filename}_results_{timestamp}.pkl', 'wb') as f:
    pickle.dump(results, f)