import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch_geometric.loader import DataLoader

with open('graphs/train.graphs', 'rb') as f:
    train_graphs = pickle.load(f)
with open('graphs/val.graphs', 'rb') as f:
    val_graphs = pickle.load(f)

def extract_features(graph):
    node_features = graph.x
    
    mean_features = node_features.mean(dim=0).numpy()
    
    std_features = node_features.std(dim=0).numpy()
    
    return np.concatenate([mean_features, std_features])


X_train = np.array([extract_features(g) for g in train_graphs])
y_train = np.array([g.y for g in train_graphs])

X_val = np.array([extract_features(g) for g in val_graphs])
y_val = np.array([g.y for g in val_graphs])

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print(f'Validation MAE: {mae:.3f} eV')

with open('linear_regression_baseline.pkl', 'wb') as f:
    pickle.dump(model, f)