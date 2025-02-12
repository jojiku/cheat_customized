import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from cheatools.plot import plot_parity

filename = 'linear_regression_baseline' 

with open(f'{filename}.pkl', 'rb') as input_file:
    regressor = pickle.load(input_file)

def extract_features(graph):
    node_features = graph.x
    
    mean_features = node_features.mean(dim=0).numpy()
    
    std_features = node_features.std(dim=0).numpy()
    
    return np.concatenate([mean_features, std_features])

with open('graphs/test.graphs', 'rb') as f:
    test_set = pickle.load(f)

X_test = np.array([extract_features(graph) for graph in test_set])
y_true = np.array([graph.y for graph in test_set])

y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_true, y_pred)
print(f'Validation MAE (Linear Regression): {mae:.3f} eV')

ads_list = [graph.ads for graph in test_set]  

true_dict = {ads: [] for ads in ['O', 'OH']}
pred_dict = {ads: [] for ads in ['O', 'OH']}

for i, p in enumerate(y_pred):
    adsorbate = ads_list[i]
    if adsorbate in true_dict:
        true_dict[adsorbate].append(y_true[i])
        pred_dict[adsorbate].append(p)

colors = ['firebrick', 'steelblue']
header = 'Linear Regression Baseline'


fig = plot_parity(true_dict, pred_dict, colors, header, [-0.75, 2.25])
plt.savefig(f'baseline_parity/{filename}_test.png')
plt.show()