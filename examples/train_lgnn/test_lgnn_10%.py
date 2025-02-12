import pickle
from torch_geometric.loader import DataLoader
from cheatools.lgnn import lGNN
from cheatools.plot import plot_parity

filename = 'lGNN_stratified_30percent' 

with open(f'{filename}.state', 'rb') as input:
    regressor = lGNN(trained_state=pickle.load(input))

for s in ['train', 'val', 'test']:  
    with open(f'graphs/{s}.graphs', 'rb') as input:
        test_set = pickle.load(input)
    test_loader = DataLoader(test_set, batch_size=len(test_set), drop_last=True, shuffle=False)
    pred, true, ads = regressor.test(test_loader, len(test_set))

    mae = sum(abs(p - t) for p, t in zip(pred, true)) / len(pred)
    print(f"\n{s.upper()} set results:")
    print(f"Total samples: {len(pred)}")
    print(f"Overall MAE: {mae:.3f} eV")

    true_dict = {ads: [] for ads in ['O','OH']}
    pred_dict = {ads: [] for ads in ['O','OH']}

    for i, p in enumerate(pred):
        true_dict[ads[i]].append(true[i])
        pred_dict[ads[i]].append(pred[i])    

    for adsorbate in ['O', 'OH']:
        mae_ads = sum(abs(p - t) for p, t in zip(pred_dict[adsorbate], true_dict[adsorbate])) / len(pred_dict[adsorbate])
        print(f"{adsorbate} MAE: {mae_ads:.3f} eV (n={len(pred_dict[adsorbate])})")

    colors = ['firebrick','steelblue']
    arr = zip(true_dict.values(), pred_dict.values())
    header = r'LeanGNN IS2RE (30% Stratified Training)'  

    fig = plot_parity(true_dict, pred_dict, colors, header, [-0.75,2.25])
    fig.savefig(f'parity/{filename}_{s}.png')