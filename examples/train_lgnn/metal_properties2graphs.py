import re, glob, pickle, ase, json
import numpy as np
from cheatools.graphtools import atoms2graph
from tqdm import tqdm

# Metal to exclude from training/validation and use for testing
metal_to_change = 'Ru'

# Load metal properties
with open('metal_properties.json', 'r') as f:
    metal_data = json.load(f)['metal_properties_integrated']

# Define properties to use
property_keys = [
    'atomic_radius',
    'energy_band',
    'ionization_potential',
    'band_structure_width',
    'nuclear_charge',
    'electron_affinity'
]

# Normalize properties
def normalize_properties():
    normalized_props = {}
    property_stats = {key: [] for key in property_keys}
    
    # Collect all values for each property
    for metal in metal_data:
        for key in property_keys:
            property_stats[key].append(metal_data[metal][key])
    
    # Calculate mean and std for each property
    stats = {
        key: {
            'mean': np.mean(property_stats[key]),
            'std': np.std(property_stats[key])
        }
        for key in property_keys
    }
    
    # Normalize each metal's properties
    for metal in metal_data:
        normalized_props[metal] = np.array([
            (metal_data[metal][key] - stats[key]['mean']) / stats[key]['std']
            for key in property_keys
        ], dtype=np.float32)
    
    return normalized_props, stats

# Get normalized properties
normalized_metal_props, norm_stats = normalize_properties()

# For H and O, we'll use zero vectors with same length as metal properties
non_metal_props = {
    'H': np.zeros(len(property_keys), dtype=np.float32),
    'O': np.zeros(len(property_keys), dtype=np.float32)
}

# import gasphase reference
e_h20 = ase.io.read('../gpaw/refs/h2o.traj',-1).get_potential_energy()
e_h2 = ase.io.read('../gpaw/refs/h2.traj',-1).get_potential_energy()
ref_dict = {'O':e_h20-e_h2, 'OH':e_h20-e_h2/2}

# Define valid elements (needed for graph creation)
valid_elements = list(normalized_metal_props.keys()) + list(non_metal_props.keys())

# Modified atoms2graph function to use properties instead of one-hot
def atoms2graph_with_properties(atoms, normalized_props, non_metal_props):
    # Get the original graph structure
    g = atoms2graph(atoms, valid_elements)
    
    # Replace one-hot encoding with property vectors
    new_features = []
    for atom in atoms:
        if atom.symbol in normalized_props:
            props = normalized_props[atom.symbol]
        else:
            props = non_metal_props[atom.symbol]
        new_features.append(props)
    
    # Convert features to numpy array
    g.x = np.array(new_features, dtype=np.float32)
    return g

# Save normalization stats for later use
stats_data = {
    'property_keys': property_keys,
    'normalization_stats': norm_stats
}
with open('property_stats.json', 'w') as f:
    json.dump(stats_data, f, indent=2)

# First create train and val sets WITHOUT specified metal
for s in ['train','val']:     
    slab_paths = glob.glob(f'../gpaw/{s}/*_slab.traj')
    graph_list = []
    
    # slab loop
    for sp in tqdm(slab_paths, desc=f'Processing {s} set', total=len(slab_paths)):
        slabId = re.findall(r'\d{4}', sp)[0]
        slab = ase.io.read(sp,'-1')
        
        # Skip if slab contains specified metal
        if metal_to_change in [atom.symbol for atom in slab]:
            continue
            
        slab_e = slab.get_potential_energy()
        
        # adsorbates on current slabId 
        ads_paths = glob.glob(f'../gpaw/{s}/{str(slabId).zfill(4)}_ads*.traj')
        
        # adsorbate loop
        for ap in ads_paths:
            atoms = ase.io.read(ap,'-1')

            ads_e = atoms.get_potential_energy()
            ads = ''.join([a.symbol for a in atoms if a.tag == 0])
            e = ads_e - slab_e - ref_dict[ads]
            
            g = atoms2graph_with_properties(atoms, normalized_metal_props, non_metal_props)
            g.y = e
            graph_list.append(g)

    print(f'Created {len(graph_list)} graphs for {s} set (excluding {metal_to_change} structures)')
    with open(f'graphs/{s}_properties.graphs', 'wb') as output:
        pickle.dump(graph_list, output)

# Then create test set WITH ONLY specified metal
slab_paths = glob.glob(f'../gpaw/test/*_slab.traj')
metal_graph_list = []

for sp in tqdm(slab_paths, desc=f'Processing {metal_to_change} test set', total=len(slab_paths)):
    slabId = re.findall(r'\d{4}', sp)[0]
    slab = ase.io.read(sp,'-1')
    
    # Only process if slab contains specified metal
    if metal_to_change not in [atom.symbol for atom in slab]:
        continue
        
    slab_e = slab.get_potential_energy()
    
    # adsorbates on current slabId 
    ads_paths = glob.glob(f'../gpaw/test/{str(slabId).zfill(4)}_ads*.traj')
    
    # adsorbate loop
    for ap in ads_paths:
        atoms = ase.io.read(ap,'-1')

        ads_e = atoms.get_potential_energy()
        ads = ''.join([a.symbol for a in atoms if a.tag == 0])
        e = ads_e - slab_e - ref_dict[ads]
        
        g = atoms2graph_with_properties(atoms, normalized_metal_props, non_metal_props)
        g.y = e
        metal_graph_list.append(g)

print(f'Created {len(metal_graph_list)} graphs for {metal_to_change} test set')
with open(f'graphs/{metal_to_change.lower()}_test_properties.graphs', 'wb') as output:
    pickle.dump(metal_graph_list, output)

# Print feature dimension information
print("\nFeature vector information:")
print(f"Number of properties per atom: {len(property_keys)}")
print("Properties used:", property_keys)