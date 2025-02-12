import re, glob, pickle, ase
from cheatools.graphtools import atoms2graph
from tqdm import tqdm

metal_to_change = 'Ru'  # Single control variable - change this to switch metals

# import gasphase reference
e_h20 = ase.io.read('../gpaw/refs/h2o.traj',-1).get_potential_energy()
e_h2 = ase.io.read('../gpaw/refs/h2.traj',-1).get_potential_energy()
ref_dict = {'O':e_h20-e_h2, 'OH':e_h20-e_h2/2}

# set onehot labels for node vectors -> stored in data_object.onehot_labels
onehot_labels = ['Ag','Ir','Pd','Pt','Ru','H','O']

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
            
            g = atoms2graph(atoms, onehot_labels)
            g.y = e
            graph_list.append(g)

    print(f'Created {len(graph_list)} graphs for {s} set (excluding {metal_to_change} structures)')
    with open(f'graphs/{s}.graphs', 'wb') as output:
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
        
        g = atoms2graph(atoms, onehot_labels)
        g.y = e
        metal_graph_list.append(g)

print(f'Created {len(metal_graph_list)} graphs for {metal_to_change} test set')
with open(f'graphs/{metal_to_change.lower()}_test.graphs', 'wb') as output:
    pickle.dump(metal_graph_list, output)