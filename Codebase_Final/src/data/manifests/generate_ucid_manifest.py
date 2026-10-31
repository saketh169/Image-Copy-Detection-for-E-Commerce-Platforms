import json
import random
from pathlib import Path

def create_manifest(data_dir, out_path, num_pairs=2000):
    images = list(Path(data_dir).glob("*.jpg"))
    
    # Group by image ID (part before _)
    groups = {}
    for p in images:
        img_id = p.name.split("_")[0]
        if img_id not in groups:
            groups[img_id] = []
        groups[img_id].append(str(p.absolute()))
        
    ids = list(groups.keys())
    
    manifest = []
    
    # Generate 50% positive (copies) and 50% negative (different products)
    pos_count = num_pairs // 2
    neg_count = num_pairs - pos_count
    
    for i in range(pos_count):
        id_val = random.choice(ids)
        while len(groups[id_val]) < 2:
            id_val = random.choice(ids)
            
        p1, p2 = random.sample(groups[id_val], 2)
        pair_type = random.choice(["soft_pos", "hard_pos"]) # Using their taxonomy
        
        manifest.append({
            "row_id": f"ucid_pos_{i}",
            "path_1": p1,
            "path_2": p2,
            "label": 1,
            "pair_type": pair_type,
            "priority": 0.0,
            "times_trained": 0,
            "pair_id": f"{pair_type}_{i}"
        })
        
    for i in range(neg_count):
        id1, id2 = random.sample(ids, 2)
        p1 = random.choice(groups[id1])
        p2 = random.choice(groups[id2])
        pair_type = random.choice(["soft_neg", "hard_neg"])
        
        manifest.append({
            "row_id": f"ucid_neg_{i}",
            "path_1": p1,
            "path_2": p2,
            "label": 0,
            "pair_type": pair_type,
            "priority": 0.0,
            "times_trained": 0,
            "pair_id": f"{pair_type}_{i}"
        })
        
    random.shuffle(manifest)
    
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Generated {len(manifest)} pairs at {out_path}")

create_manifest("/home/saketh/UCID_Processed_Randomized/train", "/home/saketh/Codebase/ucid_manifest.json", 6000)
