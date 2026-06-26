import json, sys

nb_path = sys.argv[1]
nb = json.load(open(nb_path, encoding="utf-8"))

for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    tag = c["cell_type"].upper()
    print(f"\n{'='*80}")
    print(f"CELL {i} [{tag}]")
    print(f"{'='*80}")
    print(src[:2000])
