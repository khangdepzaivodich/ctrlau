import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau.drawio')
root = tree.getroot()

all_node_ids = set()
all_edge_ids = set()
broken_edges = []

for cell in root.iter('mxCell'):
    cid = cell.get('id')
    is_edge = cell.get('edge') == '1'
    if is_edge:
        all_edge_ids.add(cid)
        s = cell.get('source')
        t = cell.get('target')
        # check later
    else:
        all_node_ids.add(cid)

for cell in root.iter('mxCell'):
    if cell.get('edge') == '1':
        cid = cell.get('id')
        s = cell.get('source')
        t = cell.get('target')
        if s and s not in all_node_ids:
            broken_edges.append((cid, 'source', s))
        if t and t not in all_node_ids:
            broken_edges.append((cid, 'target', t))

print(f"Total Nodes: {len(all_node_ids)}")
print(f"Total Edges: {len(all_edge_ids)}")
print(f"Broken Edge References: {len(broken_edges)}")
if broken_edges:
    for b in broken_edges:
        print(f"  Edge {b[0]} has missing {b[1]} id: {b[2]}")
else:
    print("ALL edge source and target IDs are 100% valid and verified!")
