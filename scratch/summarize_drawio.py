import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau.drawio')
root = tree.getroot()

containers = []
nodes = []
edges = []

for cell in root.iter('mxCell'):
    cid = cell.get('id', '')
    val = cell.get('value', '')
    style = cell.get('style', '')
    parent = cell.get('parent', '')
    source = cell.get('source', '')
    target = cell.get('target', '')
    edge = cell.get('edge', '')
    geom = cell.find('mxGeometry')
    x = float(geom.get('x')) if geom is not None and geom.get('x') is not None else None
    y = float(geom.get('y')) if geom is not None and geom.get('y') is not None else None
    w = float(geom.get('width')) if geom is not None and geom.get('width') is not None else None
    h = float(geom.get('height')) if geom is not None and geom.get('height') is not None else None

    # check if container / large box
    if w is not None and h is not None and (w > 100 or h > 100) and edge != '1':
        containers.append((cid, val, x, y, w, h, style[:60]))
    elif edge == '1':
        edges.append((cid, source, target, val))
    else:
        nodes.append((cid, val, x, y, w, h, style[:40]))

print(f"Total containers/boxes: {len(containers)}")
for c in containers:
    print(f"Container [{c[0]}]: '{c[1]}' at ({c[2]}, {c[3]}) size ({c[4]}x{c[5]}) style={c[6]}")

print(f"\nTotal other nodes: {len(nodes)}")
for n in nodes:
    if n[1]: # non-empty label
        print(f"Node [{n[0]}]: '{n[1][:40]}' at ({n[2]}, {n[3]})")

print(f"\nTotal edges: {len(edges)}")
