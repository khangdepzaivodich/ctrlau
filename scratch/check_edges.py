import xml.etree.ElementTree as ET
import re

tree = ET.parse('ctrlau.drawio')
root = tree.getroot()

nodes = {}
edges = []

for c in root.iter('mxCell'):
    cid = c.get('id')
    geom = c.find('mxGeometry')
    if c.get('vertex') == '1' and geom is not None:
        nodes[cid] = {
            'x': float(geom.get('x', 0)),
            'y': float(geom.get('y', 0)),
            'w': float(geom.get('width', 0)),
            'h': float(geom.get('height', 0)),
            'val': c.get('value', '')
        }
    elif c.get('edge') == '1':
        edges.append({
            'id': cid,
            'source': c.get('source'),
            'target': c.get('target'),
            'style': c.get('style', ''),
            'val': c.get('value', '')
        })

print(f'Total nodes: {len(nodes)}, Total edges: {len(edges)}')

def get_style_val(style, key, default=0.5):
    m = re.search(f'{key}=([0-9.]+)', style)
    return float(m.group(1)) if m else default

bents = []
for e in edges:
    s = nodes.get(e['source'])
    t = nodes.get(e['target'])
    if not s or not t:
        print('Missing node for edge:', e['id'], e['source'], '->', e['target'])
        continue
    style = e['style']
    
    exit_x = get_style_val(style, 'exitX', 0.5)
    exit_y = get_style_val(style, 'exitY', 0.5)
    entry_x = get_style_val(style, 'entryX', 0.5)
    entry_y = get_style_val(style, 'entryY', 0.5)
    
    s_x = s['x'] + s['w'] * exit_x
    s_y = s['y'] + s['h'] * exit_y
    t_x = t['x'] + t['w'] * entry_x
    t_y = t['y'] + t['h'] * entry_y
    
    # Check orthogonal straightness
    if 'edgeStyle=straightEdgeStyle' in style:
        print(f"DirectStraight-Edge [{e['id']}] {e['source']} -> {e['target']}: STRAIGHT")
    elif abs(exit_x - 1.0) < 0.01 and abs(entry_x - 0.0) < 0.01:
        diff_y = abs(s_y - t_y)
        status = 'STRAIGHT' if diff_y < 0.1 else f'BENT (diff_y={diff_y:.2f})'
        if diff_y >= 0.1:
            bents.append((e['id'], 'H', e['source'], e['target'], diff_y))
        print(f"H-Edge [{e['id']}] {e['source']} -> {e['target']}: {status} (s_y={s_y:.1f}, t_y={t_y:.1f})")
    elif abs(exit_y - 1.0) < 0.01 and abs(entry_y - 0.0) < 0.01:
        diff_x = abs(s_x - t_x)
        status = 'STRAIGHT' if diff_x < 0.1 else f'BENT (diff_x={diff_x:.2f})'
        if diff_x >= 0.1:
            bents.append((e['id'], 'V', e['source'], e['target'], diff_x))
        print(f"V-Edge [{e['id']}] {e['source']} -> {e['target']}: {status} (s_x={s_x:.1f}, t_x={t_x:.1f})")
    else:
        print(f"Other-Edge [{e['id']}] {e['source']} -> {e['target']}")

print(f"\nTotal bent edges: {len(bents)}")
for b in bents:
    print(b)
