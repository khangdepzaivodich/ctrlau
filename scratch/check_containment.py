import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau.drawio')
root = tree.getroot()

boxes = [
    ("box_vis_extract", -715, 25, 410, 240),
    ("box_clip_area", -715, 275, 410, 165),
    ("box_hsic", -290, 25, 420, 415),
    ("box_multibranch", 145, 25, 555, 415),
    ("box_au_au", -715, 525, 455, 275),
    ("box_bipartite", -245, 525, 480, 275),
    ("box_himod", 250, 525, 450, 275),
]

for name, bx, by, bw, bh in boxes:
    print(f"\n--- Checking contents of {name} (bounds: x=[{bx}, {bx+bw}], y=[{by}, {by+bh}]) ---")
    count = 0
    for cell in root.iter('mxCell'):
        if cell.get('edge') == '1':
            continue
        geom = cell.find('mxGeometry')
        if geom is None or geom.get('x') is None:
            continue
        x = float(geom.get('x'))
        y = float(geom.get('y'))
        w = float(geom.get('width'))
        h = float(geom.get('height'))
        cid = cell.get('id')
        if cid in [name, 'box_phase1', 'box_phase2', 'banner_phase1', 'banner_phase2']:
            continue
        # Check if inside or overlapping
        if (x >= bx - 10 and x + w <= bx + bw + 10 and
            y >= by - 10 and y + h <= by + bh + 10):
            count += 1
    print(f"  Total items inside: {count}")
