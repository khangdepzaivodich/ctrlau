import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau.drawio')
root = tree.getroot()

print("Root tag:", root.tag)
diagram = root.find('.//diagram')
if diagram is not None:
    print("Diagram name:", diagram.get('name'))

cells = []
for cell in root.iter('mxCell'):
    cid = cell.get('id', '')
    val = cell.get('value', '')
    style = cell.get('style', '')
    parent = cell.get('parent', '')
    geom = cell.find('mxGeometry')
    geom_str = ''
    if geom is not None:
        geom_str = f"x={geom.get('x')}, y={geom.get('y')}, w={geom.get('width')}, h={geom.get('height')}"
    if 'image=' in style:
        style = style.split('image=')[0] + 'image=[IMAGE]...'
    if val:
        val = val.replace('\n', ' ')
        if len(val) > 80:
            val = val[:80] + '...'
    cells.append((cid, parent, val, style, geom_str))

print(f"Total cells: {len(cells)}")
print("\n--- Non-empty cells ---")
for c in cells:
    if c[2] or 'shape=' in c[3] or 'fillColor=' in c[3]:
        print(f"[{c[0]}] (P:{c[1]}) '{c[2]}' | Geom: {c[4]} | Style: {c[3][:60]}")
