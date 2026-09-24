import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau.drawio')
root = tree.getroot()

print("=== ALL CELLS ===")
for cell in root.iter('mxCell'):
    cid = cell.get('id', '')
    val = cell.get('value', '').replace('\n', ' ')
    geom = cell.find('mxGeometry')
    x = geom.get('x') if geom is not None else None
    y = geom.get('y') if geom is not None else None
    w = geom.get('width') if geom is not None else None
    h = geom.get('height') if geom is not None else None
    style = cell.get('style', '')
    is_edge = cell.get('edge') == '1'
    source = cell.get('source', '')
    target = cell.get('target', '')
    parent = cell.get('parent', '')

    if 'image=' in style:
        style_short = 'image=[...]'
    else:
        style_short = style[:60]

    pos = f"({x}, {y}, {w}x{h})" if x is not None else ""
    if is_edge:
        print(f"E: {cid} | {source} -> {target} | '{val}' | style={style_short}")
    else:
        print(f"N: {cid} | p={parent} | {pos} | '{val}' | style={style_short}")
