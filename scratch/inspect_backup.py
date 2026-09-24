import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau_backup.drawio')
root = tree.getroot()

for c in root.iter('mxCell'):
    val = c.get('value', '')
    cid = c.get('id', '')
    style = c.get('style', '')
    geom = c.find('mxGeometry')
    x = geom.get('x') if geom is not None else ''
    y = geom.get('y') if geom is not None else ''
    w = geom.get('width') if geom is not None else ''
    h = geom.get('height') if geom is not None else ''
    edge = c.get('edge')
    vertex = c.get('vertex')
    src = c.get('source')
    tgt = c.get('target')
    
    if edge == '1':
        print(f"EDGE: [{cid}] {src} -> {tgt} : {val}")
    elif val:
        val_clean = val.replace('\n', ' ')[:60]
        print(f"NODE: [{cid}] (x={x}, y={y}, w={w}, h={h}): {val_clean}")
