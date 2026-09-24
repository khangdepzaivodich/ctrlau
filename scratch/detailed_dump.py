import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau.drawio')
root = tree.getroot()

for cell in root.iter('mxCell'):
    cid = cell.get('id', '')
    val = cell.get('value', '')
    style = cell.get('style', '')
    parent = cell.get('parent', '')
    source = cell.get('source', '')
    target = cell.get('target', '')
    edge = cell.get('edge', '')
    geom = cell.find('mxGeometry')
    geom_str = ''
    if geom is not None:
        geom_str = f"x={geom.get('x')}, y={geom.get('y')}, w={geom.get('width')}, h={geom.get('height')}"
    if 'image=' in style:
        style = style.split('image=')[0] + 'image=[IMG]...'
    val_clean = val.replace('\n', ' ')
    if edge == '1':
        print(f"EDGE [{cid}] s={source} -> t={target} | '{val_clean}' | style={style[:40]}")
    else:
        print(f"NODE [{cid}] (P:{parent}) '{val_clean}' | {geom_str} | style={style[:50]}")
