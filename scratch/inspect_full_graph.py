import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau.drawio')
root = tree.getroot()

id_to_val = {}
for cell in root.iter('mxCell'):
    cid = cell.get('id', '')
    val = cell.get('value', '').replace('\n', ' ')
    if not val:
        style = cell.get('style', '')
        if 'trapezoid' in style:
            val = '[Trapezoid]'
        elif 'shape=cube' in style:
            val = '[Cube/Feature]'
        elif 'ellipse' in style:
            val = '[Graph Node]'
        elif 'image=' in style:
            val = '[Face Image]'
        elif 'dashed=1' in style:
            val = '[Dashed Box]'
        elif cell.get('edge') == '1':
            val = '[Edge]'
        else:
            val = f"[Box {cid}]"
    id_to_val[cid] = val

print("=== ALL CELLS & POSITIONS ===")
for cell in root.iter('mxCell'):
    cid = cell.get('id', '')
    val = cell.get('value', '').replace('\n', ' ')
    geom = cell.find('mxGeometry')
    x = geom.get('x') if geom is not None else ''
    y = geom.get('y') if geom is not None else ''
    w = geom.get('width') if geom is not None else ''
    h = geom.get('height') if geom is not None else ''
    style = cell.get('style', '')
    is_edge = cell.get('edge') == '1'
    source = cell.get('source', '')
    target = cell.get('target', '')
    
    if is_edge:
        s_name = id_to_val.get(source, source)
        t_name = id_to_val.get(target, target)
        print(f"EDGE [{cid}] ({s_name}) ---> ({t_name}) | val: '{val}'")
    else:
        print(f"CELL [{cid}] at ({x},{y},{w}x{h}) | '{val}' | style={style[:40]}")
