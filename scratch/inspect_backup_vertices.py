import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau_backup.drawio')
root = tree.getroot()

print("=== ALL VERTICES IN ctrlau_backup.drawio ===")
for c in root.iter('mxCell'):
    if c.get('vertex') == '1':
        cid = c.get('id')
        val = c.get('value', '').replace('\n', ' ')
        geom = c.find('mxGeometry')
        coords = f"x={geom.get('x')}, y={geom.get('y')}, w={geom.get('width')}, h={geom.get('height')}" if geom is not None else ""
        style = c.get('style', '')
        shape = [s for s in style.split(';') if 'shape=' in s or s in ['rounded=1', 'ellipse', 'text']]
        print(f"[{cid}] ({coords}): val='{val}' | style_hint={shape}")
