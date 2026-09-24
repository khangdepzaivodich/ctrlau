import xml.etree.ElementTree as ET

tree = ET.parse('ctrlau.drawio')
root = tree.getroot()

for c in root.iter('mxCell'):
    if c.get('id') == 'GjfUm5Ax_Uw3Q_51WUAJ-1':
        geom = c.find('mxGeometry')
        style = c.get('style', '')
        print("Image cell found:")
        print("id:", c.get('id'))
        print("parent:", c.get('parent'))
        print("geom: x=", geom.get('x'), "y=", geom.get('y'), "w=", geom.get('width'), "h=", geom.get('height'))
        print("style start:", style[:80])
        print("style length:", len(style))
