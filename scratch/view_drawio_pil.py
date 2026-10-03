import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

def render_drawio(xml_file, out_png):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    width, height = 3000, 2000
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    for cell in root.iter('mxCell'):
        val = cell.get('value', '')
        style = cell.get('style', '')
        geom = cell.find('mxGeometry')
        
        if geom is not None and geom.get('width'):
            x = float(geom.get('x', 0))
            y = float(geom.get('y', 0))
            w = float(geom.get('width'))
            h = float(geom.get('height'))
            
            color = "blue"
            if "text;" in style:
                color = "red"
            
            draw.rectangle([x, y, x+w, y+h], outline=color, width=1)
            
        elif geom is not None and cell.get('edge') == '1':
            pts_elem = geom.find('Array')
            if pts_elem is not None:
                pts = []
                for pt in pts_elem.findall('mxPoint'):
                    pts.append((float(pt.get('x')), float(pt.get('y'))))
                
                if len(pts) > 1:
                    draw.line(pts, fill="green", width=2)

    img.save(out_png)
    print(f"Saved to {out_png}")

if __name__ == "__main__":
    render_drawio(r"C:\Users\khang\OneDrive\Desktop\ctrlau\ctrlau_architecture.drawio", r"C:\Users\khang\OneDrive\Desktop\ctrlau\scratch\layout.png")
