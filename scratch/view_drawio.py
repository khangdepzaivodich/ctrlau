import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def render_drawio(xml_file, out_png):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    fig, ax = plt.subplots(figsize=(24, 14))
    
    # We want top-left to be (0,0), so invert y-axis
    ax.invert_yaxis()

    # Find all mxGeometry elements
    for cell in root.iter('mxCell'):
        val = cell.get('value', '')
        style = cell.get('style', '')
        geom = cell.find('mxGeometry')
        
        if geom is not None and geom.get('width'):
            x = float(geom.get('x', 0))
            y = float(geom.get('y', 0))
            w = float(geom.get('width'))
            h = float(geom.get('height'))
            
            # Draw box
            edgecolor = 'blue'
            if 'text;' in style:
                edgecolor = 'red'
                
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=edgecolor, facecolor='none', alpha=0.5)
            ax.add_patch(rect)
            
            # Label
            if val:
                # simplify val
                val = val.replace('&lt;font', '<f').split(';')[0][:15] 
                ax.text(x + w/2, y + h/2, val, ha='center', va='center', fontsize=6, color='black')
                
        elif geom is not None and cell.get('edge') == '1':
            source = cell.get('source')
            target = cell.get('target')
            
            pts_elem = geom.find('Array')
            pts = []
            if pts_elem is not None:
                for pt in pts_elem.findall('mxPoint'):
                    pts.append((float(pt.get('x')), float(pt.get('y'))))
            
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color='green', linewidth=1, alpha=0.5)

    ax.set_xlim(0, 2700)
    ax.set_ylim(1600, 0)
    plt.savefig(out_png, dpi=200)
    print(f"Saved to {out_png}")

if __name__ == "__main__":
    render_drawio(r"C:\Users\khang\OneDrive\Desktop\ctrlau\ctrlau_architecture.drawio", r"C:\Users\khang\OneDrive\Desktop\ctrlau\scratch\layout.png")
