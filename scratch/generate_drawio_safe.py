import xml.etree.ElementTree as ET

def create_cell(root, id, value, style, x, y, w, h, parent="1"):
    cell = ET.SubElement(root, "mxCell", id=id, value=value, style=style, vertex="1", parent=parent)
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
    return cell

def create_edge(root, id, src, tgt, style, waypoints=None, parent="1"):
    cell = ET.SubElement(root, "mxCell", id=id, style=style, edge="1", parent=parent, source=src, target=tgt)
    geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
    if waypoints:
        arr = ET.SubElement(geo, "Array", **{"as": "points"})
        for wx, wy in waypoints:
            ET.SubElement(arr, "mxPoint", x=str(wx), y=str(wy))
    return cell

def create_matrix(root, base_id, x, y, rows, cols, size=15):
    for r in range(rows):
        for c in range(cols):
            cell_id = f"{base_id}_{r}_{c}"
            fill = "#ffffff" if (r+c)%2==0 else "#e0e0e0" # simple checkerboard
            style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#999999;"
            create_cell(root, cell_id, "", style, x + c*size, y + r*size, size, size)
    create_cell(root, f"{base_id}_box", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#333333;strokeWidth=2;", x, y, cols*size, rows*size)
    return f"{base_id}_box"

def generate_drawio():
    mxfile = ET.Element("mxfile", version="14.6.11")
    diagram = ET.SubElement(mxfile, "diagram", id="ctrlau_fixed", name="CtrlAU Architecture")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="1000", dy="1000", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="2000", pageHeight="1500", math="0", shadow="0")
    root = ET.SubElement(mxGraphModel, "root")
    
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    # Safe styles without complex formatting
    s_box = "rounded=1;whiteSpace=wrap;html=1;fillColor=#e3f2fd;strokeColor=#1565c0;strokeWidth=2;fontSize=14;fontStyle=1;"
    s_tensor = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.1;darkOpacity2=0.1;fillColor=#e8f5e9;strokeColor=#2e7d32;strokeWidth=2;fontSize=14;"
    s_arrow = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#333333;endArrow=block;endFill=1;"

    # 1. Main Pipeline
    create_cell(root, "bg1", "1. Feature Extraction", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#cccccc;align=left;verticalAlign=top;", 50, 50, 450, 200)
    create_cell(root, "img", "Input Image", s_box, 80, 100, 100, 60)
    create_cell(root, "backbone", "ResNet-50", s_box, 220, 100, 100, 60)
    create_cell(root, "feat", "Feature Z", s_tensor, 360, 90, 80, 80)
    create_edge(root, "e1", "img", "backbone", s_arrow)
    create_edge(root, "e2", "backbone", "feat", s_arrow)

    # 2. Idea 1: Semantic Nulling Demonstration
    create_cell(root, "bg2", "2. Semantic Subspace Nulling (Idea 1)", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#cccccc;align=left;verticalAlign=top;", 50, 300, 450, 250)
    
    create_cell(root, "vec_v", "Visual V", "endArrow=block;html=1;strokeWidth=4;strokeColor=#1565c0;endFill=1;fontSize=14;", 100, 480, 150, -100)
    create_cell(root, "vec_t", "Text T", "endArrow=block;html=1;strokeWidth=4;strokeColor=#ef6c00;endFill=1;fontSize=14;", 100, 480, 200, 0)
    create_cell(root, "proj", "Projection", "endArrow=none;dashed=1;html=1;strokeWidth=2;strokeColor=#999999;fontSize=12;", 250, 380, 50, 100)
    create_cell(root, "erased", "Erased", "shape=flexArrow;endArrow=classic;html=1;fillColor=#ffcdd2;strokeColor=#c62828;", 100, 490, 150, 0)

    # 3. Idea 2: Dynamic Graph Matrices Demonstration
    create_cell(root, "bg3", "3. Dynamic Graph Convolution (Idea 2)", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#cccccc;align=left;verticalAlign=top;", 550, 50, 600, 200)
    
    create_matrix(root, "qmat", 600, 100, 4, 4, 15)
    create_cell(root, "lbl_q", "Q Matrix", "text;html=1;strokeColor=none;fillColor=none;align=center;", 600, 170, 60, 30)
    
    create_matrix(root, "kmat", 700, 100, 4, 4, 15)
    create_cell(root, "lbl_k", "K Matrix", "text;html=1;strokeColor=none;fillColor=none;align=center;", 700, 170, 60, 30)
    
    create_cell(root, "mul1", "x", "shape=ellipse;fillColor=#fff;strokeColor=#000;", 670, 120, 20, 20)
    
    create_matrix(root, "dynmat", 820, 100, 4, 4, 15)
    create_cell(root, "lbl_dyn", "Dynamic Graph", "text;html=1;strokeColor=none;fillColor=none;align=center;", 800, 170, 100, 30)
    create_edge(root, "e_dyn", "kmat_box", "dynmat_box", s_arrow)

    create_matrix(root, "invmat", 950, 100, 4, 4, 15)
    create_cell(root, "lbl_inv", "Invariant DAG", "text;html=1;strokeColor=none;fillColor=none;align=center;", 930, 170, 100, 30)
    
    create_cell(root, "mul2", "x", "shape=ellipse;fillColor=#fff;strokeColor=#000;", 900, 120, 20, 20)

    # 4. Idea 4: Cycle Consistency Matrices
    create_cell(root, "bg4", "4. Cycle Consistency (Idea 4)", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#cccccc;align=left;verticalAlign=top;", 550, 300, 600, 250)
    
    create_matrix(root, "pexp", 600, 400, 1, 6, 25)
    create_cell(root, "lbl_pexp", "Emotion Probs", "text;html=1;strokeColor=none;fillColor=none;align=center;", 600, 430, 150, 30)
    
    create_matrix(root, "mae", 800, 370, 6, 8, 20)
    create_cell(root, "lbl_mae", "Prior Matrix M_AE", "text;html=1;strokeColor=none;fillColor=none;align=center;", 800, 500, 160, 30)
    
    create_cell(root, "mul3", "x", "shape=ellipse;fillColor=#fff;strokeColor=#000;", 760, 405, 20, 20)
    
    create_matrix(root, "ppseudo", 1000, 400, 1, 8, 25)
    create_cell(root, "lbl_ppseudo", "Expected AU Probs", "text;html=1;strokeColor=none;fillColor=none;align=center;", 1000, 430, 200, 30)
    create_edge(root, "e_cyc", "mae_box", "ppseudo_box", s_arrow)

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ", level=0)
    tree.write("ctrlau_architecture.drawio", encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    generate_drawio()
