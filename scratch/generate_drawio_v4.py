import xml.etree.ElementTree as ET
import math

def create_cell(root, id, value, style, x, y, width, height, parent="1"):
    cell = ET.SubElement(root, "mxCell", id=id, value=value, style=style, vertex="1", parent=parent)
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(width), height=str(height), **{"as": "geometry"})
    return cell

def create_edge(root, id, source, target, style, waypoints=None, parent="1"):
    cell = ET.SubElement(root, "mxCell", id=id, style=style, edge="1", parent=parent, source=source, target=target)
    geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
    if waypoints:
        arr = ET.SubElement(geo, "Array", **{"as": "points"})
        for wx, wy in waypoints:
            ET.SubElement(arr, "mxPoint", x=str(wx), y=str(wy))
    return cell

def create_matrix(root, base_id, x, y, rows, cols, size=10, colors=None):
    # Creates a visual matrix grid
    for r in range(rows):
        for c in range(cols):
            cell_id = f"{base_id}_{r}_{c}"
            fill = colors[r][c] if colors else "#ffffff"
            style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#cccccc;"
            create_cell(root, cell_id, "", style, x + c*size, y + r*size, size, size)
    # Add a bounding box
    create_cell(root, f"{base_id}_box", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;strokeWidth=2;", x, y, cols*size, rows*size)

def create_visual_graph(root, base_id, x, y, num_nodes=4, radius=30, node_size=20):
    # Creates a circular graph with nodes and fully connected edges
    nodes = []
    cx, cy = x + radius, y + radius
    for i in range(num_nodes):
        angle = i * (2 * math.pi / num_nodes)
        nx = cx + radius * math.cos(angle) - node_size/2
        ny = cy + radius * math.sin(angle) - node_size/2
        node_id = f"{base_id}_n{i}"
        create_cell(root, node_id, "", "shape=ellipse;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=2;", nx, ny, node_size, node_size)
        nodes.append(node_id)
    
    # Connect them
    edge_idx = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edge_id = f"{base_id}_e{edge_idx}"
            create_edge(root, edge_id, nodes[i], nodes[j], "edgeStyle=none;html=1;strokeWidth=1;strokeColor=#999999;endArrow=none;")
            edge_idx += 1

def generate_drawio():
    mxfile = ET.Element("mxfile", version="14.6.11")
    diagram = ET.SubElement(mxfile, "diagram", id="ctrlau_ultimate", name="CtrlAU Ultimate Architecture")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="2000", dy="1500", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1600", pageHeight="1200", math="1", shadow="1")
    root = ET.SubElement(mxGraphModel, "root")
    
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    # STYLES
    s_tensor = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.1;darkOpacity2=0.1;fillColor=#e3f2fd;strokeColor=#1565c0;strokeWidth=2;size=15;"
    s_module = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#424242;strokeWidth=2;shadow=1;fontStyle=1;"
    s_loss = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#c62828;strokeWidth=2;dashed=1;fontColor=#b71c1c;fontStyle=1;"
    s_text = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;fontStyle=1;"
    s_arrow = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#333333;endArrow=block;endFill=1;"

    # -------------------------------------------------------------
    # 1. FEATURE EXTRACTION
    # -------------------------------------------------------------
    create_cell(root, "img", "Input Image", "shape=parallelogram;fillColor=#eceff1;strokeColor=#546e7a;strokeWidth=2;", 50, 200, 120, 80)
    
    create_cell(root, "resnet_stack1", "", "rounded=1;fillColor=#f5f5f5;strokeColor=#999999;strokeWidth=2;", 220, 220, 60, 100)
    create_cell(root, "resnet_stack2", "", "rounded=1;fillColor=#e0e0e0;strokeColor=#666666;strokeWidth=2;", 230, 210, 60, 100)
    create_cell(root, "resnet", "ResNet-50", "rounded=1;fillColor=#fff3e0;strokeColor=#ef6c00;strokeWidth=2;", 240, 200, 80, 100)
    
    create_cell(root, "feat_z", "$$Z$$", s_tensor, 370, 210, 80, 80)
    
    create_cell(root, "au_head", "AU Head", s_module, 500, 150, 100, 60)
    create_cell(root, "exp_head", "Emotion Head", s_module, 500, 280, 100, 60)
    
    create_cell(root, "v_au", "$$V_{AU}$$", s_tensor, 650, 140, 60, 80)
    create_cell(root, "v_exp", "$$V_{Exp}$$", s_tensor, 650, 270, 60, 80)

    create_edge(root, "e1", "img", "resnet_stack1", s_arrow)
    create_edge(root, "e2", "resnet", "feat_z", s_arrow)
    create_edge(root, "e3", "feat_z", "au_head", s_arrow)
    create_edge(root, "e4", "feat_z", "exp_head", s_arrow)
    create_edge(root, "e5", "au_head", "v_au", s_arrow)
    create_edge(root, "e6", "exp_head", "v_exp", s_arrow)

    # -------------------------------------------------------------
    # 2. SYMBOLIC PRIORS & MASKS
    # -------------------------------------------------------------
    create_cell(root, "box_priors", "Knowledge Priors & Masks", "rounded=1;fillColor=none;strokeColor=#bdbdbd;dashed=1;align=left;verticalAlign=top;spacingLeft=10;", 480, 420, 360, 220)
    
    # Generate Visual Matrices
    m_colors = [["#000", "#fff", "#ddd", "#000"], ["#ddd", "#000", "#fff", "#000"], ["#fff", "#ddd", "#000", "#fff"], ["#000", "#000", "#ddd", "#fff"]]
    create_matrix(root, "mat_imp", 520, 480, 4, 4, 15, m_colors)
    create_cell(root, "lbl_imp", "Importance Mask\n$$M_{imp}$$", s_text, 500, 550, 100, 40)
    
    create_matrix(root, "mat_pol", 670, 480, 4, 4, 15, m_colors)
    create_cell(root, "lbl_pol", "Polarity Mask\n$$M_{pol}$$", s_text, 650, 550, 100, 40)
    
    # Text CLIP
    create_cell(root, "clip", "CLIP Text\n$$T$$", s_tensor, 580, 680, 80, 80)
    create_cell(root, "gs_ortho", "Gram-Schmidt\n(Idea 1.1)", s_module, 720, 690, 100, 60)
    create_edge(root, "e_clip_gs", "clip", "gs_ortho", s_arrow)

    # -------------------------------------------------------------
    # 3. INTERVENTIONS (THE HIGHLIGHT)
    # -------------------------------------------------------------
    create_cell(root, "box_interv", "Neuro-Symbolic Causal Interventions", "rounded=1;fillColor=#f3e5f5;strokeColor=#ab47bc;strokeWidth=2;align=left;verticalAlign=top;spacingLeft=10;", 850, 120, 600, 400)
    
    # Idea 1: Subspace Nulling (Vector Projection Graphic)
    create_cell(root, "lbl_idea1", "Semantic Subspace Nulling (Idea 1)", s_text, 870, 160, 250, 30)
    # Draw vector V
    create_cell(root, "vec_v", "", "endArrow=block;html=1;strokeWidth=3;strokeColor=#1565c0;", 880, 280, 80, -80)
    create_cell(root, "lbl_vec_v", "$$V$$", s_text, 960, 190, 30, 30)
    # Draw vector T (Orthogonal)
    create_cell(root, "vec_t", "", "endArrow=block;html=1;strokeWidth=3;strokeColor=#ef6c00;", 880, 280, 100, 0)
    create_cell(root, "lbl_vec_t", "$$T_{ortho}$$", s_text, 980, 280, 50, 30)
    # Projection line (dashed)
    create_cell(root, "vec_proj", "", "endArrow=none;dashed=1;html=1;strokeWidth=2;strokeColor=#999999;", 960, 200, 0, 80)
    # Erased Subspace
    create_cell(root, "vec_erased", "Erased", "shape=flexArrow;endArrow=classic;html=1;fillColor=#ffcccc;strokeColor=#cc0000;", 880, 290, 80, 0)
    
    # Connection from Ortho to Nulling
    create_edge(root, "e_gs_null", "gs_ortho", "lbl_idea1", s_arrow, waypoints=[(770, 175)])
    
    # Counterfactual Split
    create_cell(root, "cf_split", "Factual vs Counterfactual\nIntervention Split", "shape=hexagon;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=2;", 1100, 180, 150, 60)
    create_edge(root, "e_null_split", "lbl_vec_v", "cf_split", s_arrow)
    
    create_cell(root, "cf_imp", "Important\n(Intervened)", "shape=step;fillColor=#ffcdd2;strokeColor=#c62828;", 1300, 150, 120, 50)
    create_cell(root, "cf_unimp", "Unimportant\n(Consistent)", "shape=step;fillColor=#c8e6c9;strokeColor=#2e7d32;", 1300, 230, 120, 50)
    create_edge(root, "e_split_imp", "cf_split", "cf_imp", s_arrow)
    create_edge(root, "e_split_unimp", "cf_split", "cf_unimp", s_arrow)

    # Idea 2: Dynamic Causal Graph
    create_cell(root, "lbl_idea2", "Dense Dynamic Causal Graph (Idea 2)", s_text, 870, 350, 250, 30)
    # Matrix G_dyn
    create_matrix(root, "mat_gdyn", 880, 400, 4, 4, 15, m_colors)
    create_cell(root, "lbl_gdyn", "$$\mathcal{G}_{dyn}$$", s_text, 880, 470, 60, 30)
    # Matrix G_inv
    create_matrix(root, "mat_ginv", 1000, 400, 4, 4, 15, m_colors)
    create_cell(root, "lbl_ginv", "$$\mathcal{G}_{inv}$$", s_text, 1000, 470, 60, 30)
    
    create_cell(root, "op_mul2", "$$\otimes$$", "shape=ellipse;fillColor=#fff;strokeWidth=2;", 960, 415, 30, 30)
    
    # Visual Network Graph
    create_visual_graph(root, "graph_vis", 1120, 380, num_nodes=5, radius=40, node_size=20)
    create_cell(root, "lbl_graph", "Evolved Graph\nNodes", s_text, 1100, 480, 120, 30)
    
    create_edge(root, "e_g_to_g", "mat_ginv_box", "graph_vis_n0", s_arrow)

    # DAG Regularization
    create_cell(root, "dag_loss", "Global DAG Reg", s_loss, 1000, 500, 120, 40)
    create_edge(root, "e_dag_inv", "dag_loss", "lbl_ginv", "edgeStyle=none;html=1;strokeWidth=2;strokeColor=#c62828;dashed=1;endArrow=block;")

    # -------------------------------------------------------------
    # 4. ABDUCTIVE INFERENCE & CYCLE CONSISTENCY
    # -------------------------------------------------------------
    create_cell(root, "box_abd", "Test-Time Inference & Consistency", "rounded=1;fillColor=#e8eaf6;strokeColor=#3f51b5;strokeWidth=2;align=left;verticalAlign=top;spacingLeft=10;", 1100, 580, 450, 320)
    
    create_cell(root, "logits", "Logits\n$$\hat{Y}$$", s_tensor, 1150, 650, 60, 80)
    create_cell(root, "optimizer", "Energy Minimizer\n$$\\nabla \mathcal{E}$$", "shape=ellipse;fillColor=#ffecb3;strokeColor=#ffb300;strokeWidth=2;", 1320, 650, 120, 80)
    
    create_cell(root, "facs_loss", "FACS Energy\n$$\mathcal{E}_{FACS}$$", s_loss, 1200, 800, 100, 50)
    
    # Visual Cycle Matrix M_AE
    create_cell(root, "lbl_mae", "Prior $$M_{AE}$$ Projection\n(Idea 4)", s_text, 1370, 780, 150, 40)
    create_matrix(root, "mat_mae", 1400, 820, 3, 5, 12, None)
    create_cell(root, "cycle_loss", "Cycle Consistency\n$$\mathcal{E}_{Cycle}$$", s_loss, 1350, 880, 140, 50)
    
    create_edge(root, "e_log_opt", "logits", "optimizer", s_arrow)
    # Loop back
    create_edge(root, "e_opt_log", "optimizer", "logits", "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#ffb300;endArrow=block;", waypoints=[(1380, 600), (1180, 600)])
    
    create_edge(root, "e_facs_opt", "facs_loss", "optimizer", "edgeStyle=orthogonalEdgeStyle;rounded=1;dashed=1;strokeColor=#c62828;endArrow=block;")
    create_edge(root, "e_cyc_opt", "cycle_loss", "optimizer", "edgeStyle=orthogonalEdgeStyle;rounded=1;dashed=1;strokeColor=#c62828;endArrow=block;")
    
    # Save to file
    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ", level=0)
    tree.write("ctrlau_ultimate_figure.drawio", encoding="utf-8", xml_declaration=True)
    print("Ultimate Drawio generated!")

if __name__ == "__main__":
    generate_drawio()
