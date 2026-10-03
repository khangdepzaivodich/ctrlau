import xml.etree.ElementTree as ET
import math

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

def create_matrix(root, base_id, x, y, rows, cols, size=15, colors=None):
    for r in range(rows):
        for c in range(cols):
            cell_id = f"{base_id}_{r}_{c}"
            fill = colors[r][c] if colors else "#ffffff"
            style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#999999;strokeWidth=1;"
            create_cell(root, cell_id, "", style, x + c*size, y + r*size, size, size)
    create_cell(root, f"{base_id}_box", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#333333;strokeWidth=2;", x, y, cols*size, rows*size)
    return f"{base_id}_box"

def create_tensor_stack(root, base_id, x, y, w, h, num=3, offset=10, fill="#e3f2fd", stroke="#1565c0"):
    for i in range(num):
        create_cell(root, f"{base_id}_{i}", "", f"shape=parallelogram;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;", x + i*offset, y - i*offset, w, h)
    return f"{base_id}_{num-1}"

def create_visual_graph(root, base_id, x, y, num_nodes=5, radius=40, node_size=20):
    nodes = []
    cx, cy = x + radius, y + radius
    for i in range(num_nodes):
        angle = i * (2 * math.pi / num_nodes)
        nx = cx + radius * math.cos(angle) - node_size/2
        ny = cy + radius * math.sin(angle) - node_size/2
        node_id = f"{base_id}_n{i}"
        create_cell(root, node_id, "", "shape=ellipse;fillColor=#ffcdd2;strokeColor=#c62828;strokeWidth=2;", nx, ny, node_size, node_size)
        nodes.append(node_id)
    edge_idx = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            create_edge(root, f"{base_id}_e{edge_idx}", nodes[i], nodes[j], "edgeStyle=none;html=1;strokeWidth=2;strokeColor=#424242;endArrow=none;")
            edge_idx += 1
    create_cell(root, f"{base_id}_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;", x, y, radius*2, radius*2)
    return f"{base_id}_box"

def generate_drawio():
    mxfile = ET.Element("mxfile", version="14.6.11")
    diagram = ET.SubElement(mxfile, "diagram", id="ctrlau_visual", name="CtrlAU Visual Master")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="2000", dy="2000", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="3500", pageHeight="2000", math="1", shadow="0")
    root = ET.SubElement(mxGraphModel, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    s_bg = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=#cfd8dc;strokeWidth=2;dashed=1;verticalAlign=top;align=left;spacingLeft=15;spacingTop=15;fontSize=22;fontStyle=1;"
    s_op = "shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffffff;strokeColor=#000000;strokeWidth=2;fontSize=20;fontStyle=1;"
    s_text = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=18;fontStyle=1;"
    s_arrow = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=3;strokeColor=#263238;endArrow=block;endFill=1;"

    # Colors for matrices
    c_rand = [["#ddd","#fff","#ccc","#eee"], ["#fff","#aaa","#ddd","#ccc"], ["#eee","#ccc","#bbb","#fff"], ["#ccc","#fff","#eee","#aaa"]]
    c_inv = [["#000","#fff","#ddd","#000"], ["#ddd","#000","#fff","#000"], ["#fff","#ddd","#000","#fff"], ["#000","#000","#ddd","#fff"]]
    
    # ================= MAIN FLOW =================
    create_cell(root, "bg_main", "MAIN MACRO FLOW", s_bg, 50, 50, 2800, 250)
    
    # Image icon
    create_cell(root, "m_img", "Input Image", "shape=image;image=img/lib/clip_art/people/Worker_Man_128x128.png;verticalLabelPosition=bottom;verticalAlign=top;", 100, 120, 80, 80)
    # ResNet Tensor Stack
    m_res = create_tensor_stack(root, "m_res", 300, 150, 60, 80, 4, 15)
    create_cell(root, "lbl_res", "ResNet $Z$", s_text, 300, 240, 100, 30)
    # Base Features Matrix
    m_base = create_matrix(root, "m_basev", 550, 130, 4, 2, 20)
    create_cell(root, "lbl_base", "Base Features $V$", s_text, 530, 240, 150, 30)
    # Nulled Features Matrix
    m_null = create_matrix(root, "m_nullv", 800, 130, 4, 2, 20)
    create_cell(root, "lbl_null", "Nulled Features $\hat{V}$", s_text, 780, 240, 150, 30)
    # Graph Visual
    m_graph = create_visual_graph(root, "m_graphv", 1050, 110, 5, 40, 15)
    create_cell(root, "lbl_graphv", "Evolved Graph", s_text, 1050, 240, 150, 30)

    create_edge(root, "em1", "m_img", "m_res_0", s_arrow)
    create_edge(root, "em2", m_res, m_base, s_arrow)
    create_edge(root, "em3", m_base, m_null, s_arrow)
    create_edge(root, "em4", m_null, m_graph, s_arrow)

    # ================= MODULE C: DYNAMIC GRAPH (VISUAL DEMONSTRATION) =================
    create_cell(root, "bg_modc", "VISUAL DEMONSTRATION: Dense Dynamic Graph Conv (Idea 2)", s_bg, 50, 400, 1400, 550)
    
    # Input V matrix
    vin = create_matrix(root, "mc_vin", 100, 600, 4, 2, 25)
    create_cell(root, "lbl_vin", "Input Features $\hat{V}$", s_text, 100, 720, 150, 30)

    # WQ, WK visually splitting
    q_mat = create_matrix(root, "mc_q", 350, 500, 4, 4, 20, c_rand)
    create_cell(root, "lbl_q", "Queries $Q$", s_text, 350, 600, 100, 30)
    
    k_mat = create_matrix(root, "mc_k", 350, 700, 4, 4, 20, c_rand)
    create_cell(root, "lbl_k", "Keys $K^T$", s_text, 350, 800, 100, 30)
    
    create_edge(root, "ec1", vin, q_mat, s_arrow, waypoints=[(250, 650), (250, 540)])
    create_edge(root, "ec2", vin, k_mat, s_arrow, waypoints=[(250, 650), (250, 740)])

    # Matmul Op
    op_mul1 = create_cell(root, "mc_mul1", "$\otimes$", s_op, 550, 630, 60, 60)
    create_edge(root, "ec3", q_mat, op_mul1, s_arrow, waypoints=[(480, 540), (480, 660)])
    create_edge(root, "ec4", k_mat, op_mul1, s_arrow, waypoints=[(480, 740), (480, 660)])

    # Dynamic Matrix
    gdyn_mat = create_matrix(root, "mc_gdyn", 700, 610, 4, 4, 25, c_rand)
    create_cell(root, "lbl_gdyn", "Dynamic Graph\n$\mathcal{G}_{dyn}$", s_text, 700, 730, 100, 40)
    create_edge(root, "ec5", op_mul1, gdyn_mat, s_arrow)

    # Invariant Matrix
    ginv_mat = create_matrix(root, "mc_ginv", 700, 400, 4, 4, 25, c_inv)
    create_cell(root, "lbl_ginv", "Invariant DAG\n$\mathcal{G}_{inv}$", s_text, 700, 520, 100, 40)

    # Element wise multiply
    op_mul2 = create_cell(root, "mc_mul2", "$\odot$", s_op, 900, 520, 60, 60)
    create_edge(root, "ec6", gdyn_mat, op_mul2, s_arrow, waypoints=[(850, 660), (850, 550)])
    create_edge(root, "ec7", ginv_mat, op_mul2, s_arrow, waypoints=[(850, 450), (850, 550)])

    # Final Graph network
    gfinal = create_visual_graph(root, "mc_gfinal", 1050, 480, 6, 60, 20)
    create_cell(root, "lbl_gfinal", "Evolved Network\nGraph Nodes", s_text, 1050, 620, 150, 40)
    create_edge(root, "ec8", op_mul2, gfinal, s_arrow)

    # ================= MODULE B: SEMANTIC NULLING (VISUAL DEMONSTRATION) =================
    create_cell(root, "bg_modb", "VISUAL DEMONSTRATION: Semantic Subspace Nulling", s_bg, 1500, 400, 900, 550)
    
    # Vector V
    create_cell(root, "mb_v", "", "endArrow=block;html=1;strokeWidth=6;strokeColor=#1565c0;endFill=1;", 1600, 700, 150, -150)
    create_cell(root, "lbl_mbv", "Visual Feature Vector $V$", s_text, 1750, 520, 200, 30)
    
    # Vector T ortho
    create_cell(root, "mb_t", "", "endArrow=block;html=1;strokeWidth=6;strokeColor=#ef6c00;endFill=1;", 1600, 700, 250, 0)
    create_cell(root, "lbl_mbt", "Orthogonal Text Vector $T_{ortho}$", s_text, 1850, 720, 250, 30)
    
    # Projection line
    create_cell(root, "mb_proj", "", "endArrow=none;dashed=1;html=1;strokeWidth=3;strokeColor=#999999;", 1750, 550, 0, 150)
    create_cell(root, "lbl_proj", "Projection", s_text, 1760, 620, 100, 30)
    
    # Erased subspace
    create_cell(root, "mb_erase", "Erased Subspace", "shape=flexArrow;endArrow=classic;html=1;fillColor=#ffcdd2;strokeColor=#c62828;", 1600, 720, 150, 0)
    
    # Remaining Vector V_hat
    create_cell(root, "mb_vhat", "", "endArrow=block;html=1;strokeWidth=6;strokeColor=#2e7d32;endFill=1;", 1750, 700, 0, -150)
    create_cell(root, "lbl_vhat", "Remaining Independent\nFeatures $\hat{V}$", s_text, 1600, 500, 200, 50)

    # ================= MODULE D: CYCLE CONSISTENCY (VISUAL DEMONSTRATION) =================
    create_cell(root, "bg_modd", "VISUAL DEMONSTRATION: Bidirectional Cycle Consistency", s_bg, 50, 1000, 1400, 500)
    
    # P_Exp Vector
    pexp = create_matrix(root, "md_pexp", 150, 1200, 1, 6, 40)
    create_cell(root, "lbl_pexp", "Emotion Probabilities $P_{Exp}$", s_text, 150, 1280, 250, 30)
    
    # Matmul
    op_mul3 = create_cell(root, "md_mul3", "$\otimes$", s_op, 450, 1200, 60, 60)
    create_edge(root, "ed1", pexp, op_mul3, s_arrow)
    
    # M_AE Prior Matrix (Visual grid)
    mae_colors = [["#000","#fff","#ddd","#000","#fff","#ddd","#000","#fff"], 
                  ["#ddd","#000","#fff","#000","#ddd","#000","#fff","#000"], 
                  ["#fff","#ddd","#000","#fff","#fff","#ddd","#000","#fff"], 
                  ["#000","#fff","#ddd","#000","#000","#fff","#ddd","#000"],
                  ["#000","#fff","#ddd","#000","#fff","#ddd","#000","#fff"],
                  ["#ddd","#000","#fff","#000","#ddd","#000","#fff","#000"]]
    mae = create_matrix(root, "md_mae", 600, 1150, 6, 8, 25, mae_colors)
    create_cell(root, "lbl_mae", "Multi-view Prior Matrix $M_{AE}$", s_text, 600, 1330, 250, 30)
    create_edge(root, "ed2", op_mul3, mae, s_arrow)
    
    # P_Pseudo Vector
    ppseudo = create_matrix(root, "md_ppseudo", 950, 1200, 1, 8, 40)
    create_cell(root, "lbl_ppseudo", "Expected AUs $P_{pseudo}$", s_text, 950, 1280, 250, 30)
    create_edge(root, "ed3", mae, ppseudo, s_arrow)
    
    # Minus operator
    op_sub = create_cell(root, "md_sub", "$\ominus$", s_op, 1250, 1200, 60, 60)
    create_edge(root, "ed4", ppseudo, op_sub, s_arrow)
    
    # Real P_AU vector coming in
    pau = create_matrix(root, "md_pau", 1200, 1050, 1, 8, 30)
    create_cell(root, "lbl_pau", "Predicted $P_{AU}$", s_text, 1200, 1020, 200, 30)
    create_edge(root, "ed5", pau, op_sub, s_arrow)

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ", level=0)
    tree.write("ctrlau_visual_demonstration.drawio", encoding="utf-8", xml_declaration=True)
    print("Visual Drawio generated!")

if __name__ == "__main__":
    generate_drawio()
