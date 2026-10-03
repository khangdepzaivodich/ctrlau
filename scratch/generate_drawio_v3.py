import xml.etree.ElementTree as ET

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

def generate_drawio():
    mxfile = ET.Element("mxfile", version="14.6.11")
    diagram = ET.SubElement(mxfile, "diagram", id="ctrlau_cvpr", name="CtrlAU CVPR Style")
    # Enable math typesetting (math="1") and shadows (shadow="1")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="1500", dy="1000", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1169", pageHeight="827", math="1", shadow="1")
    root = ET.SubElement(mxGraphModel, "root")
    
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    # ================= STYLES =================
    # Modern paper styles: thick borders, clean pastel fills, mathematical fonts
    style_tensor = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;fillColor=#e3f2fd;strokeColor=#1565c0;strokeWidth=2;fontColor=#000000;size=15;"
    style_tensor_green = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;fillColor=#e8f5e9;strokeColor=#2e7d32;strokeWidth=2;fontColor=#000000;size=15;"
    style_tensor_orange = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;fillColor=#fff3e0;strokeColor=#ef6c00;strokeWidth=2;fontColor=#000000;size=15;"
    
    style_module = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#424242;strokeWidth=2;fontStyle=1;shadow=1;"
    style_loss = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#c62828;strokeWidth=2;dashed=1;fontColor=#b71c1c;fontStyle=1;"
    
    style_op = "shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffffff;strokeColor=#000000;strokeWidth=2;fontSize=24;fontStyle=1;"
    
    style_bg = "rounded=1;whiteSpace=wrap;html=1;fillColor=#fafafa;strokeColor=#bdbdbd;dashed=1;strokeWidth=2;verticalAlign=top;align=left;spacingLeft=10;spacingTop=10;fontSize=16;fontStyle=1;"

    style_edge = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#424242;endArrow=block;endFill=1;"
    style_edge_dash = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#c62828;dashed=1;endArrow=block;endFill=1;"
    style_edge_straight = "edgeStyle=none;html=1;strokeWidth=2;strokeColor=#424242;endArrow=block;endFill=1;"

    # ================= MACRO ARCHITECTURE (TOP) =================
    create_cell(root, "macro_bg", "Macro Pipeline: Joint Feature Learning & Graph Reasoning", style_bg, 20, 20, 1100, 300)
    
    # Inputs
    create_cell(root, "img_txt", "$$X \in \mathbb{R}^{3 \\times 224 \\times 224}$$", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=14;", 40, 80, 150, 30)
    create_cell(root, "img", "Input Image", "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;fillColor=#eceff1;strokeColor=#455a64;strokeWidth=2;", 40, 120, 150, 60)
    
    # Backbone
    create_cell(root, "resnet", "ResNet-50\nEncoder", style_module, 240, 120, 100, 60)
    create_cell(root, "feat_map", "$$Z \in \mathbb{R}^{C \\times H \\times W}$$", style_tensor, 390, 110, 120, 80)
    
    # Extractors
    create_cell(root, "extractors", "Feature\nExtractors\n$$(g_{AU}, g_{Exp})$$", style_module, 560, 120, 100, 60)
    create_cell(root, "base_v", "$$V \in \mathbb{R}^{N \\times D}$$", style_tensor_green, 710, 110, 120, 80)
    
    # Graph Module (Zoom target)
    create_cell(root, "graph_module", "Causal\nIntervention\nBlock (See Below)", style_module + "fillColor=#fff9c4;strokeColor=#fbc02d;", 880, 100, 120, 100)
    
    # Outputs
    create_cell(root, "out_probs", "$$\hat{Y} \in \mathbb{R}^{N}$$", style_tensor_orange, 1050, 120, 80, 60)
    
    # Connections Macro
    create_edge(root, "e_m1", "img", "resnet", style_edge_straight)
    create_edge(root, "e_m2", "resnet", "feat_map", style_edge_straight)
    create_edge(root, "e_m3", "feat_map", "extractors", style_edge_straight)
    create_edge(root, "e_m4", "extractors", "base_v", style_edge_straight)
    create_edge(root, "e_m5", "base_v", "graph_module", style_edge_straight)
    create_edge(root, "e_m6", "graph_module", "out_probs", style_edge_straight)

    # ================= MICRO ARCHITECTURE (BOTTOM ZOOM-IN) =================
    create_cell(root, "micro_bg", "Micro Architecture: Neuro-Symbolic Causal Interventions", style_bg, 20, 360, 1100, 420)
    
    # Visual Link from top
    create_cell(root, "v_in", "$$V$$", style_tensor_green, 50, 480, 80, 60)
    
    # --- Idea 1 & 1.1: Semantic Subspace Nulling ---
    create_cell(root, "box_nulling", "Semantic Subspace Nulling (Idea 1 & 1.1)", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#9e9e9e;dashed=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=10;", 40, 420, 460, 320)
    
    create_cell(root, "text_input", "CLIP Text\n$$T_{raw}$$", style_tensor_orange, 180, 450, 80, 60)
    create_cell(root, "gs_ortho", "Gram-Schmidt\nOrthogonalization", style_module, 160, 550, 120, 60)
    create_edge(root, "e_t1", "text_input", "gs_ortho", style_edge_straight)
    
    create_cell(root, "t_ortho", "$$T_{ortho}$$", style_tensor_orange, 180, 650, 80, 60)
    create_edge(root, "e_t2", "gs_ortho", "t_ortho", style_edge_straight)
    
    # Projection operator
    create_cell(root, "op_proj", "$$\otimes$$", style_op, 330, 660, 40, 40)
    create_edge(root, "e_proj1", "v_in", "op_proj", style_edge, waypoints=[(90, 680)])
    create_edge(root, "e_proj2", "t_ortho", "op_proj", style_edge_straight)
    
    # Subtraction operator
    create_cell(root, "op_sub", "$$\ominus$$", style_op, 330, 490, 40, 40)
    create_edge(root, "e_sub1", "v_in", "op_sub", style_edge_straight)
    create_edge(root, "e_sub2", "op_proj", "op_sub", style_edge_straight)
    
    create_cell(root, "v_null", "$$\hat{V}$$", style_tensor_green, 420, 480, 60, 60)
    create_edge(root, "e_sub_out", "op_sub", "v_null", style_edge_straight)

    # --- Idea 2: Dynamic Causal Graph ---
    create_cell(root, "box_graph", "Dense Dynamic Graph Convolution (Idea 2)", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#9e9e9e;dashed=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=10;", 520, 420, 260, 320)
    
    create_cell(root, "g_dyn", "Sample-Adaptive\n$$\mathcal{G}_{dyn} = \sigma(QK^T)$$", style_module, 540, 480, 120, 60)
    create_cell(root, "g_inv", "Global Invariant\n$$\mathcal{G}_{inv}$$", style_module, 540, 580, 120, 60)
    
    create_cell(root, "dag_reg", "$$\mathcal{L}_{DAG}$$", style_loss, 550, 670, 100, 40)
    create_edge(root, "e_dag", "dag_reg", "g_inv", style_edge_dash)
    
    create_cell(root, "op_mul", "$$\otimes$$", style_op, 700, 540, 40, 40)
    create_edge(root, "e_g1", "g_dyn", "op_mul", style_edge, waypoints=[(720, 510)])
    create_edge(root, "e_g2", "g_inv", "op_mul", style_edge, waypoints=[(720, 610)])
    
    create_cell(root, "v_out", "$$V_{out}$$", style_tensor_green, 810, 480, 60, 60)
    create_edge(root, "e_g_out", "v_null", "g_dyn", style_edge_straight)
    # The output of the graph convolution
    create_edge(root, "e_g_mul_out", "op_mul", "v_out", style_edge, waypoints=[(720, 450), (840, 450)])

    # --- Idea 3 & 4: Abductive Inference ---
    create_cell(root, "box_abd", "Test-Time Abductive Inference (Idea 3 & 4)", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#9e9e9e;dashed=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=10;", 800, 420, 300, 320)
    
    create_cell(root, "logits", "Classifier\nLogits", style_module, 890, 480, 80, 60)
    create_edge(root, "e_vout_log", "v_out", "logits", style_edge_straight)
    
    create_cell(root, "abd_opt", "Energy Minimization\n$$\\min_{\hat{Y}} \mathcal{E}$$", style_module + "fillColor=#fff9c4;", 990, 480, 100, 60)
    create_edge(root, "e_log_opt", "logits", "abd_opt", style_edge_straight)
    
    create_cell(root, "e_facs", "$$\mathcal{E}_{FACS}$$", style_loss, 850, 600, 80, 40)
    create_cell(root, "e_cycle", "$$\mathcal{E}_{Cycle} (Idea 4)$$", style_loss, 950, 600, 120, 40)
    
    # Optimizer loop (back edge)
    create_edge(root, "e_opt_back", "abd_opt", "logits", style_edge, waypoints=[(1040, 450), (930, 450)])
    
    create_edge(root, "e_facs_opt", "e_facs", "abd_opt", style_edge_dash, waypoints=[(890, 570), (1040, 570)])
    create_edge(root, "e_cyc_opt", "e_cycle", "abd_opt", style_edge_dash, waypoints=[(1010, 570)])

    # Save to file
    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ", level=0)
    tree.write("ctrlau_cvpr_figure.drawio", encoding="utf-8", xml_declaration=True)
    print("CVPR Drawio generated!")

if __name__ == "__main__":
    generate_drawio()
