import xml.etree.ElementTree as ET

def create_block(root, id, value, x, y, width, height, style):
    cell = ET.SubElement(root, "mxCell", id=id, value=value, style=style, vertex="1", parent="1")
    geo = ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(width), height=str(height), **{"as": "geometry"})

def create_edge(root, id, source, target, style, waypoints=None):
    cell = ET.SubElement(root, "mxCell", id=id, style=style, edge="1", parent="1", source=source, target=target)
    geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
    if waypoints:
        arr = ET.SubElement(geo, "Array", **{"as": "points"})
        for wx, wy in waypoints:
            ET.SubElement(arr, "mxPoint", x=str(wx), y=str(wy))

def create_text(root, id, value, x, y, width, height, style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontStyle=1;fontSize=14;"):
    cell = ET.SubElement(root, "mxCell", id=id, value=value, style=style, vertex="1", parent="1")
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(width), height=str(height), **{"as": "geometry"})

def generate_drawio():
    mxfile = ET.Element("mxfile", version="14.6.11")
    diagram = ET.SubElement(mxfile, "diagram", id="ctrlau_diagram_v2", name="CtrlAU Architecture V2")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="1500", dy="1000", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1169", pageHeight="827", math="0", shadow="0")
    root = ET.SubElement(mxGraphModel, "root")
    
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    # STYLES (Paper-like aesthetic: flat, soft colors, sharp or slightly rounded corners)
    style_bg = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=#ced4da;dashed=1;strokeWidth=2;"
    style_blue = "rounded=0;whiteSpace=wrap;html=1;fillColor=#e3f2fd;strokeColor=#1e88e5;fontColor=#000000;strokeWidth=1.5;"
    style_green = "rounded=0;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#43a047;fontColor=#000000;strokeWidth=1.5;"
    style_pink = "rounded=0;whiteSpace=wrap;html=1;fillColor=#fce4ec;strokeColor=#e53935;fontColor=#000000;strokeWidth=1.5;"
    style_orange = "rounded=0;whiteSpace=wrap;html=1;fillColor=#fff3e0;strokeColor=#fb8c00;fontColor=#000000;strokeWidth=1.5;"
    style_purple = "rounded=0;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#8e24aa;fontColor=#000000;strokeWidth=1.5;"
    style_loss = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#d32f2f;fontColor=#b71c1c;dashed=1;strokeWidth=2;"
    style_input = "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;fillColor=#eceff1;strokeColor=#546e7a;strokeWidth=1.5;"
    style_data = "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;fillColor=#e0f7fa;strokeColor=#006064;strokeWidth=1.5;"

    style_edge = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#333333;endArrow=block;endFill=1;"
    style_edge_dashed = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#d32f2f;dashed=1;endArrow=block;endFill=1;"

    # BACKGROUND BOXES
    create_block(root, "bg_p1", "", 20, 150, 240, 500, style_bg)
    create_text(root, "txt_p1", "Phase 1: Feature Extraction", 20, 110, 240, 30)
    
    create_block(root, "bg_p2", "", 280, 150, 480, 500, style_bg)
    create_text(root, "txt_p2", "Phase 2: Relational Graph & Causal Interventions", 280, 110, 480, 30)
    
    create_block(root, "bg_p3", "", 780, 150, 300, 500, style_bg)
    create_text(root, "txt_p3", "Phase 3: Abductive Inference & Outputs", 780, 110, 300, 30)

    # -------------------------------------------------------------
    # PHASE 1: FEATURE EXTRACTION (x = 40 to 240)
    # -------------------------------------------------------------
    create_block(root, "input_img", "Face Image\n(224x224)", 70, 180, 140, 40, style_input)
    create_block(root, "resnet", "ResNet-50\n(Backbone)", 70, 250, 140, 40, style_blue)
    create_block(root, "linear", "Linear Projection\n(2048 -> 512)", 70, 320, 140, 40, style_blue)
    
    create_block(root, "au_head", "AU Head\n(8 Branches)", 40, 390, 90, 50, style_blue)
    create_block(root, "exp_head", "Emotion Head\n(6 Branches)", 150, 390, 90, 50, style_blue)
    
    create_block(root, "au_emb", "Base AU\nEmbeddings", 40, 470, 90, 50, style_data)
    create_block(root, "exp_emb", "Base Emotion\nEmbeddings", 150, 470, 90, 50, style_data)
    
    create_block(root, "hsic_loss", "HSIC Disentangle\nLoss", 40, 560, 90, 40, style_loss)

    create_edge(root, "e1", "input_img", "resnet", style_edge)
    create_edge(root, "e2", "resnet", "linear", style_edge)
    create_edge(root, "e3", "linear", "au_head", style_edge)
    create_edge(root, "e4", "linear", "exp_head", style_edge)
    create_edge(root, "e5", "au_head", "au_emb", style_edge)
    create_edge(root, "e6", "exp_head", "exp_emb", style_edge)
    create_edge(root, "e_hsic", "au_emb", "hsic_loss", style_edge_dashed)

    # -------------------------------------------------------------
    # PHASE 2: CAUSAL GRAPH (x = 300 to 740)
    # -------------------------------------------------------------
    create_block(root, "text_input", "CLIP Text Prompts\n(AU & Emotions)", 300, 180, 140, 40, style_input)
    create_block(root, "clip", "Frozen CLIP Encoder", 300, 250, 140, 40, style_orange)
    create_block(root, "gram_schmidt", "Gram-Schmidt\nOrthogonalization\n(Idea 1.1)", 300, 320, 140, 50, style_orange)
    
    create_edge(root, "e_txt1", "text_input", "clip", style_edge)
    create_edge(root, "e_txt2", "clip", "gram_schmidt", style_edge)

    # Masks & DAG
    create_block(root, "mask_mod", "Mask Module\n(Importance & Polarity)", 480, 180, 140, 40, style_purple)
    create_block(root, "dag_loss", "Global DAG Reg\n(Idea 2)", 640, 180, 100, 40, style_loss)
    create_edge(root, "e_mask_dag", "mask_mod", "dag_loss", style_edge_dashed)

    # Level 1: AU-AU Graph
    create_block(root, "sub_null_1", "Semantic Subspace\nNulling (L1: AU-AU)\n(Idea 1)", 300, 420, 140, 50, style_pink)
    create_block(root, "gat_1", "Dense Dynamic Graph\nConv (AU-AU)\n(Idea 2)", 480, 420, 140, 50, style_pink)
    create_block(root, "upd_au", "Updated AU\nNodes", 650, 420, 90, 50, style_data)
    
    # Level 2: AU-Exp Graph
    create_block(root, "sub_null_2", "Semantic Subspace\nNulling (L2: AU-Exp)\n(Idea 1)", 300, 520, 140, 50, style_pink)
    create_block(root, "gat_2", "Dense Dynamic Graph\nConv (AU-Exp)\n(Idea 2)", 480, 520, 140, 50, style_pink)
    create_block(root, "upd_exp", "Updated Emotion\nNodes", 650, 520, 90, 50, style_data)
    
    # Phase 2 connections
    create_edge(root, "e_au_null1", "au_emb", "sub_null_1", style_edge)
    create_edge(root, "e_exp_null2", "exp_emb", "sub_null_2", style_edge)
    
    create_edge(root, "e_gs_null1", "gram_schmidt", "sub_null_1", style_edge)
    create_edge(root, "e_gs_null2", "gram_schmidt", "sub_null_2", style_edge)
    
    create_edge(root, "e_null1_gat1", "sub_null_1", "gat_1", style_edge)
    create_edge(root, "e_null2_gat2", "sub_null_2", "gat_2", style_edge)
    
    create_edge(root, "e_mask_gat1", "mask_mod", "gat_1", style_edge, waypoints=[(550, 220)])
    create_edge(root, "e_mask_gat2", "mask_mod", "gat_2", style_edge, waypoints=[(550, 220)])
    
    create_edge(root, "e_gat1_upd", "gat_1", "upd_au", style_edge)
    create_edge(root, "e_gat2_upd", "gat_2", "upd_exp", style_edge)

    # -------------------------------------------------------------
    # PHASE 3: ABDUCTIVE INFERENCE (x = 800 to 1050)
    # -------------------------------------------------------------
    create_block(root, "class_au", "AU Graph\nClassifiers", 800, 420, 100, 50, style_blue)
    create_block(root, "class_exp", "Emotion Graph\nClassifiers", 800, 520, 100, 50, style_blue)
    
    create_block(root, "abd_inf", "Test-Time Abductive\nInference (Idea 3)", 950, 420, 110, 150, style_green)
    
    create_block(root, "facs_loss", "FACS Symbolic\nRule Violations", 950, 310, 110, 40, style_loss)
    create_block(root, "cyc_loss", "Bidirectional Cycle\nConsistency (Idea 4)", 950, 230, 110, 50, style_loss)
    
    create_block(root, "final_out", "Final Predictions", 950, 600, 110, 40, style_input)

    create_edge(root, "e_upd1_c1", "upd_au", "class_au", style_edge)
    create_edge(root, "e_upd2_c2", "upd_exp", "class_exp", style_edge)
    
    create_edge(root, "e_c1_abd", "class_au", "abd_inf", style_edge)
    create_edge(root, "e_c2_abd", "class_exp", "abd_inf", style_edge)
    
    create_edge(root, "e_abd_facs", "abd_inf", "facs_loss", style_edge_dashed)
    create_edge(root, "e_abd_cyc", "abd_inf", "cyc_loss", style_edge_dashed)
    
    create_edge(root, "e_abd_out", "abd_inf", "final_out", style_edge)
    
    # Save to file
    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ", level=0)
    tree.write("ctrlau_architecture_v2.drawio", encoding="utf-8", xml_declaration=True)
    print("V2 Drawio generated!")

if __name__ == "__main__":
    generate_drawio()
