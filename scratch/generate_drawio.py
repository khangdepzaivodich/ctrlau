import xml.etree.ElementTree as ET
import os

def create_block(root, id, value, x, y, width, height, style):
    cell = ET.SubElement(root, "mxCell")
    cell.set("id", id)
    cell.set("value", value)
    cell.set("style", style)
    cell.set("vertex", "1")
    cell.set("parent", "1")
    
    geo = ET.SubElement(cell, "mxGeometry")
    geo.set("x", str(x))
    geo.set("y", str(y))
    geo.set("width", str(width))
    geo.set("height", str(height))
    geo.set("as", "geometry")

def create_edge(root, id, source, target, style, waypoints=None):
    cell = ET.SubElement(root, "mxCell")
    cell.set("id", id)
    cell.set("style", style)
    cell.set("edge", "1")
    cell.set("parent", "1")
    cell.set("source", source)
    cell.set("target", target)
    
    geo = ET.SubElement(cell, "mxGeometry")
    geo.set("relative", "1")
    geo.set("as", "geometry")
    
    if waypoints:
        arr = ET.SubElement(geo, "Array")
        arr.set("as", "points")
        for wx, wy in waypoints:
            pt = ET.SubElement(arr, "mxPoint")
            pt.set("x", str(wx))
            pt.set("y", str(wy))

def generate_drawio():
    mxfile = ET.Element("mxfile", version="14.6.11")
    diagram = ET.SubElement(mxfile, "diagram", id="ctrlau_diagram", name="CtrlAU Architecture")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="1000", dy="1000", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="827", pageHeight="1169", math="0", shadow="0")
    root = ET.SubElement(mxGraphModel, "root")
    
    # Base elements
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    # STYLES (Attention Is All You Need colors)
    # Pink: Multi-Head Attention
    style_pink = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontColor=#000000;fontStyle=1"
    # Blue: Feed Forward
    style_blue = "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontColor=#000000;fontStyle=1"
    # Yellow: Add & Norm
    style_yellow = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontColor=#000000;fontStyle=1"
    # Green: Softmax/Output
    style_green = "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontColor=#000000;fontStyle=1"
    # Gray: Text/Inputs
    style_gray = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontColor=#333333;fontStyle=1"
    
    # Edge styles
    style_edge = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#000000;"
    style_edge_dashed = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#000000;dashed=1;"
    
    # -------------------------------------------------------------
    # LEFT COLUMN (Phase 1: Feature Extraction)
    # -------------------------------------------------------------
    x_left = 150
    w = 180
    h = 40
    
    create_block(root, "img_input", "Face Image Input", x_left, 100, w, h, style_gray)
    create_block(root, "resnet", "ResNet-50 Backbone", x_left, 180, w, h, style_blue)
    create_block(root, "linear", "Linear Projection", x_left, 260, w, h, style_blue)
    
    # Add & Norm for features
    create_block(root, "addnorm1", "Add & Norm", x_left, 340, w, h, style_yellow)
    
    create_block(root, "extractors", "AU & Emotion Extractors", x_left, 420, w, h, style_blue)
    
    create_block(root, "base_emb", "Base Visual Embeddings", x_left, 500, w, h, style_green)

    # Left connections
    create_edge(root, "e_img_res", "img_input", "resnet", style_edge)
    create_edge(root, "e_res_lin", "resnet", "linear", style_edge)
    create_edge(root, "e_lin_an1", "linear", "addnorm1", style_edge)
    create_edge(root, "e_an1_ext", "addnorm1", "extractors", style_edge)
    create_edge(root, "e_ext_emb", "extractors", "base_emb", style_edge)
    
    # Left residual (resnet to addnorm1)
    create_edge(root, "e_res_res1", "resnet", "addnorm1", style_edge, waypoints=[(x_left-30, 200), (x_left-30, 360)])

    # -------------------------------------------------------------
    # RIGHT COLUMN (Phase 2 & 3: Causal Graph)
    # -------------------------------------------------------------
    x_right = 500
    
    create_block(root, "text_input", "CLIP Text Prompts", x_right, 100, w, h, style_gray)
    create_block(root, "clip", "Frozen CLIP Encoder", x_right, 180, w, h, style_blue)
    
    create_block(root, "gs_ortho", "Gram-Schmidt Orthogonalization\n(Idea 1.1)", x_right, 260, w, h, style_yellow)
    
    create_block(root, "sub_null", "Semantic Subspace Nulling\n(Idea 1)", x_right, 340, w, h, style_pink)
    
    create_block(root, "addnorm2", "Add & Norm", x_right, 420, w, h, style_yellow)
    
    create_block(root, "dense_gat", "Dense Dynamic Causal Graph\n(Idea 2)", x_right, 500, w, h, style_pink)
    
    create_block(root, "addnorm3", "Add & Norm", x_right, 580, w, h, style_yellow)
    
    create_block(root, "abductive", "Test-Time Abductive Inference\n(Idea 3)", x_right, 660, w, h, style_blue)
    
    create_block(root, "output", "Final Emotion & AU Probs", x_right, 740, w, h, style_green)
    
    # Right connections
    create_edge(root, "e_txt_clip", "text_input", "clip", style_edge)
    create_edge(root, "e_clip_gs", "clip", "gs_ortho", style_edge)
    create_edge(root, "e_gs_null", "gs_ortho", "sub_null", style_edge)
    create_edge(root, "e_null_an2", "sub_null", "addnorm2", style_edge)
    create_edge(root, "e_an2_gat", "addnorm2", "dense_gat", style_edge)
    create_edge(root, "e_gat_an3", "dense_gat", "addnorm3", style_edge)
    create_edge(root, "e_an3_abd", "addnorm3", "abductive", style_edge)
    create_edge(root, "e_abd_out", "abductive", "output", style_edge)
    
    # Right residuals
    create_edge(root, "e_res_an2", "gs_ortho", "addnorm2", style_edge, waypoints=[(x_right+w+30, 280), (x_right+w+30, 440)])
    create_edge(root, "e_res_an3", "addnorm2", "addnorm3", style_edge, waypoints=[(x_right+w+30, 440), (x_right+w+30, 600)])

    # -------------------------------------------------------------
    # CROSS COLUMN CONNECTIONS
    # -------------------------------------------------------------
    # Base Emb to Subspace Nulling (Visual Input)
    create_edge(root, "e_cross_1", "base_emb", "sub_null", style_edge, waypoints=[(x_left+w/2, 540), (x_left+w/2, 360)])

    # -------------------------------------------------------------
    # LOSSES & REGULARIZATIONS (Dashed)
    # -------------------------------------------------------------
    create_block(root, "dag_loss", "Global DAG Reg", x_right+250, 500, 100, 40, "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#b71c1c;dashed=1;")
    create_edge(root, "e_dag_gat", "dag_loss", "dense_gat", style_edge_dashed)
    
    create_block(root, "cycle_loss", "Bidirectional Cycle Loss\n(Idea 4)", x_right+250, 660, 150, 40, "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#b71c1c;dashed=1;")
    create_edge(root, "e_cyc_abd", "cycle_loss", "abductive", style_edge_dashed)

    # Convert to XML string and save
    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ", level=0)
    tree.write("ctrlau_architecture.drawio", encoding="utf-8", xml_declaration=True)
    print("Drawio generated!")

if __name__ == "__main__":
    generate_drawio()
