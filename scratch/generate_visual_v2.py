import math
import os

def build():
    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append('<mxfile version="14.6.11">')
    xml.append('  <diagram id="ctrlau_visual" name="CtrlAU">')
    xml.append('    <mxGraphModel dx="2000" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2200" pageHeight="1600" math="0" shadow="0">')
    xml.append('      <root>')
    xml.append('        <mxCell id="0"/>')
    xml.append('        <mxCell id="1" parent="0"/>')

    def add_node(id, value, style, x, y, w, h):
        # We pass actual <font> in python, this replaces it to &lt;font&gt; for XML
        value = value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        xml.append(f'        <mxCell id="{id}" value="{value}" style="{style}" vertex="1" parent="1">')
        xml.append(f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>')
        xml.append('        </mxCell>')

    def add_edge(id, source, target, style, points=None):
        xml.append(f'        <mxCell id="{id}" style="{style}" edge="1" parent="1" source="{source}" target="{target}">')
        if points:
            pts = "".join([f'<mxPoint x="{px}" y="{py}"/>' for px, py in points])
            xml.append(f'          <mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry>')
        else:
            xml.append(f'          <mxGeometry relative="1" as="geometry"/>')
        xml.append('        </mxCell>')

    # -------------------------------------------------------------
    # PHASE BOUNDARIES
    # -------------------------------------------------------------
    add_node("bg1", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=none;opacity=50;", 10, 10, 500, 1500)
    add_node("bg2", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f8e9;strokeColor=none;opacity=50;", 530, 10, 1200, 1500)
    add_node("bg3", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff8e1;strokeColor=none;opacity=50;", 1750, 10, 420, 1500)

    add_node("p1_lbl", "Phase 1: Feature Extraction", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#343a40;", 20, 20, 480, 40)
    add_node("p2_lbl", "Phase 2: Causal Graph Reasoning & Intervention", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#33691e;", 550, 20, 1160, 40)
    add_node("p3_lbl", "Phase 3: Abductive Inference", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#ff6f00;", 1760, 20, 400, 40)

    # -------------------------------------------------------------
    # 1. INPUT IMAGE
    # -------------------------------------------------------------
    img_x = 50
    img_y = 150
    add_node("img_p2", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#424242;strokeWidth=2;", img_x+10, img_y-10, 80, 80)
    add_node("img_p1", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#eceff1;strokeColor=#424242;strokeWidth=2;", img_x, img_y, 80, 80)
    add_node("img_face", "<font style='font-size:36px'>👤</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;", img_x+10, img_y+10, 60, 60)
    add_node("img_lbl", "Input\nImage", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;fontSize=14;fontStyle=1;", img_x, img_y+95, 80, 40)

    # -------------------------------------------------------------
    # 2. RESNET BACKBONE
    # -------------------------------------------------------------
    rn_x = 50
    rn_y = 330
    add_node("cnn_trap", "ResNet-50", "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;direction=south;fillColor=#bbdefb;strokeColor=#1565c0;strokeWidth=2;fontSize=14;fontStyle=1;", rn_x+10, rn_y, 60, 120)
    add_edge("e_img_cnn", "img_p1", "cnn_trap", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=3;strokeColor=#455a64;endArrow=block;")

    # -------------------------------------------------------------
    # 3. FEATURE MAP Z
    # -------------------------------------------------------------
    z_x = 50
    z_y = 540
    add_node("z_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#eeeeee;strokeColor=#757575;", z_x, z_y, 80, 80)
    for i in range(4):
        for j in range(4):
            fc = "#cfd8dc" if (i+j)%2==0 else "#ffffff"
            add_node(f"z_cell_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#b0bec5;", z_x + j*20, z_y + i*20, 20, 20)
    add_node("z_lbl", "Visual Features Z", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;fontSize=14;fontStyle=1;", z_x-20, z_y+90, 120, 40)
    add_edge("e_cnn_z", "cnn_trap", "z_bg", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=3;strokeColor=#455a64;endArrow=block;")

    # -------------------------------------------------------------
    # 4. SYM-AU & SYM-EXPR HEADS
    # -------------------------------------------------------------
    head_x = 240
    head_y_au = 420
    head_y_ex = 650
    
    add_node("au_head_lbl", "SymAU Head (8 Branches)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontStyle=1;fontSize=14;fontColor=#2e7d32;", head_x, head_y_au-40, 200, 20)
    au_nodes = []
    for i in range(4):
        nid = f"au_branch_{i}"
        au_nodes.append(nid)
        lbl = f"AU{i+1}" if i < 3 else "..."
        add_node(nid, lbl, "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#c8e6c9;strokeColor=#388e3c;strokeWidth=2;fontStyle=1;", head_x+50, head_y_au + i*45, 35, 35)
        add_edge(f"e_z_{nid}", "z_bg", nid, "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor=#81c784;endArrow=block;", [(head_x-40, z_y+40), (head_x-20, head_y_au+17+i*45)])
    
    # Vector Representation Visual
    for i in range(4):
        nid = f"v_au_{i}"
        add_node(f"{nid}_vis", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#a5d6a7;strokeColor=#1b5e20;", head_x+130, head_y_au + i*45 + 5, 40, 10)
        add_node(f"{nid}_vis2", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#81c784;strokeColor=#1b5e20;", head_x+170, head_y_au + i*45 + 5, 10, 10)
        add_node(nid, "V_a", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=12;", head_x+130, head_y_au + i*45 + 15, 40, 20)
        add_edge(f"e_aub_{nid}", au_nodes[i], f"{nid}_vis", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;")

    add_node("ex_head_lbl", "SymExpr Head (7 Branches)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontStyle=1;fontSize=14;fontColor=#c62828;", head_x, head_y_ex-40, 200, 20)
    ex_nodes = []
    for i in range(4):
        nid = f"ex_branch_{i}"
        ex_nodes.append(nid)
        lbl = ["Anger","Fear","Happy","..."][i]
        add_node(nid, lbl, "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffcdd2;strokeColor=#e53935;strokeWidth=2;fontSize=10;fontStyle=1;", head_x+50, head_y_ex + i*45, 35, 35)
        add_edge(f"e_z_{nid}", "z_bg", nid, "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor=#e57373;endArrow=block;", [(head_x-40, z_y+40), (head_x-20, head_y_ex+17+i*45)])

    for i in range(4):
        nid = f"v_ex_{i}"
        add_node(f"{nid}_vis", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef9a9a;strokeColor=#b71c1c;", head_x+130, head_y_ex + i*45 + 5, 40, 10)
        add_node(f"{nid}_vis2", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#e57373;strokeColor=#b71c1c;", head_x+170, head_y_ex + i*45 + 5, 10, 10)
        add_node(nid, "V_e", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=12;", head_x+130, head_y_ex + i*45 + 15, 40, 20)
        add_edge(f"e_exb_{nid}", ex_nodes[i], f"{nid}_vis", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;")

    # -------------------------------------------------------------
    # 5. TEXT & CLIP
    # -------------------------------------------------------------
    clip_x = 240
    clip_y = 120
    add_node("text_doc", "AU / Expr\nDescriptions", "shape=document;whiteSpace=wrap;html=1;boundedLbl=1;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=2;fontSize=12;fontStyle=1;", clip_x, clip_y, 100, 70)
    add_node("clip_enc", "Frozen CLIP\nText Encoder", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe082;strokeColor=#ff8f00;strokeWidth=2;fontStyle=1;fontSize=12;", clip_x+140, clip_y+10, 100, 50)
    add_edge("e_doc_clip", "text_doc", "clip_enc", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#f57f17;endArrow=block;")
    
    # Text embeddings T
    add_node("t_emb_grp", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#ff6f00;strokeWidth=2;dashed=1;", clip_x+280, clip_y-10, 70, 90)
    add_node("t_emb_lbl", "Text Embeddings", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=bottom;fontSize=12;fontStyle=1;fontColor=#e65100;", clip_x+265, clip_y-35, 100, 20)
    for i in range(3):
        add_node(f"t_vec_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffcc80;strokeColor=#e65100;", clip_x+295, clip_y + i*25, 40, 10)
        add_node(f"t_vec_{i}_lbl", "T", "text;html=1;strokeColor=none;fillColor=none;align=right;fontSize=10;", clip_x+280, clip_y + i*25 - 5, 10, 20)
    add_edge("e_clip_t", "clip_enc", "t_emb_grp", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#f57f17;endArrow=block;")

    # -------------------------------------------------------------
    # 6. GRAM-SCHMIDT & VECTOR PROJECTION (Idea 1)
    # -------------------------------------------------------------
    proj_x = 580
    proj_y = 150
    # Fixed HTML raw text
    add_node("idea1_lbl", "<font color='#1565c0'><b>Idea 1 & 1.1: Orthogonalization & Semantic Subspace Nulling</b></font>", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;", proj_x, proj_y-50, 500, 30)
    
    add_node("proj_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#1565c0;strokeWidth=2;", proj_x, proj_y, 450, 180)
    
    # Visual Vector Math
    ox2, oy2 = proj_x + 150, proj_y + 140
    add_node("pt_o2", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2, oy2, 1, 1)
    
    # T direction (Orthogonalized Text)
    add_node("pt_tdir", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2+180, oy2, 1, 1)
    add_edge("v_tdir", "pt_o2", "pt_tdir", "endArrow=classic;html=1;strokeWidth=3;strokeColor=#e65100;dashed=1;")
    add_node("tdir_lbl", "T_ortho (Orthogonal Text Direction)", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=12;fontColor=#e65100;fontStyle=1;", ox2+100, oy2+10, 200, 20)
    
    # V_orig (Visual feature)
    add_node("pt_v", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2+120, oy2-100, 1, 1)
    add_edge("v_vorig", "pt_o2", "pt_v", "endArrow=classic;html=1;strokeWidth=4;strokeColor=#1b5e20;")
    add_node("vorig_lbl", "V_orig", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontColor=#1b5e20;fontStyle=1;", ox2+120, oy2-125, 50, 20)
    
    # Projection down
    add_node("pt_proj", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2+120, oy2, 1, 1)
    add_edge("v_projline", "pt_v", "pt_proj", "endArrow=none;html=1;strokeWidth=2;strokeColor=#9e9e9e;dashed=1;")
    
    # V_counter (Nullified vector)
    add_node("pt_vortho", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2, oy2-100, 1, 1)
    add_edge("v_vortho", "pt_o2", "pt_vortho", "endArrow=classic;html=1;strokeWidth=4;strokeColor=#4a148c;")
    add_node("vortho_lbl", "V_counter (Intervened)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontColor=#4a148c;fontStyle=1;", ox2-100, oy2-125, 180, 20)
    add_node("eq_null", "V_counter = V - (V·T)T", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontColor=#4a148c;fontStyle=1;", proj_x+250, proj_y+20, 180, 20)

    # Connections to projection
    add_edge("e_temb_proj", "t_emb_grp", "proj_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#f57f17;endArrow=block;", [(clip_x+400, clip_y+35), (clip_x+400, proj_y+90)])
    add_edge("e_vau_proj", "v_au_1_vis", "proj_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#43a047;endArrow=block;", [(head_x+210, head_y_au+50), (head_x+210, proj_y+90)])
    
    # Output of SSN branches into Important / Unimportant Paths
    add_node("cf_imp", "Important Path\n(Intervened V_counter)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#c62828;strokeWidth=2;fontStyle=1;fontSize=12;", proj_x+60, proj_y+240, 160, 50)
    add_node("cf_unimp", "Unimportant Path\n(Consistent V_orig)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#2e7d32;strokeWidth=2;fontStyle=1;fontSize=12;", proj_x+250, proj_y+240, 160, 50)
    add_edge("e_proj_imp", "proj_box", "cf_imp", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;")
    add_edge("e_proj_unimp", "proj_box", "cf_unimp", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#2e7d32;endArrow=block;")

    # -------------------------------------------------------------
    # 7. CAUSAL GRAPHS (Idea 2)
    # -------------------------------------------------------------
    graph_x = 580
    graph_y = 550
    # Fixed HTML raw text
    add_node("idea2_lbl", "<font color='#2e7d32'><b>Idea 2: Sample-Adaptive Causal Routing Graph</b></font>", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;", graph_x, graph_y-40, 450, 30)
    
    # Level 1 AU-AU Graph
    add_node("g1_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#455a64;strokeWidth=2;", graph_x, graph_y, 500, 340)
    add_node("g1_lbl", "Level 1: AU → AU Causal Relational Graph", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;fontSize=14;fontStyle=1;", graph_x, graph_y+15, 500, 20)
    
    # Visual Graph Drawing
    g1_r = 70
    g1_cx, g1_cy = graph_x + 130, graph_y + 170
    g1_nodes = []
    for i in range(4):
        nid = f"g1_n{i}"
        g1_nodes.append(nid)
        ang = i * math.pi / 2 - math.pi/4
        nx = g1_cx + g1_r * math.cos(ang) - 25
        ny = g1_cy + g1_r * math.sin(ang) - 25
        add_node(nid, f"AU{i+1}", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#c8e6c9;strokeColor=#388e3c;strokeWidth=2;fontStyle=1;fontSize=14;", nx, ny, 50, 50)
    
    add_edge("e_g1_01", g1_nodes[0], g1_nodes[1], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=block;")
    add_edge("e_g1_02", g1_nodes[0], g1_nodes[2], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=block;")
    add_edge("e_g1_13", g1_nodes[1], g1_nodes[3], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=block;")
    add_edge("e_g1_23", g1_nodes[2], g1_nodes[3], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=block;")
    
    # Matrices
    adj_x = graph_x + 270
    adj_y = graph_y + 60
    
    add_node("g_inv_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=2;", adj_x, adj_y, 75, 75)
    for i in range(3):
        for j in range(3):
            fc = "#fff176" if (i*3+j)%4==0 else "#ffffff"
            add_node(f"ginv_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#ffd54f;", adj_x + j*25, adj_y + i*25, 25, 25)
    # Move label to the right of the matrix
    add_node("ginv_lbl", "G_inv\n(Global Prior)", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=12;fontStyle=1;", adj_x+90, adj_y+20, 100, 30)
    
    add_node("g_dyn_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#e3f2fd;strokeColor=#1e88e5;strokeWidth=2;", adj_x, adj_y+160, 75, 75)
    for i in range(3):
        for j in range(3):
            fc = "#64b5f6" if (i*3+j)%5==0 else "#e3f2fd"
            add_node(f"gdyn_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#90caf9;", adj_x + j*25, adj_y + 160 + i*25, 25, 25)
    # Move label to the right of the matrix
    add_node("gdyn_lbl", "G_dyn = σ(QKᵀ)\n(Sample-Adaptive)", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=12;fontStyle=1;", adj_x+90, adj_y+180, 120, 30)
    
    # Put operator perfectly between them
    add_node("op_mul", "⊙", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=36;fontStyle=1;", adj_x+25, adj_y+105, 25, 25)

    # LEVEL 2 GRAPH
    graph2_y = 950
    add_node("g2_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#455a64;strokeWidth=2;", graph_x, graph2_y, 500, 260)
    add_node("g2_lbl", "Level 2: AU → Expression Bipartite Causal Graph", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;fontSize=14;fontStyle=1;", graph_x, graph2_y+15, 500, 20)
    
    b_cx1 = graph_x + 110
    b_cx2 = graph_x + 300
    b_nodes_a = []
    b_nodes_e = []
    for i in range(3):
        nid_a = f"g2_a{i}"
        b_nodes_a.append(nid_a)
        # Moved nodes down slightly to avoid overlap with label
        add_node(nid_a, f"AU{i+1}", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#c8e6c9;strokeColor=#388e3c;strokeWidth=2;fontSize=14;fontStyle=1;", b_cx1, graph2_y+60 + i*60, 45, 45)
        
        nid_e = f"g2_e{i}"
        b_nodes_e.append(nid_e)
        add_node(nid_e, f"E{i+1}", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffcdd2;strokeColor=#c62828;strokeWidth=2;fontSize=14;fontStyle=1;", b_cx2, graph2_y+60 + i*60, 45, 45)
        
    for i in range(3):
        for j in range(3):
            if (i+j)%2 == 0 or i==j:
                add_edge(f"e_g2_{i}{j}", b_nodes_a[i], b_nodes_e[j], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor=#455a64;endArrow=block;")

    # MASK MODULE VISUALIZATION
    mask_x = 1130
    mask_y = 700
    # Increase height from 200 to 240
    add_node("mask_mod_bg", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#7b1fa2;strokeWidth=2;", mask_x, mask_y, 140, 240)
    add_node("mask_mod_lbl", "Mask Module", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#6a1b9a;", mask_x, mask_y+10, 140, 20)
    
    # Draw Mask M_imp
    add_node("mask_imp_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;strokeWidth=2;", mask_x+40, mask_y+40, 60, 60)
    add_node("mv_b1", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#000000;strokeColor=none;", mask_x+40, mask_y+40, 20, 20)
    add_node("mv_b2", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#000000;strokeColor=none;", mask_x+60, mask_y+80, 20, 20)
    add_node("mv_b3", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#000000;strokeColor=none;", mask_x+80, mask_y+60, 20, 20)
    add_node("mimp_lbl", "M_imp (0/1)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;", mask_x+20, mask_y+105, 100, 20)
    
    # Mask polarity M_pol (moved further down)
    add_node("mask_pol_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#000000;strokeWidth=2;", mask_x+40, mask_y+140, 60, 60)
    add_node("mv_r1", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f44336;strokeColor=none;opacity=70;", mask_x+40, mask_y+140, 20, 20)
    add_node("mv_g1", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#4caf50;strokeColor=none;opacity=70;", mask_x+60, mask_y+180, 20, 20)
    add_node("mpol_lbl", "M_pol (+1/-1)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;", mask_x+20, mask_y+210, 100, 20)

    # Connections for masks and graphs
    add_edge("e_imp_g1", "cf_imp", "g1_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;")
    add_edge("e_unimp_g1", "cf_unimp", "g1_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#2e7d32;endArrow=block;")
    
    add_edge("e_vau_g1", "v_au_0_vis", "g1_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#388e3c;endArrow=block;", [(head_x+230, head_y_au+10), (head_x+230, graph_y-20), (graph_x+50, graph_y-20)])
    
    add_edge("e_g1_g2", "g1_box", "g2_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=5;strokeColor=#388e3c;endArrow=block;")
    add_edge("e_vex_g2", "v_ex_0_vis", "g2_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;", [(head_x+230, head_y_ex+10), (head_x+230, graph2_y-20), (graph_x+50, graph2_y-20)])
    
    # Mask connecting to graphs
    add_edge("e_mask_g1", "mask_mod_bg", "g1_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#7b1fa2;endArrow=block;dashed=1;", [(mask_x+70, mask_y-30), (graph_x+225, mask_y-30)])

    # -------------------------------------------------------------
    # 8. CYCLE CONSISTENCY (Idea 4)
    # -------------------------------------------------------------
    cyc_x = 1300
    cyc_y = 800
    # Fixed HTML raw text
    add_node("idea4_lbl", "<font color='#7b1fa2'><b>Idea 4: Causal Cycle Consistency</b></font>", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;", cyc_x, cyc_y-40, 400, 30)
    
    add_node("cyc_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fafafa;strokeColor=#7b1fa2;strokeWidth=3;dashed=1;", cyc_x, cyc_y, 420, 200)
    
    # Bar charts for distributions
    add_node("cyc_p_ex", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;", cyc_x+30, cyc_y+60, 80, 100)
    for i, h in enumerate([50, 90, 30, 60]):
        add_node(f"cb_e{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef5350;strokeColor=#b71c1c;", cyc_x+35+i*15, cyc_y+150-h, 12, h)
    add_node("cb_e_lbl", "P_Exp", "text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=14;", cyc_x+30, cyc_y+155, 80, 20)

    add_node("cyc_mul", "×", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=36;fontStyle=1;", cyc_x+115, cyc_y+90, 30, 30)

    # Matrix M_AE
    mx_x = cyc_x + 160
    mx_y = cyc_y + 50
    add_node("mae_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#8e24aa;strokeWidth=2;", mx_x, mx_y, 80, 100)
    for i in range(5):
        for j in range(4):
            fc = "#ce93d8" if (i*2+j)%3==0 else "#f3e5f5"
            add_node(f"mae_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#ab47bc;", mx_x + j*20, mx_y + i*20, 20, 20)
    add_node("mae_lbl", "M_AEᵀ Prior Matrix", "text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=14;fontColor=#8e24aa;", mx_x-20, mx_y+105, 120, 20) # Made y offset smaller for less overlap with matrix

    add_node("cyc_eq", "≈", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=36;fontStyle=1;", mx_x+90, cyc_y+90, 30, 30)

    # Target AU bars
    add_node("cyc_p_au", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;", cyc_x+280, cyc_y+60, 80, 100)
    for i, h in enumerate([70, 40, 95, 50]):
        add_node(f"cb_a{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#66bb6a;strokeColor=#2e7d32;", cyc_x+285+i*15, cyc_y+150-h, 12, h)
    add_node("cb_a_lbl", "P_AU", "text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=14;", cyc_x+280, cyc_y+155, 80, 20)

    add_edge("e_g2_cyc", "g2_box", "cyc_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#7b1fa2;endArrow=block;dashed=1;")

    # -------------------------------------------------------------
    # 9. ABDUCTIVE INFERENCE (Idea 3)
    # -------------------------------------------------------------
    abd_x = 1800
    abd_y = 550
    # Fixed HTML raw text
    add_node("idea3_lbl", "<font color='#ff6f00'><b>Idea 3: Test-Time Abductive Inference</b></font>", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;", abd_x, abd_y-40, 350, 30)
    
    add_node("abd_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#ff8f00;strokeWidth=4;", abd_x, abd_y, 320, 350)
    add_node("abd_lbl", "Energy Minimization", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor=#e65100;", abd_x, abd_y+15, 320, 30)
    
    # Circular Iteration Path
    add_node("loop_bg", "", "ellipse;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#ffca28;strokeWidth=5;dashed=1;", abd_x+85, abd_y+70, 150, 150) # Moved down slightly
    add_edge("e_loop1", "loop_bg", "loop_bg", "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=5;strokeColor=#ff8f00;endArrow=block;", [(abd_x+85, abd_y+145), (abd_x+115, abd_y+80), (abd_x+160, abd_y+70)])
    
    add_node("loop_txt", "Adam Optimizer\n(15 iterations)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontStyle=1;fontSize=14;", abd_x+90, abd_y+125, 140, 40)
    
    # Equation
    add_node("abd_eq", "min <font color='#1565c0'>E(FACS)</font> + <font color='#7b1fa2'>E(Cycle)</font> + <font color='#2e7d32'>E(Prior)</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", abd_x+10, abd_y+250, 300, 30)
    add_node("abd_desc", "Adjusts logits at test-time to\nsatisfy symbolic constraints", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;", abd_x+10, abd_y+290, 300, 40)
    
    # Outputs
    add_node("final_pred_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e0f7fa;strokeColor=#006064;strokeWidth=3;", abd_x+60, abd_y+460, 200, 140)
    add_node("final_lbl", "Final Refined\nPredictions", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;", abd_x+60, abd_y+470, 200, 40)
    
    for i, h in enumerate([30, 60, 90, 50]):
        add_node(f"fp_a{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#66bb6a;strokeColor=#2e7d32;", abd_x+85+i*15, abd_y+580-h, 12, h)
    for i, h in enumerate([80, 40, 30, 70]):
        add_node(f"fp_e{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef5350;strokeColor=#b71c1c;", abd_x+165+i*15, abd_y+580-h, 12, h)

    add_edge("e_abd_final", "abd_box", "final_pred_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#006064;endArrow=block;")

    add_edge("e_g2_abd", "g2_box", "abd_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#455a64;endArrow=block;", [(graph_x+500, graph2_y+130), (abd_x-50, graph2_y+130), (abd_x-50, abd_y+160)])

    # -------------------------------------------------------------
    # LOSSES
    # -------------------------------------------------------------
    add_node("p1_loss_box", "Phase 1 Losses:\nL_WA, L_Expr, L_contrastive, L_HSIC, L_FACS", "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;darkOpacity=0.05;fillColor=#ffffff;strokeColor=#999999;size=15;fontSize=12;fontStyle=1;", 50, 1000, 300, 60)
    add_node("p2_loss_box", "Phase 2 Losses:\nL_DAG, L_CF, L_violation, L_cycle", "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;darkOpacity=0.05;fillColor=#ffffff;strokeColor=#999999;size=15;fontSize=12;fontStyle=1;", graph_x+50, graph2_y+320, 300, 60)

    # Connections for losses
    add_edge("e_l1", "v_au_2_vis", "p1_loss_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#9e9e9e;endArrow=none;dashed=1;", [(head_x+190, head_y_au+100), (head_x+190, 950), (200, 950)])
    
    xml.append('      </root>')
    xml.append('    </mxGraphModel>')
    xml.append('  </diagram>')
    xml.append('</mxfile>')

    with open(r"C:\Users\khang\OneDrive\Desktop\ctrlau\ctrlau_architecture.drawio", "w", encoding="utf-8") as f:
        f.write("\n".join(xml))

if __name__ == "__main__":
    build()
