import math
import os

def build():
    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append('<mxfile version="14.6.11">')
    xml.append('  <diagram id="ctrlau_v3" name="CtrlAU Architecture">')
    # Width 2700, Height 1400
    xml.append('    <mxGraphModel dx="2000" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2700" pageHeight="1400" math="0" shadow="0">')
    xml.append('      <root>')
    xml.append('        <mxCell id="0"/>')
    xml.append('        <mxCell id="1" parent="0"/>')

    def add_node(id, value, style, x, y, w, h):
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
    # BACKGROUNDS & LABELS
    # -------------------------------------------------------------
    add_node("bg1", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=none;opacity=50;", 10, 10, 650, 1350)
    add_node("bg2", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f8e9;strokeColor=none;opacity=50;", 680, 10, 1150, 1350)
    add_node("bg3", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff8e1;strokeColor=none;opacity=50;", 1850, 10, 700, 1350)

    add_node("p1_lbl", "Phase 1: Feature Extraction & Alignment", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#343a40;", 20, 20, 630, 40)
    add_node("p2_lbl", "Phase 2: Parallel Causal Routing & Counterfactual Intervention", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#33691e;", 700, 20, 1100, 40)
    add_node("p3_lbl", "Phase 3: Abductive Inference & Cycle Consistency", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#ff6f00;", 1860, 20, 680, 40)

    # -------------------------------------------------------------
    # PHASE 1: TEXT & VISION
    # -------------------------------------------------------------
    # Text Stream
    clip_x, clip_y = 100, 100
    add_node("text_doc", "AU / Expr\nDescriptions", "shape=document;whiteSpace=wrap;html=1;boundedLbl=1;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=2;fontSize=12;fontStyle=1;", clip_x, clip_y, 100, 70)
    add_node("clip_enc", "Frozen CLIP\nText Encoder", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe082;strokeColor=#ff8f00;strokeWidth=2;fontStyle=1;fontSize=12;", clip_x+150, clip_y+10, 100, 50)
    add_edge("e_doc_clip", "text_doc", "clip_enc", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#f57f17;endArrow=block;")
    
    add_node("t_emb_grp", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#ff6f00;strokeWidth=2;dashed=1;", clip_x+300, clip_y-10, 90, 100)
    add_node("t_emb_lbl", "Text Embeddings", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=bottom;fontSize=12;fontStyle=1;fontColor=#e65100;", clip_x+280, clip_y-35, 120, 20)
    for i in range(3):
        add_node(f"t_vec_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffcc80;strokeColor=#e65100;", clip_x+320, clip_y + i*25, 40, 10)
        add_node(f"t_vec_lbl_{i}", "T", "text;html=1;strokeColor=none;fillColor=none;align=right;fontSize=12;fontStyle=1;", clip_x+295, clip_y + i*25 - 5, 20, 20)
    add_edge("e_clip_t", "clip_enc", "t_emb_grp", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#f57f17;endArrow=block;")

    # Vision Stream
    img_x, img_y = 50, 450
    add_node("img_p2", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#424242;strokeWidth=2;", img_x+10, img_y-10, 80, 80)
    add_node("img_p1", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#eceff1;strokeColor=#424242;strokeWidth=2;", img_x, img_y, 80, 80)
    add_node("img_face", "<font style='font-size:36px'>👤</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;", img_x+10, img_y+10, 60, 60)
    add_node("img_lbl", "Input Image", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;fontSize=14;fontStyle=1;", img_x, img_y+90, 80, 20)

    rn_y = 650
    add_node("cnn_trap", "ResNet-50", "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;direction=south;fillColor=#bbdefb;strokeColor=#1565c0;strokeWidth=2;fontSize=14;fontStyle=1;", img_x+10, rn_y, 60, 120)
    add_edge("e_img_cnn", "img_p1", "cnn_trap", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=3;strokeColor=#455a64;endArrow=block;")

    z_y = 850
    add_node("z_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#eeeeee;strokeColor=#757575;", img_x, z_y, 80, 80)
    for i in range(4):
        for j in range(4):
            fc = "#cfd8dc" if (i+j)%2==0 else "#ffffff"
            add_node(f"z_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#b0bec5;", img_x + j*20, z_y + i*20, 20, 20)
    add_node("z_lbl", "Visual Features Z", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;fontSize=14;fontStyle=1;", img_x-20, z_y+90, 120, 30)
    add_edge("e_cnn_z", "cnn_trap", "z_bg", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=3;strokeColor=#455a64;endArrow=block;")

    # Sym Heads
    head_x = 240
    head_y_au = 600
    head_y_ex = 950
    
    # AU Head
    add_node("au_head_lbl", "SymAU Head (8 Branches)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontStyle=1;fontSize=14;fontColor=#2e7d32;", head_x, head_y_au-40, 200, 20)
    au_nodes = []
    for i in range(4):
        nid = f"au_branch_{i}"
        au_nodes.append(nid)
        lbl = f"AU{i+1}" if i < 3 else "..."
        add_node(nid, lbl, "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#c8e6c9;strokeColor=#388e3c;strokeWidth=2;fontStyle=1;", head_x+50, head_y_au + i*45, 35, 35)
        add_edge(f"e_z_{nid}", "z_bg", nid, "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor=#81c784;endArrow=block;", [(head_x-40, z_y+40), (head_x-20, head_y_au+17+i*45)])
    
    add_node("vau_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#1b5e20;strokeWidth=2;dashed=1;", head_x+120, head_y_au-10, 60, 200)
    add_node("vau_lbl", "V_a\n(Embeddings)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#1b5e20;", head_x+115, head_y_au-35, 70, 30)
    for i in range(4):
        nid = f"v_au_{i}"
        add_node(f"{nid}_vis", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#a5d6a7;strokeColor=#1b5e20;", head_x+130, head_y_au + i*45 + 5, 40, 10)
        add_edge(f"e_aub_{nid}", au_nodes[i], f"{nid}_vis", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;")

    # AU Phase 1 Probabilities
    add_node("pa1_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#2e7d32;strokeWidth=2;", head_x+220, head_y_au+20, 120, 140)
    add_node("pa1_lbl", "Phase 1\nAU Classifiers", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;", head_x+220, head_y_au+30, 120, 30)
    add_edge("e_vau_pa1", "vau_box", "pa1_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#2e7d32;endArrow=block;")
    
    add_node("pa1_bar_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=none;", head_x+240, head_y_au+70, 80, 80)
    for i, h in enumerate([40, 70, 30, 50]):
        add_node(f"pa1_bar_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#66bb6a;strokeColor=#2e7d32;", head_x+245+i*15, head_y_au+150-h, 12, h)
    add_node("pa1_out", "P_a^(1)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#2e7d32;", head_x+240, head_y_au+150, 80, 20)

    # Expr Head
    add_node("ex_head_lbl", "SymExpr Head (7 Branches)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontStyle=1;fontSize=14;fontColor=#c62828;", head_x, head_y_ex-40, 200, 20)
    ex_nodes = []
    for i in range(4):
        nid = f"ex_branch_{i}"
        ex_nodes.append(nid)
        lbl = ["Anger","Fear","Happy","..."][i]
        add_node(nid, lbl, "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffcdd2;strokeColor=#e53935;strokeWidth=2;fontSize=10;fontStyle=1;", head_x+50, head_y_ex + i*45, 35, 35)
        add_edge(f"e_z_{nid}", "z_bg", nid, "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor=#e57373;endArrow=block;", [(head_x-40, z_y+40), (head_x-20, head_y_ex+17+i*45)])

    add_node("vex_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#b71c1c;strokeWidth=2;dashed=1;", head_x+120, head_y_ex-10, 60, 200)
    add_node("vex_lbl", "V_e\n(Embeddings)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#b71c1c;", head_x+115, head_y_ex-35, 70, 30)
    for i in range(4):
        nid = f"v_ex_{i}"
        add_node(f"{nid}_vis", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef9a9a;strokeColor=#b71c1c;", head_x+130, head_y_ex + i*45 + 5, 40, 10)
        add_edge(f"e_exb_{nid}", ex_nodes[i], f"{nid}_vis", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;")

    # Expr Phase 1 Probabilities
    add_node("pe1_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#c62828;strokeWidth=2;", head_x+220, head_y_ex+20, 120, 140)
    add_node("pe1_lbl", "Phase 1\nExpr Classifiers", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;", head_x+220, head_y_ex+30, 120, 30)
    add_edge("e_vex_pe1", "vex_box", "pe1_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;")
    
    add_node("pe1_bar_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=none;", head_x+240, head_y_ex+70, 80, 80)
    for i, h in enumerate([20, 80, 40, 60]):
        add_node(f"pe1_bar_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef5350;strokeColor=#b71c1c;", head_x+245+i*15, head_y_ex+150-h, 12, h)
    add_node("pe1_out", "P_e^(1)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#c62828;", head_x+240, head_y_ex+150, 80, 20)

    # Contrastive Alignment
    add_node("align_box", "Contrastive\nAlignment\n(L_contra, L_HSIC)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e3f2fd;strokeColor=#1565c0;strokeWidth=2;fontStyle=1;", 400, 300, 140, 60)
    add_edge("e_t_align", "t_emb_grp", "align_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1565c0;endArrow=block;startArrow=block;dashed=1;")
    add_edge("e_vau_align", "vau_box", "align_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1565c0;endArrow=block;startArrow=block;dashed=1;", [(head_x+150, head_y_au-10), (head_x+150, 330)])


    # -------------------------------------------------------------
    # PHASE 2: PARALLEL CAUSAL ROUTING
    # -------------------------------------------------------------
    # We fork V_au into Standard Pass and CF Pass
    p2_start_x = 750
    
    # Text routing to Phase 2
    add_edge("e_t_p2", "t_emb_grp", "cf_null_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#e65100;endArrow=classic;", [(500, 140), (820, 140), (820, 450)])

    # Standard Pass Label
    add_node("std_pass_lbl", "<font color='#1b5e20'><b>Standard Pass</b><br>Original Features (V_orig)</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;", p2_start_x, 250, 250, 40)
    add_edge("e_vau_std", "vau_box", "std_pass_lbl", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#1b5e20;endArrow=classic;", [(head_x+180, head_y_au+20), (700, head_y_au+20), (700, 270)])
    
    # CF Pass Label & Nulling Box
    add_node("cf_pass_lbl", "<font color='#4a148c'><b>Counterfactual Pass</b><br>Intervened Features (V_counter)</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;", p2_start_x, 400, 250, 40)
    add_edge("e_vau_cf", "vau_box", "cf_null_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#4a148c;endArrow=classic;", [(head_x+180, head_y_au+80), (700, head_y_au+80), (700, 520)])

    # Nulling Box (Idea 1)
    add_node("cf_null_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#1565c0;strokeWidth=2;", p2_start_x, 450, 350, 180)
    add_node("cf_null_title", "Semantic Subspace Nulling (Idea 1)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=12;fontColor=#1565c0;", p2_start_x, 460, 350, 20)
    
    # Geometric projection inside Nulling Box
    ox2, oy2 = p2_start_x + 100, 450 + 130
    add_node("pt_o2", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2, oy2, 1, 1)
    add_node("pt_tdir", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2+150, oy2, 1, 1)
    add_edge("v_tdir", "pt_o2", "pt_tdir", "endArrow=classic;html=1;strokeWidth=2;strokeColor=#e65100;dashed=1;")
    add_node("tdir_lbl", "T_ortho", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=12;fontColor=#e65100;fontStyle=1;", ox2+80, oy2+5, 80, 20)
    
    add_node("pt_v", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2+100, oy2-80, 1, 1)
    add_edge("v_vorig", "pt_o2", "pt_v", "endArrow=classic;html=1;strokeWidth=3;strokeColor=#1b5e20;")
    add_node("vorig_lbl", "V_orig", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontColor=#1b5e20;fontStyle=1;", ox2+100, oy2-100, 50, 20)
    
    add_node("pt_proj", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2+100, oy2, 1, 1)
    add_edge("v_projline", "pt_v", "pt_proj", "endArrow=none;html=1;strokeWidth=2;strokeColor=#9e9e9e;dashed=1;")
    
    add_node("pt_vortho", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2, oy2-80, 1, 1)
    add_edge("v_vortho", "pt_o2", "pt_vortho", "endArrow=classic;html=1;strokeWidth=4;strokeColor=#4a148c;")
    add_node("vortho_lbl", "V_counter", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontColor=#4a148c;fontStyle=1;", ox2-60, oy2-100, 80, 20)
    add_node("eq_null", "V_counter = V - (V·T)T", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontColor=#4a148c;fontStyle=1;", p2_start_x+180, 480, 150, 20)

    # -------------------------------------------------------------
    # SHARED CAUSAL GRAPH MODULES
    # -------------------------------------------------------------
    sg_x = 1180
    sg_y = 200
    add_node("sg_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fafafa;strokeColor=#37474f;strokeWidth=4;", sg_x, sg_y, 450, 950)
    add_node("sg_title", "<font style='font-size:18px' color='#263238'><b>Shared Causal Graph Modules (Idea 2)</b></font>", "text;html=1;strokeColor=none;fillColor=none;align=center;", sg_x, sg_y+15, 450, 30)

    # We draw the Level 1 and Level 2 graphs inside, in the center.
    g_cx = sg_x + 225

    # LEVEL 1 GRAPH (Centered)
    l1_y = sg_y + 150
    add_node("l1_lbl", "Level 1: AU → AU Graph", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", g_cx-100, l1_y-30, 200, 20)
    g1_r = 50
    g1_nodes = []
    for i in range(4):
        nid = f"sg1_n{i}"
        g1_nodes.append(nid)
        ang = i * math.pi / 2 - math.pi/4
        nx = g_cx + g1_r * math.cos(ang) - 20
        ny = l1_y + g1_r * math.sin(ang) - 20
        add_node(nid, f"AU{i+1}", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#c8e6c9;strokeColor=#388e3c;strokeWidth=2;fontStyle=1;fontSize=12;", nx, ny, 40, 40)
    add_edge("e_sg1_01", g1_nodes[0], g1_nodes[1], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor=#1b5e20;endArrow=block;")
    add_edge("e_sg1_02", g1_nodes[0], g1_nodes[2], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor=#1b5e20;endArrow=block;")
    add_edge("e_sg1_13", g1_nodes[1], g1_nodes[3], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor=#1b5e20;endArrow=block;")
    add_edge("e_sg1_23", g1_nodes[2], g1_nodes[3], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor=#1b5e20;endArrow=block;")

    # MATRICES (Centered)
    mat_y = sg_y + 350
    add_node("mat_lbl", "Adjacency Formulation", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;", g_cx-100, mat_y-20, 200, 20)
    
    add_node("g_inv_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=1;", g_cx-70, mat_y, 50, 50)
    for i in range(3):
        for j in range(3):
            fc = "#fff176" if (i*3+j)%4==0 else "#ffffff"
            add_node(f"sginv_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#ffd54f;", g_cx-70 + j*16.6, mat_y + i*16.6, 16.6, 16.6)
    add_node("sginv_lbl", "G_inv", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontStyle=1;", g_cx-70, mat_y+55, 50, 20)
    
    add_node("op_mul", "⊙", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=24;fontStyle=1;", g_cx-10, mat_y+10, 20, 20)

    add_node("g_dyn_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#e3f2fd;strokeColor=#1e88e5;strokeWidth=1;", g_cx+20, mat_y, 50, 50)
    for i in range(3):
        for j in range(3):
            fc = "#64b5f6" if (i*3+j)%5==0 else "#e3f2fd"
            add_node(f"sgdyn_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#90caf9;", g_cx+20 + j*16.6, mat_y + i*16.6, 16.6, 16.6)
    add_node("sgdyn_lbl", "G_dyn", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontStyle=1;", g_cx+20, mat_y+55, 50, 20)

    # LEVEL 2 GRAPH (Centered)
    l2_y = sg_y + 550
    add_node("l2_lbl", "Level 2: AU → Expr Graph", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", g_cx-100, l2_y-30, 200, 20)
    b_cx1 = g_cx - 40
    b_cx2 = g_cx + 40
    b_nodes_a = []
    b_nodes_e = []
    for i in range(3):
        nid_a = f"sg2_a{i}"
        b_nodes_a.append(nid_a)
        add_node(nid_a, f"AU{i+1}", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#c8e6c9;strokeColor=#388e3c;strokeWidth=2;fontSize=10;fontStyle=1;", b_cx1, l2_y + i*40, 30, 30)
        
        nid_e = f"sg2_e{i}"
        b_nodes_e.append(nid_e)
        add_node(nid_e, f"E{i+1}", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffcdd2;strokeColor=#c62828;strokeWidth=2;fontSize=10;fontStyle=1;", b_cx2, l2_y + i*40, 30, 30)
        
    for i in range(3):
        for j in range(3):
            if (i+j)%2 == 0 or i==j:
                add_edge(f"e_sg2_{i}{j}", b_nodes_a[i], b_nodes_e[j], "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=1;strokeColor=#455a64;endArrow=block;")


    # CONNECT THE PARALLEL PASSES THROUGH THE BOX
    # Standard Pass (Top half routing)
    std_route_y = sg_y + 200
    add_edge("in_std", "std_pass_lbl", "sg_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#1b5e20;endArrow=block;", [(1000, 270), (1000, std_route_y)])
    add_node("std_marker", "Standard Route", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#1b5e20;", sg_x+20, std_route_y-30, 120, 20)
    
    # Let's draw horizontal dashed arrows going across the box for Standard
    add_node("p_std_in", "", "shape=waypoint;fillColor=none;strokeColor=none;", sg_x, std_route_y, 1, 1)
    add_node("p_std_out", "", "shape=waypoint;fillColor=none;strokeColor=none;", sg_x+450, std_route_y, 1, 1)
    add_edge("e_std_flow", "p_std_in", "p_std_out", "endArrow=classic;html=1;strokeWidth=3;strokeColor=#1b5e20;dashed=1;")
    
    # Connect V_ex to Level 2 standard route
    add_edge("e_vex_std", "vex_box", "sg_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=classic;dashed=1;", [(head_x+180, head_y_ex+20), (600, head_y_ex+20), (600, std_route_y+50), (sg_x, std_route_y+50)])

    # Output Classifiers for Standard Pass
    cls_std_x = sg_x + 500
    add_node("cls_std_au", "Graph Classifier\n(Standard)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#2e7d32;strokeWidth=2;", cls_std_x, std_route_y-40, 120, 40)
    add_edge("e_std_out_au", "p_std_out", "cls_std_au", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=block;")
    
    add_node("pa2_out", "P_a^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor=#2e7d32;", cls_std_x+150, std_route_y-30, 60, 20)
    add_edge("e_cls_pa2", "cls_std_au", "pa2_out", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;")

    add_node("cls_std_ex", "Graph Classifier\n(Standard)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#c62828;strokeWidth=2;", cls_std_x, std_route_y+40, 120, 40)
    add_edge("e_std_out_ex", "p_std_out", "cls_std_ex", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=block;", [(sg_x+470, std_route_y), (sg_x+470, std_route_y+60)])
    
    add_node("pe2_out", "P_e^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor=#c62828;", cls_std_x+150, std_route_y+50, 60, 20)
    add_edge("e_cls_pe2", "cls_std_ex", "pe2_out", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;")


    # Counterfactual Pass (Bottom half routing)
    cf_route_y = sg_y + 800
    add_edge("in_cf", "cf_null_box", "sg_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#4a148c;endArrow=block;", [(1100, 540), (1130, 540), (1130, cf_route_y)])
    add_node("cf_marker", "Counterfactual Route", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#4a148c;", sg_x+20, cf_route_y-30, 160, 20)
    
    add_node("p_cf_in", "", "shape=waypoint;fillColor=none;strokeColor=none;", sg_x, cf_route_y, 1, 1)
    add_node("p_cf_out", "", "shape=waypoint;fillColor=none;strokeColor=none;", sg_x+450, cf_route_y, 1, 1)
    add_edge("e_cf_flow", "p_cf_in", "p_cf_out", "endArrow=classic;html=1;strokeWidth=3;strokeColor=#4a148c;dashed=1;")
    
    # Connect V_ex to Level 2 CF route
    add_edge("e_vex_cf", "vex_box", "sg_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=classic;dashed=1;", [(head_x+180, head_y_ex+60), (620, head_y_ex+60), (620, cf_route_y+50), (sg_x, cf_route_y+50)])

    # Output Classifiers for CF Pass
    add_node("cls_cf_au", "Graph Classifier\n(Shared Weights)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#4a148c;strokeWidth=2;", cls_std_x, cf_route_y-40, 120, 40)
    add_edge("e_cf_out_au", "p_cf_out", "cls_cf_au", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=block;")
    
    add_node("pacf_out", "P_a^(CF)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor=#4a148c;", cls_std_x+150, cf_route_y-30, 60, 20)
    add_edge("e_cls_pacf", "cls_cf_au", "pacf_out", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#4a148c;endArrow=block;")


    # -------------------------------------------------------------
    # COUNTERFACTUAL LOSS (Phase 2 Loss)
    # -------------------------------------------------------------
    lcf_x = 1850
    add_node("lcf_box", "Counterfactual\nConsistency Loss\n(L_CF)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#7b1fa2;strokeWidth=3;fontStyle=1;", lcf_x, (std_route_y+cf_route_y)/2 - 40, 160, 80)
    
    # Compare P_a_2 and P_a_CF
    add_edge("e_pa2_lcf", "pa2_out", "lcf_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#7b1fa2;endArrow=block;dashed=1;", [(cls_std_x+220, std_route_y-20), (lcf_x-20, std_route_y-20), (lcf_x-20, (std_route_y+cf_route_y)/2 - 20)])
    add_edge("e_pacf_lcf", "pacf_out", "lcf_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#7b1fa2;endArrow=block;dashed=1;", [(cls_std_x+220, cf_route_y-20), (lcf_x-20, cf_route_y-20), (lcf_x-20, (std_route_y+cf_route_y)/2 + 20)])


    # -------------------------------------------------------------
    # PHASE 3: CYCLE CONSISTENCY & ABDUCTIVE INFERENCE
    # -------------------------------------------------------------
    cyc_x = 2100
    cyc_y = 250
    add_node("idea4_lbl", "<font color='#7b1fa2'><b>Idea 4: Causal Cycle Consistency</b></font>", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;", cyc_x, cyc_y-40, 400, 30)
    add_node("cyc_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fafafa;strokeColor=#7b1fa2;strokeWidth=3;dashed=1;", cyc_x, cyc_y, 420, 200)
    
    # Route P_e_2 and P_a_2 to Cycle Consistency
    add_edge("e_pe2_cyc", "pe2_out", "cyc_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;dashed=1;", [(cls_std_x+250, std_route_y+60), (2000, std_route_y+60), (2000, cyc_y+100)])
    add_edge("e_pa2_cyc", "pa2_out", "cyc_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;dashed=1;", [(cls_std_x+250, std_route_y-20), (2050, std_route_y-20), (2050, cyc_y+100)])

    # Bar charts for distributions
    add_node("cyc_p_ex", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;", cyc_x+30, cyc_y+60, 80, 100)
    for i, h in enumerate([50, 90, 30, 60]):
        add_node(f"cb_e{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef5350;strokeColor=#b71c1c;", cyc_x+35+i*15, cyc_y+150-h, 12, h)
    add_node("cb_e_lbl", "P_e^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=14;", cyc_x+30, cyc_y+155, 80, 20)

    add_node("cyc_mul", "×", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=36;fontStyle=1;", cyc_x+115, cyc_y+90, 30, 30)

    mx_x = cyc_x + 160
    mx_y = cyc_y + 50
    add_node("mae_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#8e24aa;strokeWidth=2;", mx_x, mx_y, 80, 100)
    for i in range(5):
        for j in range(4):
            fc = "#ce93d8" if (i*2+j)%3==0 else "#f3e5f5"
            add_node(f"mae_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#ab47bc;", mx_x + j*20, mx_y + i*20, 20, 20)
    add_node("mae_lbl", "M_AEᵀ Prior Matrix", "text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=14;fontColor=#8e24aa;", mx_x-20, mx_y+160, 120, 20)

    add_node("cyc_eq", "≈", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=36;fontStyle=1;", mx_x+90, cyc_y+90, 30, 30)

    add_node("cyc_p_au", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;", cyc_x+280, cyc_y+60, 80, 100)
    for i, h in enumerate([70, 40, 95, 50]):
        add_node(f"cb_a{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#66bb6a;strokeColor=#2e7d32;", cyc_x+285+i*15, cyc_y+150-h, 12, h)
    add_node("cb_a_lbl", "P_a^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=14;", cyc_x+280, cyc_y+155, 80, 20)


    # ABDUCTIVE INFERENCE
    abd_x = 2100
    abd_y = 650
    add_node("idea3_lbl", "<font color='#ff6f00'><b>Idea 3: Test-Time Abductive Inference</b></font>", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;", abd_x, abd_y-40, 350, 30)
    add_node("abd_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#ff8f00;strokeWidth=4;", abd_x, abd_y, 350, 350)
    add_node("abd_lbl", "Energy Minimization", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor=#e65100;", abd_x, abd_y+15, 350, 30)
    
    add_node("loop_bg", "", "ellipse;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#ffca28;strokeWidth=5;dashed=1;", abd_x+100, abd_y+70, 150, 150)
    add_edge("e_loop1", "loop_bg", "loop_bg", "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=5;strokeColor=#ff8f00;endArrow=block;", [(abd_x+100, abd_y+145), (abd_x+130, abd_y+80), (abd_x+175, abd_y+70)])
    
    add_node("loop_txt", "Adam Optimizer\n(15 iterations)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontStyle=1;fontSize=14;", abd_x+105, abd_y+125, 140, 40)
    
    add_node("abd_eq", "min <font color='#1565c0'>E(FACS)</font> + <font color='#7b1fa2'>E(Cycle)</font> + <font color='#2e7d32'>E(Prior)</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", abd_x+25, abd_y+250, 300, 30)
    add_node("abd_desc", "Adjusts logits at test-time to\nsatisfy symbolic constraints", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;", abd_x+25, abd_y+290, 300, 40)
    
    add_node("final_pred_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e0f7fa;strokeColor=#006064;strokeWidth=3;", abd_x+75, abd_y+420, 200, 140)
    add_node("final_lbl", "Final Refined\nPredictions", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;", abd_x+75, abd_y+430, 200, 40)
    
    for i, h in enumerate([30, 60, 90, 50]):
        add_node(f"fp_a{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#66bb6a;strokeColor=#2e7d32;", abd_x+100+i*15, abd_y+540-h, 12, h)
    for i, h in enumerate([80, 40, 30, 70]):
        add_node(f"fp_e{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef5350;strokeColor=#b71c1c;", abd_x+180+i*15, abd_y+540-h, 12, h)

    add_edge("e_abd_final", "abd_box", "final_pred_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#006064;endArrow=block;")

    add_edge("e_pa2_abd", "pa2_out", "abd_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;dashed=1;", [(cls_std_x+230, std_route_y), (2000, std_route_y), (2000, abd_y+100)])
    add_edge("e_pe2_abd", "pe2_out", "abd_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;dashed=1;", [(cls_std_x+230, std_route_y+80), (2020, std_route_y+80), (2020, abd_y+150)])


    # LOSSES
    add_node("p1_loss_box", "Phase 1 Losses:\nL_WA, L_Expr, L_contrastive, L_HSIC, L_FACS", "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;darkOpacity=0.05;fillColor=#ffffff;strokeColor=#999999;size=15;fontSize=12;fontStyle=1;", 50, 1150, 300, 60)
    add_edge("e_l1", "pa1_out", "p1_loss_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#9e9e9e;endArrow=none;dashed=1;", [(head_x+280, head_y_au+170), (head_x+280, 1100), (200, 1100)])
    
    xml.append('      </root>')
    xml.append('    </mxGraphModel>')
    xml.append('  </diagram>')
    xml.append('</mxfile>')

    with open(r"C:\Users\khang\OneDrive\Desktop\ctrlau\ctrlau_architecture.drawio", "w", encoding="utf-8") as f:
        f.write("\n".join(xml))

if __name__ == "__main__":
    build()
