import math

def build():
    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append('<mxfile version="14.6.11">')
    xml.append('  <diagram id="ctrlau_v9" name="CtrlAU Architecture">')
    xml.append('    <mxGraphModel dx="2000" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2800" pageHeight="1800" math="0" shadow="0">')
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

    # Palette:
    C_GREEN = "#8ea66b"
    C_GREEN_DARK = "#5c6b44"
    C_PINK = "#ffdcdc"
    C_RED = "#d8a2a2"
    C_RED_DARK = "#9c7575"
    C_YELLOW = "#fff9d6"
    C_YELLOW_DARK = "#b3ae96"
    C_GREY = "#616161"

    # =========================================================
    # THE 3 MODULE BOUNDING BOXES (To avoid sequential Phase contradiction)
    # =========================================================
    add_node("bg_m1", "Module 1: Feature Extraction & Alignment", f"rounded=1;whiteSpace=wrap;html=1;fillColor={C_YELLOW};strokeColor={C_YELLOW_DARK};strokeWidth=3;dashed=1;verticalAlign=top;fontSize=24;fontStyle=1;fontColor={C_GREY};opacity=60;", 20, 40, 750, 1450)
    
    add_node("bg_m2", "Module 2: Parallel Causal Graph Routing & Intervention", f"rounded=1;whiteSpace=wrap;html=1;fillColor={C_PINK};strokeColor={C_RED_DARK};strokeWidth=3;dashed=1;verticalAlign=top;fontSize=24;fontStyle=1;fontColor={C_GREY};opacity=60;", 800, 40, 1250, 1450)
    
    add_node("bg_m3", "Module 3: End-to-End Test-Time Inference", f"rounded=1;whiteSpace=wrap;html=1;fillColor={C_YELLOW};strokeColor={C_YELLOW_DARK};strokeWidth=3;dashed=1;verticalAlign=top;fontSize=24;fontStyle=1;fontColor={C_GREY};opacity=60;", 2080, 40, 700, 1450)


    # =========================================================
    # MODULE 1
    # =========================================================
    img_x, img_y = 100, 150
    add_node("img_p2", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#424242;strokeWidth=2;", img_x+10, img_y+10, 100, 100)
    add_node("img_p1", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#eceff1;strokeColor=#424242;strokeWidth=2;", img_x, img_y, 100, 100)
    add_node("img_face", "<font style='font-size:48px'>👤</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;", img_x+10, img_y+10, 80, 80)
    
    rn_y = 400
    add_node("cnn_trap", "ResNet-50", f"shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;direction=south;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=2;fontSize=16;fontStyle=1;fontColor={C_GREY};", img_x+10, rn_y, 80, 140)
    add_edge("e_img_cnn", "img_p1", "cnn_trap", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_GREY};endArrow=block;")

    z_y = 650
    add_node("z_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};", img_x+10, z_y, 80, 80)
    for i in range(4):
        for j in range(4):
            fc = "#eeeeee" if (i+j)%2==0 else "#ffffff"
            add_node(f"z_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor={C_GREY};", img_x+10 + j*20, z_y + i*20, 20, 20)
    add_node("z_lbl", "Visual Features Z", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor={C_GREY};", img_x-10, z_y+90, 120, 30)
    add_edge("e_cnn_z", "cnn_trap", "z_bg", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_GREY};endArrow=block;")

    head_x = 350
    au_hy = 350
    add_node("au_head_lbl", "SymAU Head", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=18;fontColor={C_GREEN_DARK};", head_x, au_hy-50, 150, 30)
    au_nodes = []
    for i in range(4):
        nid = f"au_branch_{i}"
        au_nodes.append(nid)
        lbl = f"AU{i+1}" if i < 3 else "..."
        add_node(nid, lbl, f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={C_GREEN};strokeColor={C_GREEN_DARK};strokeWidth=2;fontStyle=1;fontColor=#ffffff;", head_x+50, au_hy + i*50, 40, 40)
        add_edge(f"e_z_{nid}", "z_bg", nid, f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor={C_GREEN_DARK};endArrow=block;", [(head_x-60, z_y+40), (head_x-30, au_hy+20+i*50)])

    add_node("vau_box", "", f"rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor={C_GREEN_DARK};strokeWidth=2;dashed=1;", head_x+150, au_hy-10, 60, 220)
    for i in range(4):
        add_node(f"v_au_vis_{i}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={C_GREEN};strokeColor={C_GREEN_DARK};", head_x+160, au_hy + i*50 + 5, 40, 15)
        add_edge(f"e_aub_{i}", au_nodes[i], f"v_au_vis_{i}", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor={C_GREEN_DARK};endArrow=block;")
    add_node("vau_lbl", "V<sub>a</sub>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_GREEN_DARK};", head_x+150, au_hy+220, 60, 30)

    ex_hy = 750
    add_node("ex_head_lbl", "SymExpr Head", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=18;fontColor={C_RED_DARK};", head_x, ex_hy-50, 150, 30)
    ex_nodes = []
    for i in range(4):
        nid = f"ex_branch_{i}"
        ex_nodes.append(nid)
        lbl = ["Anger","Fear","Happy","..."][i]
        add_node(nid, lbl, f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={C_RED};strokeColor={C_RED_DARK};strokeWidth=2;fontSize=10;fontStyle=1;fontColor=#ffffff;", head_x+50, ex_hy + i*50, 40, 40)
        add_edge(f"e_z_{nid}", "z_bg", nid, f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=2;strokeColor={C_RED_DARK};endArrow=block;", [(head_x-60, z_y+40), (head_x-30, ex_hy+20+i*50)])

    add_node("vex_box", "", f"rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor={C_RED_DARK};strokeWidth=2;dashed=1;", head_x+150, ex_hy-10, 60, 220)
    for i in range(4):
        add_node(f"v_ex_vis_{i}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={C_RED};strokeColor={C_RED_DARK};", head_x+160, ex_hy + i*50 + 5, 40, 15)
        add_edge(f"e_exb_{i}", ex_nodes[i], f"v_ex_vis_{i}", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor={C_RED_DARK};endArrow=block;")
    add_node("vex_lbl", "V<sub>e</sub>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_RED_DARK};", head_x+150, ex_hy+220, 60, 30)

    pa1_x = head_x + 280
    add_node("pa1_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREEN_DARK};strokeWidth=2;", pa1_x, au_hy+60, 100, 100)
    for i, h in enumerate([40, 80, 30, 60]):
        add_node(f"pa1_b_{i}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={C_GREEN};strokeColor={C_GREEN_DARK};", pa1_x+10+i*20, au_hy+150-h, 15, h)
    add_node("pa1_lbl", "P<sub>a</sub><sup>(1)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_GREEN_DARK};", pa1_x, au_hy+170, 100, 30)
    add_edge("e_va_pa1", "vau_box", "pa1_bg", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_GREEN_DARK};endArrow=classic;")

    add_node("pe1_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_RED_DARK};strokeWidth=2;", pa1_x, ex_hy+60, 100, 100)
    for i, h in enumerate([60, 30, 80, 50]):
        add_node(f"pe1_b_{i}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={C_RED};strokeColor={C_RED_DARK};", pa1_x+10+i*20, ex_hy+150-h, 15, h)
    add_node("pe1_lbl", "P<sub>e</sub><sup>(1)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_RED_DARK};", pa1_x, ex_hy+170, 100, 30)
    add_edge("e_ve_pe1", "vex_box", "pe1_bg", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_RED_DARK};endArrow=classic;")

    t_x, t_y = head_x + 150, 1250
    add_node("text_doc", "AU / Expr\nDescriptions", f"shape=document;whiteSpace=wrap;html=1;boundedLbl=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=2;fontSize=14;fontStyle=1;fontColor={C_GREY};", t_x-100, t_y, 120, 80)
    add_node("clip_enc", "Frozen CLIP\nText Encoder", f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=2;fontStyle=1;fontSize=14;fontColor={C_GREY};", t_x+80, t_y+10, 140, 60)
    add_edge("e_doc_clip", "text_doc", "clip_enc", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_GREY};endArrow=block;")
    
    add_node("t_box", "", f"rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor={C_GREY};strokeWidth=2;dashed=1;", t_x+280, t_y, 60, 220)
    for i in range(4):
        add_node(f"t_vec_{i}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#eeeeee;strokeColor={C_GREY};", t_x+290, t_y + i*50 + 5, 40, 15)
    add_node("t_lbl", "T", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_GREY};", t_x+280, t_y+220, 60, 30)
    add_edge("e_clip_t", "clip_enc", "t_box", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_GREY};endArrow=block;")

    add_node("l_contra", "L<sub>contrastive</sub>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_GREY};", pa1_x-10, 1150, 120, 40)
    add_edge("e_va_contra", "vau_box", "l_contra", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_GREY};endArrow=none;startArrow=none;dashed=1;", [(head_x+180, 1170)])
    add_edge("e_t_contra", "t_box", "l_contra", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_GREY};endArrow=none;startArrow=none;dashed=1;", [(t_x+310, 1170)])

    # =========================================================
    # MODULE 2 (Central routing)
    # =========================================================
    add_node("lgd_std", "— Standard Forward Pass", f"text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;fontStyle=1;fontColor={C_GREEN_DARK};", 850, 60, 300, 30)
    add_node("lgd_cf", "— Counterfactual Pass", f"text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;fontStyle=1;fontColor={C_RED_DARK};", 850, 100, 300, 30)
    add_node("lgd_struct", "— Structural Constraints", f"text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;fontStyle=1;fontColor={C_GREY};", 850, 140, 300, 30)
    add_edge("e_lgd_struct", "lgd_struct", "lgd_struct", f"endArrow=classic;html=1;strokeWidth=2;strokeColor={C_GREY};dashed=1;", [(850, 155), (820, 155)])

    # Semantic Nulling
    null_x, null_y = 900, 400
    add_node("null_box", "Semantic Subspace Nulling", f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=2;verticalAlign=top;fontStyle=1;fontSize=16;fontColor={C_GREY};", null_x, null_y, 250, 200)
    ox2, oy2 = null_x + 60, null_y + 160
    add_node("pt_o2", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2, oy2, 1, 1)
    add_node("pt_tdir", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2+120, oy2, 1, 1)
    add_edge("v_tdir", "pt_o2", "pt_tdir", f"endArrow=classic;html=1;strokeWidth=2;strokeColor={C_GREY};dashed=1;")
    add_node("tdir_lbl", "T", f"text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=14;fontColor={C_GREY};fontStyle=1;", ox2+60, oy2+5, 40, 20)
    add_node("pt_v", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2+80, oy2-80, 1, 1)
    add_edge("v_vorig", "pt_o2", "pt_v", f"endArrow=classic;html=1;strokeWidth=3;strokeColor={C_GREEN_DARK};")
    add_node("vorig_lbl", "V<sub>a</sub>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontColor={C_GREEN_DARK};fontStyle=1;", ox2+80, oy2-100, 50, 20)
    add_node("pt_proj", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2+80, oy2, 1, 1)
    add_edge("v_projline", "pt_v", "pt_proj", f"endArrow=none;html=1;strokeWidth=2;strokeColor={C_GREY};dashed=1;")
    add_node("pt_vortho", "", "shape=waypoint;fillColor=none;strokeColor=none;", ox2, oy2-80, 1, 1)
    add_edge("v_vortho", "pt_o2", "pt_vortho", f"endArrow=classic;html=1;strokeWidth=4;strokeColor={C_RED_DARK};")
    add_node("vortho_lbl", "V<sub>counter</sub>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontColor={C_RED_DARK};fontStyle=1;", ox2-60, oy2-100, 80, 20)
    
    add_node("vc_box", "V<sub>counter</sub>", f"rounded=1;whiteSpace=wrap;html=1;fillColor={C_RED};strokeColor={C_RED_DARK};strokeWidth=2;fontStyle=1;fontSize=16;fontColor=#ffffff;", null_x+75, null_y+220, 100, 40)


    # =========================================================
    # THE CENTRAL SHARED CAUSAL GRAPH MODULE 
    # =========================================================
    g_cx = 1450
    # Level 1 Bounding Box
    add_node("l1_box", "Level 1: AU-AU Causal Graph", f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREEN_DARK};strokeWidth=3;dashed=1;verticalAlign=top;fontStyle=1;fontSize=18;fontColor={C_GREEN_DARK};", g_cx-250, 480, 500, 200)
    
    # Level 2 Bounding Box
    add_node("l2_box", "Level 2: AU-Expression Bipartite Graph", f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_RED_DARK};strokeWidth=3;dashed=1;verticalAlign=top;fontStyle=1;fontSize=18;fontColor={C_RED_DARK};", g_cx-250, 1180, 500, 250)


    # 1. Adjacency Matrices (Y=250)
    mat_y = 250
    add_node("mat_lbl", "Adjacency Matrix Formulation", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_GREY};", g_cx-180, mat_y, 360, 30)
    add_node("g_inv_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=1;", g_cx-120, mat_y+40, 80, 80)
    for i in range(4):
        for j in range(4):
            fc = "#eeeeee" if (i*4+j)%5==0 else "#ffffff"
            add_node(f"sginv_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor={C_GREY};", g_cx-120 + j*20, mat_y+40 + i*20, 20, 20)
    add_node("sginv_lbl", "G<sub>inv</sub>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor={C_GREY};", g_cx-120, mat_y+130, 80, 20)
    add_node("op_mul", "⊙", f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=36;fontStyle=1;fontColor={C_GREY};", g_cx-30, mat_y+60, 40, 40)
    add_node("g_dyn_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=1;", g_cx+20, mat_y+40, 80, 80)
    for i in range(4):
        for j in range(4):
            fc = "#eeeeee" if (i*4+j)%3==0 else "#ffffff"
            add_node(f"sgdyn_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor={C_GREY};", g_cx+20 + j*20, mat_y+40 + i*20, 20, 20)
    add_node("sgdyn_lbl", "G<sub>dyn</sub>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor={C_GREY};", g_cx+20, mat_y+130, 80, 20)

    # Structure arrow
    add_edge("e_mat_l1", "op_mul", "l1_box", f"endArrow=classic;html=1;strokeWidth=3;strokeColor={C_GREY};dashed=1;", [(g_cx-10, mat_y+160), (g_cx-10, 480)])
    add_node("lbl_w1", "Defines Edge Weights", f"text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=12;fontStyle=1;fontColor={C_GREY};", g_cx, mat_y+180, 160, 20)

    # 2. Level 1 AU-AU Graph (Y=580)
    l1_y = 580
    g1_nodes = []
    g1_r = 60
    for i in range(4):
        ang = i * math.pi / 2 - math.pi/4
        nx = g_cx + g1_r * math.cos(ang) - 25
        ny = l1_y + g1_r * math.sin(ang) - 25
        g1_nodes.append((nx, ny))
        add_node(f"sgn1_{i}", f"AU{i+1}", f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={C_GREEN};strokeColor={C_GREEN_DARK};strokeWidth=2;fontStyle=1;fontColor=#ffffff;", nx, ny, 50, 50)
    add_edge("e_sg1_01", f"sgn1_0", f"sgn1_1", f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=3;strokeColor={C_GREEN_DARK};endArrow=block;")
    add_edge("e_sg1_02", f"sgn1_0", f"sgn1_2", f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=3;strokeColor={C_GREEN_DARK};endArrow=block;")
    add_edge("e_sg1_13", f"sgn1_1", f"sgn1_3", f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=3;strokeColor={C_GREEN_DARK};endArrow=block;")
    add_edge("e_sg1_23", f"sgn1_2", f"sgn1_3", f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=3;strokeColor={C_GREEN_DARK};endArrow=block;")

    # 3. Mask Module (Y=850)
    mask_y = 850
    add_node("mask_lbl", "Dependency Mask Module", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_GREY};", g_cx-160, mask_y, 320, 30)
    add_node("m_imp_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=1;", g_cx-90, mask_y+40, 60, 60)
    for i in range(3):
        for j in range(3):
            val = "1" if (i+j)%2==0 else "0"
            add_node(f"mimp_c_{i}_{j}", val, f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={C_GREY};fontSize=12;fontStyle=1;fontColor={C_GREY};", g_cx-90 + j*20, mask_y+40 + i*20, 20, 20)
    add_node("mimp_lbl", "Importance Mask\n(0 or 1)", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor={C_GREY};", g_cx-120, mask_y+110, 120, 30)
    add_node("m_pol_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=1;", g_cx+30, mask_y+40, 60, 60)
    for i in range(3):
        for j in range(3):
            val = "+1" if (i+j)%3==0 else "-1"
            add_node(f"mpol_c_{i}_{j}", val, f"rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={C_GREY};fontSize=12;fontStyle=1;fontColor={C_GREY};", g_cx+30 + j*20, mask_y+40 + i*20, 20, 20)
    add_node("mpol_lbl", "Polarity Mask\n(+1 or -1)", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor={C_GREY};", g_cx, mask_y+110, 120, 30)

    # Structure arrow
    add_edge("e_mask_l2", "mask_lbl", "l2_box", f"endArrow=classic;html=1;strokeWidth=3;strokeColor={C_GREY};dashed=1;", [(g_cx-10, mask_y+150), (g_cx-10, 1180)])
    add_node("lbl_w2", "Constrains Bipartite Edge Weights", f"text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=12;fontStyle=1;fontColor={C_GREY};", g_cx, mask_y+160, 200, 20)

    # 4. Level 2 AU-Expr Graph (Y=1300)
    l2_y = 1270
    for i in range(4):
        add_node(f"g2a_{i}", f"AU{i+1}", f"ellipse;whiteSpace=wrap;html=1;fillColor={C_GREEN};strokeColor={C_GREEN_DARK};fontStyle=1;fontColor=#ffffff;", g_cx-80, l2_y + i*40, 40, 30)
        add_node(f"g2e_{i}", f"E{i+1}", f"ellipse;whiteSpace=wrap;html=1;fillColor={C_RED};strokeColor={C_RED_DARK};fontStyle=1;fontColor=#ffffff;", g_cx+80, l2_y + i*40, 40, 30)
    for i in range(4):
        for j in range(4):
            if (i+j)%3 == 0 or i==j:
                add_edge(f"e_g2_{i}_{j}", f"g2a_{i}", f"g2e_{j}", f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=1;strokeColor={C_GREY};endArrow=block;")


    # =========================================================
    # DATA PATH ROUTING (Standard green & CF red)
    # =========================================================
    add_edge("e_va_g1_std", "vau_box", "sgn1_3", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor={C_GREEN_DARK};endArrow=classic;", [(head_x+260, 460), (head_x+260, 200), (1150, 200), (1150, l1_y-20), (g_cx-100, l1_y-20)])
    
    add_edge("e_va_null", "vau_box", "null_box", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor={C_RED_DARK};endArrow=classic;", [(head_x+230, 460), (head_x+230, 500), (null_x, 500)])
    add_edge("e_t_null", "t_box", "null_box", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_GREY};endArrow=classic;dashed=1;", [(t_x+350, 1360), (770, 1360), (770, 500)])
    add_edge("e_vc_l1_cf", "vc_box", "sgn1_3", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor={C_RED_DARK};endArrow=classic;", [(null_x+125, l1_y+70), (g_cx-100, l1_y+70)])

    add_node("l1_out_std", "V<sub>a</sub><sup>(2)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor={C_GREEN_DARK};", g_cx+140, l1_y-50, 80, 30)
    add_node("l1_out_cf", "V<sub>a</sub><sup>(CF)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor={C_RED_DARK};", g_cx+140, l1_y+70, 80, 30)

    add_edge("e_l1_l2_std", "l1_out_std", "g2a_0", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor={C_GREEN_DARK};endArrow=classic;", [(g_cx+180, l1_y-30), (1250, l1_y-30), (1250, l2_y-20), (g_cx-100, l2_y-20)])
    add_edge("e_l1_l2_cf", "l1_out_cf", "g2a_1", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor={C_RED_DARK};endArrow=classic;", [(g_cx+180, l1_y+90), (1200, l1_y+90), (1200, l2_y+20), (g_cx-100, l2_y+20)])
    add_edge("e_ve_l2", "vex_box", "g2e_3", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_RED_DARK};endArrow=classic;dashed=1;", [(head_x+260, 860), (g_cx+105, 860)])


    # =========================================================
    # CLASSIFIERS & LOSSES
    # =========================================================
    cls_x = 1750
    # Phase 2 Standard Outputs
    add_node("pa2_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREEN_DARK};strokeWidth=2;", cls_x, l1_y-70, 80, 80)
    for i, h in enumerate([40, 70, 30, 50]):
        add_node(f"pa2_b_{i}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={C_GREEN};strokeColor={C_GREEN_DARK};", cls_x+5+i*15, l1_y+10-h, 12, h)
    add_node("pa2_lbl", "P<sub>a</sub><sup>(2)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor={C_GREEN_DARK};", cls_x, l1_y+20, 80, 20)
    add_edge("e_l1_pa2", "sgn1_1", "pa2_bg", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor={C_GREEN_DARK};endArrow=classic;", [(g_cx+100, l1_y-30), (1700, l1_y-30)])

    # Phase 3 CF Outputs and Loss
    cls_cf_x = 1920
    add_node("pacf_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_RED_DARK};strokeWidth=2;", cls_cf_x, l1_y-70, 80, 80)
    for i, h in enumerate([70, 20, 90, 10]):
        add_node(f"pacf_b_{i}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={C_RED};strokeColor={C_RED_DARK};", cls_cf_x+5+i*15, l1_y+10-h, 12, h)
    add_node("pacf_lbl", "P<sub>a</sub><sup>(CF)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor={C_RED_DARK};", cls_cf_x, l1_y+20, 80, 20)
    add_edge("e_l1_pacf", "sgn1_1", "pacf_bg", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor={C_RED_DARK};endArrow=classic;", [(g_cx+100, l1_y+100), (1880, l1_y+100), (1880, l1_y-30)])

    add_node("lcf_au", "L<sub>CF_AU</sub>", f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_RED_DARK};strokeWidth=2;fontStyle=1;fontSize=16;fontColor={C_RED_DARK};", 1845, l1_y-150, 100, 50)
    add_edge("e_pa2_lcf", "pa2_bg", "lcf_au", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_RED_DARK};endArrow=none;dashed=1;", [(1790, l1_y-70), (1790, l1_y-125)])
    add_edge("e_pacf_lcf", "pacf_bg", "lcf_au", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_RED_DARK};endArrow=none;dashed=1;", [(1960, l1_y-70), (1960, l1_y-125)])

    # L2 Outputs
    add_node("pe2_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREEN_DARK};strokeWidth=2;", cls_x, l2_y-20, 80, 80)
    for i, h in enumerate([50, 20, 70, 40]):
        add_node(f"pe2_b_{i}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={C_GREEN};strokeColor={C_GREEN_DARK};", cls_x+5+i*15, l2_y+60-h, 12, h)
    add_node("pe2_lbl", "P<sub>e</sub><sup>(2)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor={C_GREEN_DARK};", cls_x, l2_y+70, 80, 20)
    add_edge("e_l2_pe2", "g2e_0", "pe2_bg", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor={C_GREEN_DARK};endArrow=classic;", [(g_cx+160, l2_y+20), (1700, l2_y+20)])

    add_node("pecf_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_RED_DARK};strokeWidth=2;", cls_cf_x, l2_y-20, 80, 80)
    for i, h in enumerate([80, 40, 20, 60]):
        add_node(f"pecf_b_{i}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={C_RED};strokeColor={C_RED_DARK};", cls_cf_x+5+i*15, l2_y+60-h, 12, h)
    add_node("pecf_lbl", "P<sub>e</sub><sup>(CF)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;fontColor={C_RED_DARK};", cls_cf_x, l2_y+70, 80, 20)
    add_edge("e_l2_pecf", "g2e_3", "pecf_bg", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor={C_RED_DARK};endArrow=classic;", [(g_cx+160, l2_y+150), (1880, l2_y+150), (1880, l2_y+20)])

    add_node("lcf_ex", "L<sub>CF_Expr</sub>", f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_RED_DARK};strokeWidth=2;fontStyle=1;fontSize=16;fontColor={C_RED_DARK};", 1845, l2_y-100, 100, 50)
    add_edge("e_pe2_lcf", "pe2_bg", "lcf_ex", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_RED_DARK};endArrow=none;dashed=1;", [(1790, l2_y-20), (1790, l2_y-75)])
    add_edge("e_pecf_lcf", "pecf_bg", "lcf_ex", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor={C_RED_DARK};endArrow=none;dashed=1;", [(1960, l2_y-20), (1960, l2_y-75)])


    # =========================================================
    # MODULE 3 (End-to-End Inference)
    # =========================================================
    c_x = 2150
    cyc_y = 350

    add_node("cyc_box", "Causal Cycle Consistency", f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=2;dashed=1;verticalAlign=top;fontStyle=1;fontSize=20;fontColor={C_GREY};", c_x, cyc_y, 550, 200)
    add_node("cyc_pe2", "P<sub>e</sub><sup>(2)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_RED_DARK};", c_x+40, cyc_y+60, 60, 30)
    add_node("cyc_mul", "×", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=36;fontStyle=1;fontColor={C_GREY};", c_x+130, cyc_y+90, 40, 40)
    add_node("mae_bg", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=2;fontStyle=1;", c_x+200, cyc_y+60, 80, 100)
    for i in range(5):
        for j in range(4):
            fc = "#eeeeee" if (i*2+j)%3==0 else "#ffffff"
            add_node(f"mae_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor={C_GREY};", c_x+200 + j*20, cyc_y+60 + i*20, 20, 20)
    add_node("mae_lbl", "M<sub>AE</sub><sup>T</sup> Prior Matrix", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontStyle=1;fontSize=14;fontColor={C_GREY};", c_x+180, cyc_y+170, 120, 20)
    add_node("cyc_eq", "≈", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=36;fontStyle=1;fontColor={C_GREY};", c_x+320, cyc_y+90, 40, 40)
    add_node("cyc_pa2", "P<sub>a</sub><sup>(2)</sup>", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_GREEN_DARK};", c_x+400, cyc_y+60, 60, 30)
    
    add_edge("e_pe2_cyc", "pe2_bg", "cyc_box", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor={C_RED_DARK};endArrow=block;dashed=1;", [(1830, l2_y+20), (2000, l2_y+20), (2000, cyc_y+100)])
    add_edge("e_pa2_cyc", "pa2_bg", "cyc_box", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor={C_GREEN_DARK};endArrow=block;dashed=1;", [(1830, l1_y-30), (2050, l1_y-30), (2050, cyc_y+150)])

    abd_y = 800
    add_node("abd_box", "Test-Time Abductive Inference", f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=4;verticalAlign=top;fontStyle=1;fontSize=20;fontColor={C_GREY};", c_x, abd_y, 450, 450)
    add_node("loop_bg", "", f"ellipse;whiteSpace=wrap;html=1;fillColor=none;strokeColor={C_GREY};strokeWidth=5;dashed=1;", c_x+135, abd_y+80, 180, 180)
    add_edge("e_loop", "loop_bg", "loop_bg", f"edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=5;strokeColor={C_GREY};endArrow=block;", [(c_x+135, abd_y+170), (c_x+180, abd_y+80), (c_x+225, abd_y+80)])
    add_node("loop_txt", "Adam Optimizer\n(15 iterations)", f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontStyle=1;fontSize=18;fontColor={C_GREY};", c_x+155, abd_y+150, 140, 40)
    add_node("abd_eq", "min Energy = E(FACS) + E(Cycle) + E(Prior)", f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=18;fontStyle=1;fontColor={C_GREY};", c_x+20, abd_y+300, 410, 30)
    add_node("final_pred_box", "Final Refined Predictions", f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={C_GREY};strokeWidth=3;fontStyle=1;fontSize=16;fontColor={C_GREY};", c_x+100, abd_y+360, 250, 70)
    
    add_edge("e_pa2_abd", "pa2_bg", "abd_box", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor={C_GREEN_DARK};endArrow=block;dashed=1;", [(1830, l1_y-10), (2070, l1_y-10), (2070, abd_y+150)])
    add_edge("e_pe2_abd", "pe2_bg", "abd_box", f"edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor={C_RED_DARK};endArrow=block;dashed=1;", [(1830, l2_y+40), (2090, l2_y+40), (2090, abd_y+200)])


    xml.append('      </root>')
    xml.append('    </mxGraphModel>')
    xml.append('  </diagram>')
    xml.append('</mxfile>')

    with open(r"C:\Users\khang\OneDrive\Desktop\ctrlau\ctrlau_architecture.drawio", "w", encoding="utf-8") as f:
        f.write("\n".join(xml))

if __name__ == "__main__":
    build()
