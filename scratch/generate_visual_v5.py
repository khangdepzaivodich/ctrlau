import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def build():
    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append('<mxfile version="14.6.11">')
    xml.append('  <diagram id="ctrlau_v5" name="CtrlAU Architecture">')
    xml.append('    <mxGraphModel dx="2000" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2200" pageHeight="1200" math="0" shadow="0">')
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

    # =========================================================
    # BACKGROUNDS (No ugly boxes, just subtle colored backgrounds)
    # =========================================================
    add_node("bg1", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=none;opacity=40;", 20, 20, 520, 1100)
    add_node("bg2", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f8e9;strokeColor=none;opacity=40;", 560, 20, 1100, 1100)
    add_node("bg3", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff8e1;strokeColor=none;opacity=40;", 1680, 20, 480, 1100)

    add_node("p1_lbl", "Phase 1: Feature Extraction & Alignment", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=20;fontStyle=1;fontColor=#343a40;", 20, 30, 520, 30)
    add_node("p2_lbl", "Phase 2: Parallel Causal Routing & Counterfactual Intervention", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=20;fontStyle=1;fontColor=#33691e;", 560, 30, 1100, 30)
    add_node("p3_lbl", "Phase 3: Cycle Consistency & Abductive Inference", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=20;fontStyle=1;fontColor=#ff6f00;", 1680, 30, 480, 30)

    # =========================================================
    # PHASE 1: COMPACT & SCIENTIFIC
    # =========================================================
    # Image
    img_x, img_y = 200, 100
    add_node("img_p2", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#424242;strokeWidth=2;", img_x+10, img_y+10, 80, 80)
    add_node("img_p1", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#eceff1;strokeColor=#424242;strokeWidth=2;", img_x, img_y, 80, 80)
    add_node("img_face", "<font style='font-size:36px'>👤</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;", img_x+10, img_y+10, 60, 60)
    
    # ResNet
    rn_y = 250
    add_node("resnet", "CNN\nEncoder", "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;direction=south;fillColor=#bbdefb;strokeColor=#1565c0;strokeWidth=2;fontSize=14;fontStyle=1;", img_x+10, rn_y, 60, 100)
    add_edge("e_img_res", "img_p1", "resnet", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#455a64;endArrow=block;")

    # Z Feature Map
    z_y = 400
    add_node("z_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#eeeeee;strokeColor=#757575;", img_x+10, z_y, 60, 60)
    for i in range(3):
        for j in range(3):
            fc = "#cfd8dc" if (i+j)%2==0 else "#ffffff"
            add_node(f"z_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#b0bec5;", img_x+10 + j*20, z_y + i*20, 20, 20)
    add_node("z_lbl", "Visual Z", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;", img_x, z_y+65, 80, 20)
    add_edge("e_res_z", "resnet", "z_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#455a64;endArrow=block;")

    # Split to Heads
    au_hx, au_hy = 100, 550
    ex_hx, ex_hy = 300, 550
    
    add_node("au_head", "SymAU Head\n(8 Branches)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#c8e6c9;strokeColor=#2e7d32;strokeWidth=2;fontStyle=1;", au_hx, au_hy, 120, 60)
    add_edge("e_z_au", "z_bg", "au_head", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;", [(img_x+40, 500), (au_hx+60, 500)])
    
    add_node("ex_head", "SymExpr Head\n(7 Branches)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffcdd2;strokeColor=#c62828;strokeWidth=2;fontStyle=1;", ex_hx, ex_hy, 120, 60)
    add_edge("e_z_ex", "z_bg", "ex_head", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;", [(img_x+40, 500), (ex_hx+60, 500)])

    # Embeddings
    v_y = 660
    add_node("va_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#1b5e20;strokeWidth=2;dashed=1;", au_hx+40, v_y, 40, 80)
    for i in range(3):
        add_node(f"va_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#a5d6a7;strokeColor=#1b5e20;", au_hx+45, v_y+10 + i*20, 30, 10)
    add_node("va_lbl", "V_a", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#1b5e20;", au_hx+40, v_y+85, 40, 20)
    add_edge("e_au_va", "au_head", "va_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=classic;")

    add_node("ve_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#b71c1c;strokeWidth=2;dashed=1;", ex_hx+40, v_y, 40, 80)
    for i in range(3):
        add_node(f"ve_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef9a9a;strokeColor=#b71c1c;", ex_hx+45, v_y+10 + i*20, 30, 10)
    add_node("ve_lbl", "V_e", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#b71c1c;", ex_hx+40, v_y+85, 40, 20)
    add_edge("e_ex_ve", "ex_head", "ve_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=classic;")

    # Probabilities
    p1_y = 820
    add_node("pa1_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#2e7d32;strokeWidth=2;", au_hx+20, p1_y, 80, 80)
    for i, h in enumerate([30, 60, 20, 50]):
        add_node(f"pa1_b_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#66bb6a;strokeColor=#2e7d32;", au_hx+25+i*15, p1_y+70-h, 12, h)
    add_node("pa1_lbl", "P_a^(1)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#2e7d32;", au_hx+20, p1_y+85, 80, 20)
    add_edge("e_va_pa1", "va_box", "pa1_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=classic;")

    add_node("pe1_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#c62828;strokeWidth=2;", ex_hx+20, p1_y, 80, 80)
    for i, h in enumerate([50, 20, 70, 40]):
        add_node(f"pe1_b_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef5350;strokeColor=#b71c1c;", ex_hx+25+i*15, p1_y+70-h, 12, h)
    add_node("pe1_lbl", "P_e^(1)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#c62828;", ex_hx+20, p1_y+85, 80, 20)
    add_edge("e_ve_pe1", "ve_box", "pe1_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=classic;")

    # Text & CLIP (Top right of Phase 1 to align with V_a)
    t_x, t_y = 400, 100
    add_node("text_in", "AU / Expr\nPrompts", "shape=document;whiteSpace=wrap;html=1;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=2;fontStyle=1;", t_x, t_y, 80, 60)
    add_node("clip_enc", "CLIP", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe082;strokeColor=#ff8f00;strokeWidth=2;fontStyle=1;", t_x, t_y+90, 80, 40)
    add_edge("e_text_clip", "text_in", "clip_enc", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#f57f17;endArrow=block;")
    
    add_node("t_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#e65100;strokeWidth=2;dashed=1;", t_x+20, t_y+160, 40, 80)
    for i in range(3):
        add_node(f"t_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffcc80;strokeColor=#e65100;", t_x+25, t_y+170 + i*20, 30, 10)
    add_node("t_lbl", "T", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#e65100;", t_x+20, t_y+245, 40, 20)
    add_edge("e_clip_t", "clip_enc", "t_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#f57f17;endArrow=block;")

    # Contrastive Alignment
    add_node("l_contra", "L_contrastive", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#1565c0;", 200, 750, 100, 30)
    add_edge("e_va_contra", "va_box", "l_contra", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#1565c0;endArrow=none;startArrow=none;dashed=1;", [(au_hx+80, 700), (250, 700), (250, 750)])
    add_edge("e_t_contra", "t_box", "l_contra", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#1565c0;endArrow=none;startArrow=none;dashed=1;", [(t_x+40, 240), (480, 240), (480, 765), (300, 765)])

    # =========================================================
    # PHASE 2: PARALLEL CAUSAL GRAPH (SINGLE ILLUSTRATION, 2 PATHS)
    # =========================================================
    # Legend for parallel paths
    add_node("lgd_std", "— Standard Forward Pass", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=14;fontStyle=1;fontColor=#1b5e20;", 600, 50, 200, 30)
    add_node("lgd_cf", "— Counterfactual Pass", "text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=14;fontStyle=1;fontColor=#4a148c;", 600, 80, 200, 30)

    # 1. Semantic Subspace Nulling (Only purple path goes through here)
    null_x, null_y = 650, 200
    add_node("null_box", "Semantic Nulling", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#1565c0;strokeWidth=2;verticalAlign=top;fontStyle=1;", null_x, null_y, 160, 120)
    add_node("null_eq", "V_counter = V - (V·T)T", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#1565c0;", null_x, null_y+40, 160, 20)
    add_node("vc_box", "V_counter", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1bee7;strokeColor=#4a148c;strokeWidth=2;fontStyle=1;", null_x+40, null_y+80, 80, 30)
    
    # Path splits from V_a
    # Green Path directly to Graph L1
    add_edge("e_va_g1_std", "va_box", "null_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=classic;", [(au_hx+40, 650), (520, 650), (520, 500), (950, 500)])
    
    # Purple Path to Nulling
    add_edge("e_va_null", "va_box", "null_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=classic;", [(au_hx+60, 650), (600, 650), (600, 260), (null_x, 260)])
    # Text to Nulling
    add_edge("e_t_null", "t_box", "null_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#e65100;endArrow=classic;dashed=1;", [(t_x+60, 200), (null_x, 200)])


    # THE CENTRAL SHARED CAUSAL GRAPH ILLUSTRATION
    g_cx = 1100

    # L1 Graph
    l1_y = 450
    add_node("l1_title", "Level 1: AU-AU Causal Graph", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;", g_cx-120, l1_y-100, 240, 30)
    g1_nodes = []
    for i in range(4):
        ang = i * math.pi / 2 - math.pi/4
        nx = g_cx + 50 * math.cos(ang) - 20
        ny = l1_y + 50 * math.sin(ang) - 20
        add_node(f"sgn1_{i}", f"AU{i+1}", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#c8e6c9;strokeColor=#388e3c;strokeWidth=2;", nx, ny, 40, 40)

    # Both paths enter L1 Graph!
    # Green Path from V_a
    add_node("pt_g1_std_in", "", "shape=waypoint;fillColor=none;strokeColor=none;", 950, l1_y-30, 1, 1)
    add_edge("e_in_l1_std", "pt_g1_std_in", "sgn1_3", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=classic;", [(g_cx-60, l1_y-30)])
    
    # Purple Path from V_counter
    add_node("pt_g1_cf_in", "", "shape=waypoint;fillColor=none;strokeColor=none;", 950, l1_y+30, 1, 1)
    add_edge("e_vc_l1_cf", "vc_box", "pt_g1_cf_in", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=none;", [(null_x+80, 255), (950, 255), (950, l1_y+30)])
    add_edge("e_in_l1_cf", "pt_g1_cf_in", "sgn1_3", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=classic;", [(g_cx-60, l1_y+30)])

    # Both paths exit L1 Graph!
    add_node("l1_out_std", "V_a^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#1b5e20;", g_cx+80, l1_y-40, 60, 20)
    add_node("l1_out_cf", "V_a^(CF)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#4a148c;", g_cx+80, l1_y+20, 60, 20)

    # Classifiers & Outputs for L1
    add_node("pa2_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#2e7d32;strokeWidth=2;", 1400, l1_y-60, 60, 60)
    add_node("pa2_lbl", "P_a^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#2e7d32;", 1400, l1_y+5, 60, 20)
    add_edge("e_l1_pa2", "sgn1_1", "pa2_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=classic;", [(g_cx+60, l1_y-30), (1350, l1_y-30)])

    add_node("pacf_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#4a148c;strokeWidth=2;", 1400, l1_y+30, 60, 60)
    add_node("pacf_lbl", "P_a^(CF)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#4a148c;", 1400, l1_y+95, 60, 20)
    add_edge("e_l1_pacf", "sgn1_1", "pacf_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=classic;", [(g_cx+60, l1_y+60), (1350, l1_y+60)])

    # L1 CF Loss
    add_node("lcf_au", "L_CF_AU", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#7b1fa2;strokeWidth=2;fontStyle=1;", 1520, l1_y, 80, 40)
    add_edge("e_pa2_lcf", "pa2_bg", "lcf_au", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#7b1fa2;endArrow=none;dashed=1;", [(1460, l1_y-30), (1490, l1_y-30), (1490, l1_y+10)])
    add_edge("e_pacf_lcf", "pacf_bg", "lcf_au", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#7b1fa2;endArrow=none;dashed=1;", [(1460, l1_y+60), (1490, l1_y+60), (1490, l1_y+30)])


    # L2 Graph (Bipartite)
    l2_y = 800
    add_node("l2_title", "Level 2: AU-Expr Causal Graph", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontStyle=1;", g_cx-120, l2_y-60, 240, 30)
    
    # Draw Bipartite nodes
    for i in range(3):
        add_node(f"g2a_{i}", f"AU{i+1}", "ellipse;whiteSpace=wrap;html=1;fillColor=#c8e6c9;strokeColor=#388e3c;", g_cx-40, l2_y + i*40, 40, 30)
        add_node(f"g2e_{i}", f"E{i+1}", "ellipse;whiteSpace=wrap;html=1;fillColor=#ffcdd2;strokeColor=#c62828;", g_cx+40, l2_y + i*40, 40, 30)

    # Both paths flow from L1 to L2!
    add_node("pt_g2_std_in", "", "shape=waypoint;fillColor=none;strokeColor=none;", 950, l2_y+10, 1, 1)
    add_edge("e_l1_l2_std", "l1_out_std", "pt_g2_std_in", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=none;", [(g_cx+120, l1_y-30), (950, l1_y-30), (950, l2_y+10)])
    add_edge("e_in_l2_std", "pt_g2_std_in", "g2a_0", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=classic;")
    
    add_node("pt_g2_cf_in", "", "shape=waypoint;fillColor=none;strokeColor=none;", 950, l2_y+70, 1, 1)
    add_edge("e_l1_l2_cf", "l1_out_cf", "pt_g2_cf_in", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=none;", [(g_cx+120, l1_y+30), (920, l1_y+30), (920, l2_y+70), (950, l2_y+70)])
    add_edge("e_in_l2_cf", "pt_g2_cf_in", "g2a_2", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=classic;")

    # V_e enters L2 (Shared for both passes conceptually)
    add_edge("e_ve_l2", "ve_box", "g2e_2", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=classic;dashed=1;", [(ex_hx+60, 740), (g_cx+60, 740)])

    # Classifiers & Outputs for L2
    add_node("pe2_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#c62828;strokeWidth=2;", 1400, l2_y-20, 60, 60)
    add_node("pe2_lbl", "P_e^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#c62828;", 1400, l2_y+45, 60, 20)
    add_edge("e_l2_pe2", "g2e_0", "pe2_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=classic;", [(g_cx+100, l2_y+10), (1350, l2_y+10)])

    add_node("pecf_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#4a148c;strokeWidth=2;", 1400, l2_y+70, 60, 60)
    add_node("pecf_lbl", "P_e^(CF)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#4a148c;", 1400, l2_y+135, 60, 20)
    add_edge("e_l2_pecf", "g2e_2", "pecf_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=classic;", [(g_cx+100, l2_y+100), (1350, l2_y+100)])

    # L2 CF Loss
    add_node("lcf_ex", "L_CF_Expr", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#7b1fa2;strokeWidth=2;fontStyle=1;", 1520, l2_y+40, 80, 40)
    add_edge("e_pe2_lcf", "pe2_bg", "lcf_ex", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#7b1fa2;endArrow=none;dashed=1;", [(1460, l2_y+10), (1490, l2_y+10), (1490, l2_y+50)])
    add_edge("e_pecf_lcf", "pecf_bg", "lcf_ex", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#7b1fa2;endArrow=none;dashed=1;", [(1460, l2_y+100), (1490, l2_y+100), (1490, l2_y+70)])


    # =========================================================
    # PHASE 3: CYCLE CONSISTENCY & ABDUCTIVE
    # =========================================================
    c_x = 1720

    # Cycle Consistency
    add_node("cyc_box", "Causal Cycle Consistency", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fafafa;strokeColor=#7b1fa2;strokeWidth=2;dashed=1;verticalAlign=top;fontStyle=1;fontSize=14;", c_x, 350, 400, 160)
    
    add_node("cyc_pe2", "P_e^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", c_x+40, 410, 60, 20)
    add_node("cyc_mul", "×", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=24;fontStyle=1;", c_x+110, 430, 20, 20)
    add_node("mae_bg", "M_AEᵀ", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#8e24aa;strokeWidth=2;fontStyle=1;", c_x+150, 400, 60, 80)
    add_node("cyc_eq", "≈", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=24;fontStyle=1;", c_x+230, 430, 20, 20)
    add_node("cyc_pa2", "P_a^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", c_x+280, 410, 60, 20)
    
    add_edge("e_pe2_cyc", "pe2_bg", "cyc_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;dashed=1;", [(1430, l2_y-20), (1430, 430)])
    add_edge("e_pa2_cyc", "pa2_bg", "cyc_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;dashed=1;", [(1430, l1_y-60), (1430, 430)])

    # Abductive Inference
    add_node("abd_box", "Test-Time Abductive Inference", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#ff8f00;strokeWidth=3;verticalAlign=top;fontStyle=1;fontSize=14;", c_x, 600, 400, 250)
    
    add_node("loop_bg", "Optimization Loop", "ellipse;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#ffca28;strokeWidth=3;dashed=1;fontStyle=1;", c_x+135, 650, 130, 130)
    
    add_node("abd_eq", "min E(FACS) + E(Cycle) + E(Prior)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", c_x+50, 800, 300, 30)
    
    add_edge("e_pa2_abd", "pa2_bg", "abd_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;dashed=1;", [(1450, l1_y-60), (1450, 725)])
    add_edge("e_pe2_abd", "pe2_bg", "abd_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;dashed=1;", [(1450, l2_y-20), (1450, 725)])


    xml.append('      </root>')
    xml.append('    </mxGraphModel>')
    xml.append('  </diagram>')
    xml.append('</mxfile>')

    with open(r"C:\Users\khang\OneDrive\Desktop\ctrlau\ctrlau_architecture.drawio", "w", encoding="utf-8") as f:
        f.write("\n".join(xml))

if __name__ == "__main__":
    build()
