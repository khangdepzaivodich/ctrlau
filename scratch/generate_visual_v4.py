import math
from PIL import Image, ImageDraw

def build():
    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append('<mxfile version="14.6.11">')
    xml.append('  <diagram id="ctrlau_v4" name="CtrlAU Architecture">')
    xml.append('    <mxGraphModel dx="2000" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2800" pageHeight="1400" math="0" shadow="0">')
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
    # BACKGROUNDS
    # =========================================================
    add_node("bg1", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=none;opacity=50;", 50, 50, 600, 1150)
    add_node("bg2", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f8e9;strokeColor=none;opacity=50;", 750, 50, 1200, 1150)
    add_node("bg3", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff8e1;strokeColor=none;opacity=50;", 2050, 50, 600, 1150)

    add_node("p1_lbl", "Phase 1: Feature Extraction & Alignment", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#343a40;", 50, 60, 600, 40)
    add_node("p2_lbl", "Phase 2: Parallel Causal Routing & Counterfactual Intervention", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#33691e;", 750, 60, 1200, 40)
    add_node("p3_lbl", "Phase 3: Cycle Consistency & Abductive Inference", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#ff6f00;", 2050, 60, 600, 40)

    # =========================================================
    # PHASE 1
    # =========================================================
    # Vision Pipeline (X: 100)
    add_node("img_p2", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#424242;strokeWidth=2;", 110, 140, 80, 80)
    add_node("img_p1", "", "shape=ext;double=1;rounded=0;whiteSpace=wrap;html=1;fillColor=#eceff1;strokeColor=#424242;strokeWidth=2;", 100, 150, 80, 80)
    add_node("img_face", "<font style='font-size:36px'>👤</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;", 110, 160, 60, 60)
    add_node("img_lbl", "Input Image", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;fontSize=14;fontStyle=1;", 100, 240, 80, 20)

    add_node("resnet", "ResNet-50", "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;direction=south;fillColor=#bbdefb;strokeColor=#1565c0;strokeWidth=2;fontSize=14;fontStyle=1;", 110, 300, 60, 120)
    add_edge("e_img_res", "img_p1", "resnet", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=3;strokeColor=#455a64;endArrow=block;")

    add_node("z_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#eeeeee;strokeColor=#757575;", 100, 500, 80, 80)
    for i in range(4):
        for j in range(4):
            fc = "#cfd8dc" if (i+j)%2==0 else "#ffffff"
            add_node(f"z_c_{i}_{j}", "", f"rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#b0bec5;", 100 + j*20, 500 + i*20, 20, 20)
    add_node("z_lbl", "Visual Features Z", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=top;fontSize=14;fontStyle=1;", 80, 590, 120, 30)
    add_edge("e_res_z", "resnet", "z_bg", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=3;strokeColor=#455a64;endArrow=block;")

    # Branches (X: 300)
    add_node("au_head", "SymAU Head\n(8 Branches)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#c8e6c9;strokeColor=#2e7d32;strokeWidth=2;fontSize=14;fontStyle=1;", 300, 250, 120, 100)
    add_edge("e_z_au", "z_bg", "au_head", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#2e7d32;endArrow=block;", [(140, 480), (140, 300)])
    
    add_node("va_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#1b5e20;strokeWidth=2;dashed=1;", 480, 250, 60, 100)
    for i in range(4):
        add_node(f"va_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#a5d6a7;strokeColor=#1b5e20;", 490, 260 + i*20, 40, 10)
    add_node("va_lbl", "V_a", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#1b5e20;", 490, 360, 40, 20)
    add_edge("e_au_va", "au_head", "va_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#2e7d32;endArrow=block;")

    # P_a_1 Bar Chart
    add_node("pa1_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#2e7d32;strokeWidth=2;", 470, 400, 80, 80)
    for i, h in enumerate([30, 60, 20, 50]):
        add_node(f"pa1_b_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#66bb6a;strokeColor=#2e7d32;", 475+i*15, 480-h, 12, h)
    add_node("pa1_lbl", "P_a^(1)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#2e7d32;", 470, 490, 80, 20)
    add_edge("e_va_pa1", "va_box", "pa1_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#2e7d32;endArrow=block;")

    add_node("ex_head", "SymExpr Head\n(7 Branches)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffcdd2;strokeColor=#c62828;strokeWidth=2;fontSize=14;fontStyle=1;", 300, 550, 120, 100)
    add_edge("e_z_ex", "z_bg", "ex_head", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;", [(140, 600), (140, 600)])
    
    add_node("ve_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#b71c1c;strokeWidth=2;dashed=1;", 480, 550, 60, 100)
    for i in range(4):
        add_node(f"ve_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef9a9a;strokeColor=#b71c1c;", 490, 560 + i*20, 40, 10)
    add_node("ve_lbl", "V_e", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#b71c1c;", 490, 660, 40, 20)
    add_edge("e_ex_ve", "ex_head", "ve_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;")

    # P_e_1 Bar Chart
    add_node("pe1_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#c62828;strokeWidth=2;", 470, 700, 80, 80)
    for i, h in enumerate([50, 20, 70, 40]):
        add_node(f"pe1_b_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef5350;strokeColor=#b71c1c;", 475+i*15, 780-h, 12, h)
    add_node("pe1_lbl", "P_e^(1)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#c62828;", 470, 790, 80, 20)
    add_edge("e_ve_pe1", "ve_box", "pe1_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;")

    # Text Pipeline (Bottom)
    add_node("text_in", "AU / Expr\nDescriptions", "shape=document;whiteSpace=wrap;html=1;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=2;fontStyle=1;", 100, 900, 100, 70)
    add_node("clip_enc", "Frozen CLIP\nText Encoder", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe082;strokeColor=#ff8f00;strokeWidth=2;fontStyle=1;", 250, 910, 120, 50)
    add_edge("e_text_clip", "text_in", "clip_enc", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#f57f17;endArrow=block;")
    
    add_node("t_box", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#e65100;strokeWidth=2;dashed=1;", 480, 885, 60, 100)
    for i in range(4):
        add_node(f"t_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffcc80;strokeColor=#e65100;", 490, 895 + i*20, 40, 10)
    add_node("t_lbl", "T", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#e65100;", 490, 995, 40, 20)
    add_edge("e_clip_t", "clip_enc", "t_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#f57f17;endArrow=block;")

    # Contrastive Alignment
    add_node("l_contra", "Contrastive\nAlignment\n(L_contra, L_HSIC)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e3f2fd;strokeColor=#1565c0;strokeWidth=2;fontStyle=1;", 300, 750, 120, 60)
    add_edge("e_va_contra", "va_box", "l_contra", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#1565c0;endArrow=none;startArrow=none;dashed=1;", [(460, 300), (360, 300)])
    add_edge("e_t_contra", "t_box", "l_contra", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#1565c0;endArrow=none;startArrow=none;dashed=1;", [(460, 935), (360, 935)])

    # =========================================================
    # PHASE 2
    # =========================================================
    # Top Row: Standard Pass (Y: 200)
    add_node("v_orig", "V_orig\n(Original Features)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#a5d6a7;strokeColor=#1b5e20;strokeWidth=2;fontStyle=1;", 800, 225, 120, 50)
    add_edge("e_va_vorig", "va_box", "v_orig", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#1b5e20;endArrow=block;", [(540, 250), (650, 250)])
    
    # Graph Level 1 Standard
    add_node("g1_std_box", "Graph Module Level 1\n(Shared Weights)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#388e3c;strokeWidth=2;verticalAlign=top;fontStyle=1;", 1000, 150, 250, 200)
    add_edge("e_vorig_g1", "v_orig", "g1_std_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#1b5e20;endArrow=block;")
    # Draw Graph
    g_cx1, g_cy1 = 1125, 270
    g_nodes_1 = []
    for i in range(4):
        ang = i * math.pi / 2 - math.pi/4
        nx = g_cx1 + 40 * math.cos(ang) - 20
        ny = g_cy1 + 40 * math.sin(ang) - 20
        g_nodes_1.append((nx, ny))
        add_node(f"gn1_{i}", f"AU{i+1}", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#c8e6c9;strokeColor=#388e3c;", nx, ny, 40, 40)
    
    add_node("cls1_std", "Graph Classifier", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#2e7d32;strokeWidth=2;fontStyle=1;", 1350, 225, 120, 50)
    add_edge("e_g1_cls1", "g1_std_box", "cls1_std", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#1b5e20;endArrow=block;")
    
    add_node("pa2_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#2e7d32;strokeWidth=2;", 1550, 210, 80, 80)
    for i, h in enumerate([40, 70, 30, 50]):
        add_node(f"pa2_b_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#66bb6a;strokeColor=#2e7d32;", 1555+i*15, 290-h, 12, h)
    add_node("pa2_lbl", "P_a^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#2e7d32;", 1550, 300, 80, 20)
    add_edge("e_cls1_pa2", "cls1_std", "pa2_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#2e7d32;endArrow=block;")

    # Graph Level 2 Standard
    add_node("g2_std_box", "Graph Module Level 2\n(Bipartite)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#c62828;strokeWidth=2;verticalAlign=top;fontStyle=1;", 1000, 450, 250, 200)
    add_edge("e_g1_g2", "g1_std_box", "g2_std_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#455a64;endArrow=block;")
    add_edge("e_ve_g2", "ve_box", "g2_std_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;", [(540, 600), (950, 600), (950, 550)])
    
    # Draw Graph Level 2
    g2_x, g2_y = 1050, 500
    for i in range(3):
        add_node(f"g2a_{i}", f"AU{i+1}", "ellipse;whiteSpace=wrap;html=1;fillColor=#c8e6c9;strokeColor=#388e3c;", g2_x+30, g2_y + i*40, 40, 30)
        add_node(f"g2e_{i}", f"E{i+1}", "ellipse;whiteSpace=wrap;html=1;fillColor=#ffcdd2;strokeColor=#c62828;", g2_x+180, g2_y + i*40, 40, 30)

    add_node("cls2_std", "Graph Classifier", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#c62828;strokeWidth=2;fontStyle=1;", 1350, 525, 120, 50)
    add_edge("e_g2_cls2", "g2_std_box", "cls2_std", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;")

    add_node("pe2_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#c62828;strokeWidth=2;", 1550, 510, 80, 80)
    for i, h in enumerate([50, 20, 70, 40]):
        add_node(f"pe2_b_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ef5350;strokeColor=#b71c1c;", 1555+i*15, 590-h, 12, h)
    add_node("pe2_lbl", "P_e^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#c62828;", 1550, 600, 80, 20)
    add_edge("e_cls2_pe2", "cls2_std", "pe2_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#c62828;endArrow=block;")


    # Bottom Row: Counterfactual Pass (Y: 800)
    add_node("null_box", "Semantic Subspace Nulling\n(Idea 1)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#1565c0;strokeWidth=3;verticalAlign=top;fontStyle=1;", 800, 780, 220, 160)
    add_edge("e_va_null", "va_box", "null_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#1565c0;endArrow=block;", [(540, 330), (700, 330), (700, 860)])
    add_edge("e_t_null", "t_box", "null_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#e65100;endArrow=block;", [(540, 935), (700, 935), (700, 880)])
    
    add_node("null_eq", "V_counter = V - (V·T)T", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#1565c0;", 810, 850, 200, 40)

    add_node("v_counter", "V_counter\n(Intervened)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1bee7;strokeColor=#4a148c;strokeWidth=2;fontStyle=1;", 1100, 835, 120, 50)
    add_edge("e_null_vc", "null_box", "v_counter", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#4a148c;endArrow=block;")

    add_node("g1_cf_box", "Graph Module Level 1\n(Shared Weights)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#4a148c;strokeWidth=2;verticalAlign=top;fontStyle=1;", 1300, 760, 250, 200)
    add_edge("e_vc_g1cf", "v_counter", "g1_cf_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=4;strokeColor=#4a148c;endArrow=block;")
    
    g_cx2, g_cy2 = 1425, 880
    for i in range(4):
        ang = i * math.pi / 2 - math.pi/4
        nx = g_cx2 + 40 * math.cos(ang) - 20
        ny = g_cy2 + 40 * math.sin(ang) - 20
        add_node(f"gn2_{i}", f"AU{i+1}", "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#e1bee7;strokeColor=#4a148c;", nx, ny, 40, 40)

    add_node("cls1_cf", "Graph Classifier\n(Shared Weights)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#4a148c;strokeWidth=2;fontStyle=1;", 1620, 835, 120, 50)
    add_edge("e_g1cf_cls", "g1_cf_box", "cls1_cf", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=block;")
    
    add_node("pacf_bg", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#4a148c;strokeWidth=2;", 1820, 820, 80, 80)
    for i, h in enumerate([70, 20, 90, 10]):
        add_node(f"pacf_b_{i}", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ab47bc;strokeColor=#4a148c;", 1825+i*15, 900-h, 12, h)
    add_node("pacf_lbl", "P_a^(CF)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;fontColor=#4a148c;", 1820, 910, 80, 20)
    add_edge("e_cls_pacf", "cls1_cf", "pacf_bg", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#4a148c;endArrow=block;")

    # Counterfactual Loss L_CF
    add_node("l_cf", "Counterfactual\nLoss\n(L_CF)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#7b1fa2;strokeWidth=3;fontStyle=1;fontSize=14;", 1800, 500, 120, 80)
    add_edge("e_pa2_lcf", "pa2_bg", "l_cf", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#7b1fa2;endArrow=block;dashed=1;", [(1590, 290), (1590, 540)])
    add_edge("e_pacf_lcf", "pacf_bg", "l_cf", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=3;strokeColor=#7b1fa2;endArrow=block;dashed=1;", [(1860, 820), (1860, 580)])

    # =========================================================
    # PHASE 3: CYCLE CONSISTENCY & ABDUCTIVE
    # =========================================================
    # Cycle Consistency
    add_node("cyc_box", "Causal Cycle Consistency (Idea 4)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#fafafa;strokeColor=#7b1fa2;strokeWidth=3;dashed=1;verticalAlign=top;fontStyle=1;fontSize=16;", 2100, 200, 450, 250)
    
    add_node("cyc_pe2", "P_e^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", 2150, 280, 60, 20)
    add_node("cyc_mul", "×", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=36;fontStyle=1;", 2250, 310, 30, 30)
    add_node("mae_bg", "M_AEᵀ\nPrior Matrix", "rounded=0;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#8e24aa;strokeWidth=2;fontStyle=1;", 2320, 270, 80, 100)
    add_node("cyc_eq", "≈", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=36;fontStyle=1;", 2440, 310, 30, 30)
    add_node("cyc_pa2", "P_a^(2)", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", 2510, 280, 60, 20)
    
    add_edge("e_pe2_cyc", "pe2_bg", "cyc_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;dashed=1;", [(1630, 550), (2000, 550), (2000, 325)])
    add_edge("e_pa2_cyc", "pa2_bg", "cyc_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;dashed=1;", [(1630, 250), (2100, 250)])

    # Abductive Inference
    add_node("abd_box", "Test-Time Abductive Inference (Idea 3)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#ff8f00;strokeWidth=4;verticalAlign=top;fontStyle=1;fontSize=16;", 2100, 600, 450, 400)
    
    add_node("loop_bg", "", "ellipse;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#ffca28;strokeWidth=5;dashed=1;", 2250, 680, 150, 150)
    add_edge("e_loop", "loop_bg", "loop_bg", "edgeStyle=orthogonalEdgeStyle;curved=1;html=1;strokeWidth=5;strokeColor=#ff8f00;endArrow=block;", [(2250, 755), (2280, 690), (2325, 680)])
    add_node("loop_txt", "Adam Optimizer\n(15 iterations)", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontStyle=1;fontSize=14;", 2255, 735, 140, 40)
    
    add_node("abd_eq", "min <font color='#1565c0'>E(FACS)</font> + <font color='#7b1fa2'>E(Cycle)</font> + <font color='#2e7d32'>E(Prior)</font>", "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=14;fontStyle=1;", 2175, 850, 300, 30)
    
    add_node("final_pred_box", "Final Refined Predictions", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e0f7fa;strokeColor=#006064;strokeWidth=3;fontStyle=1;", 2225, 920, 200, 60)
    
    add_edge("e_pa2_abd", "pa2_bg", "abd_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#2e7d32;endArrow=block;dashed=1;", [(1630, 270), (2020, 270), (2020, 800)])
    add_edge("e_pe2_abd", "pe2_bg", "abd_box", "edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;dashed=1;", [(1630, 570), (2040, 570), (2040, 850)])

    xml.append('      </root>')
    xml.append('    </mxGraphModel>')
    xml.append('  </diagram>')
    xml.append('</mxfile>')

    with open(r"C:\Users\khang\OneDrive\Desktop\ctrlau\ctrlau_architecture.drawio", "w", encoding="utf-8") as f:
        f.write("\n".join(xml))

if __name__ == "__main__":
    build()
