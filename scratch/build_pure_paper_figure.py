import xml.etree.ElementTree as ET

def build_pure_paper_figure():
    # 1. Extract base64 face image from backup
    tree = ET.parse('ctrlau_backup.drawio')
    root = tree.getroot()
    image_style = None
    for c in root.iter('mxCell'):
        if c.get('id') == 'GjfUm5Ax_Uw3Q_51WUAJ-1':
            image_style = c.get('style')
            break
            
    if not image_style:
        raise ValueError("Image not found in backup")

    cells = []
    
    def add_node(cid, parent, val, x, y, w, h, style):
        cells.append({
            'type': 'node',
            'id': cid,
            'parent': parent,
            'value': val,
            'x': round(x, 1), 'y': round(y, 1), 'w': round(w, 1), 'h': round(h, 1),
            'style': style
        })
        return cid

    def add_edge(cid, parent, source, target, val, style, points=None):
        cells.append({
            'type': 'edge',
            'id': cid,
            'parent': parent,
            'source': source,
            'target': target,
            'value': val,
            'style': style,
            'points': points or []
        })
        return cid

    # Root
    cells.append({'type': 'node', 'id': '0', 'parent': None, 'value': '', 'style': ''})
    cells.append({'type': 'node', 'id': '1', 'parent': '0', 'value': '', 'style': ''})

    # =========================================================================
    # STYLES (Pure Academic Paper - CVPR/ICCV Aesthetic)
    # =========================================================================
    FONT = "fontFamily=Helvetica;"
    
    # Section Header Styles
    STYLE_PHASE_HEADER = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontStyle=1;fontSize=15;fontColor=#1a237e;{FONT}"
    STYLE_SUBTITLE = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontColor=#546e7a;fontSize=9.5;{FONT}"
    BANNER_P2 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#1b5e20;strokeColor=none;fontColor=#ffffff;align=left;spacingLeft=15;fontStyle=1;fontSize=12;{FONT}"

    # Visual Components
    STYLE_BACKBONE = f"shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=10;fillColor=#e3f2fd;strokeColor=#1976d2;strokeWidth=1.5;fontColor=#0d47a1;fontSize=11;fontStyle=1;{FONT}"
    STYLE_LINEAR_BLOCK = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#9e9e9e;strokeWidth=1.2;fontColor=#212121;fontSize=10;{FONT}"
    STYLE_LINEAR = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;strokeWidth=1.2;fontColor=#212121;fontSize=10;{FONT}"
    STYLE_CLIP = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#d1c4e9;strokeColor=#7e57c2;strokeWidth=1.5;fontColor=#311b92;fontSize=10;fontStyle=1;{FONT}"

    # 3D Tensor Cubes
    STYLE_CUBE_BLUE = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#bbdefb;strokeColor=#1976d2;strokeWidth=1.2;fontColor=#0d47a1;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_CYAN = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#e1f5fe;strokeColor=#0288d1;strokeWidth=1.2;fontColor=#01579b;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_AU = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#ffffff;strokeColor=#7e57c2;strokeWidth=1.5;fontColor=#4a148c;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_EXP = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#ffffff;strokeColor=#d81b60;strokeWidth=1.5;fontColor=#880e4f;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_GREEN = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#e8f5e9;strokeColor=#43a047;strokeWidth=1.5;fontColor=#1b5e20;fontSize=10;fontStyle=1;{FONT}"
    STYLE_CUBE_YELLOW = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=1.5;fontColor=#f57f17;fontSize=10;fontStyle=1;{FONT}"
    STYLE_CUBE_PURPLE = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.5;fontColor=#4a148c;fontSize=10;fontStyle=1;{FONT}"

    # Multi-branch CNN blocks
    STYLE_CNN_AU = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=1.2;fontColor=#311b92;fontSize=10;fontStyle=1;{FONT}"
    STYLE_CNN_EXP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=1.2;fontColor=#880e4f;fontSize=10;fontStyle=1;{FONT}"

    # Prediction bars
    STYLE_BAR_AU = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ab47bc;strokeColor=#6a1b9a;strokeWidth=1;{FONT}"
    STYLE_BAR_EXP = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#e91e63;strokeColor=#ad1457;strokeWidth=1;{FONT}"

    # Academic Loss Badges (Ellipses & Compact Pills)
    STYLE_LOSS_CIRCLE_RED = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffebee;strokeColor=#d32f2f;strokeWidth=1.5;fontColor=#b71c1c;fontSize=10;{FONT}"
    STYLE_LOSS_CIRCLE_GREEN = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#e8f5e9;strokeColor=#388e3c;strokeWidth=1.5;fontColor=#1b5e20;fontSize=10;{FONT}"
    STYLE_LOSS_CIRCLE_ORANGE = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#fff3e0;strokeColor=#f57c00;strokeWidth=1.5;fontColor=#e65100;fontSize=10;{FONT}"
    
    STYLE_LOSS_PILL_RED = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#e53935;strokeWidth=1.2;fontColor=#b71c1c;fontSize=9.5;{FONT}"
    STYLE_LOSS_PILL_GREEN = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#43a047;strokeWidth=1.2;fontColor=#1b5e20;fontSize=9.5;{FONT}"
    STYLE_FORMULA = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#b0bec5;strokeWidth=1;fontColor=#37474f;fontSize=9.5;{FONT}"

    # Graph Nodes
    STYLE_GRAPH_NODE_AU = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.5;fontColor=#4a148c;fontStyle=1;fontSize=10;{FONT}"
    STYLE_GRAPH_NODE_EXP = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#fce4ec;strokeColor=#d81b60;strokeWidth=1.5;fontColor=#880e4f;fontStyle=1;fontSize=10;{FONT}"

    # Laser-straight connectives
    ARROW_H = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#37474f;strokeWidth=1.6;exitX=1;exitY=0.5;entryX=0;entryY=0.5;{FONT}"
    ARROW_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#37474f;strokeWidth=1.6;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_DASH_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#78909c;strokeWidth=1.2;dashed=1;dashPattern=4 3;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_CASCADE = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2e7d32;strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;fontColor=#1b5e20;fontStyle=1;fontSize=11;{FONT}"
    ARROW_GRAPH = f"edgeStyle=straightEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2e7d32;strokeWidth=1.5;{FONT}"

    # =========================================================================
    # 1. PHASE 1: REPRESENTATION LEARNING & MULTI-BRANCH CLASSIFIER
    # =========================================================================
    add_node("title_p1", "1", "<b>1. Multi-Branch Feature Learning &amp; Representation Disentanglement (Phase 1)</b>", 60, 20, 800, 25, STYLE_PHASE_HEADER)

    # 1A. CLIP Text Stream (Center Y = 75)
    node_txt = add_node("node_txt", "1", "<b>FACS Descriptions</b><br><font style=\"font-size:8.5px;color:#666;\">(Text Prompts)</font>", 60, 55, 130, 40, STYLE_LINEAR_BLOCK)
    node_clip = add_node("node_clip", "1", "<b>CLIP Text</b><br><font style=\"font-size:8.5px;\">Encoder (❄️)</font>", 235, 50, 95, 50, STYLE_CLIP)
    node_temb = add_node("node_temb", "1", "<b>`\\mathbf{T}_a, \\mathbf{T}_e`</b><br><font style=\"font-size:8.5px;\">(256)</font>", 375, 55, 75, 40, STYLE_CUBE_PURPLE)
    node_lcon = add_node("node_lcon", "1", "<span style=\"color:#e65100;\"><b>`\\mathcal{L}_{\\text{con}}`</b></span>", 495, 57, 60, 36, STYLE_LOSS_CIRCLE_ORANGE)

    add_edge("e_txt_clip", "1", "node_txt", "node_clip", "", ARROW_H)
    add_edge("e_clip_temb", "1", "node_clip", "node_temb", "", ARROW_H)
    add_edge("e_temb_lcon", "1", "node_temb", "node_lcon", "", ARROW_H)

    # 1B. Visual Feature Stream (Center Y = 175)
    img_face = add_node("GjfUm5Ax_Uw3Q_51WUAJ-1", "1", "", 60, 140, 70, 70, image_style)
    node_bb = add_node("node_bb", "1", "<b>ResNet-50</b><br><font style=\"font-size:8.5px;\">(2048 &times; 7 &times; 7)</font>", 175, 145, 95, 60, STYLE_BACKBONE)
    node_lp = add_node("node_lp", "1", "<b>LinearBlock</b><br><font style=\"font-size:8.5px;\">`2048 \\to 512`</font>", 310, 150, 85, 50, STYLE_LINEAR_BLOCK)
    node_feat = add_node("node_feat", "1", "<b>Feature Map</b><br><font style=\"font-size:8px;\">`49 \\times 512`</font>", 435, 140, 75, 70, STYLE_CUBE_BLUE)
    node_grid = add_node("node_grid", "1", "<b>Patches</b><br><font style=\"font-size:8px;\">`49 \\times 512`</font>", 550, 145, 80, 60, STYLE_CUBE_CYAN)

    add_edge("e_f_bb", "1", "GjfUm5Ax_Uw3Q_51WUAJ-1", "node_bb", "", ARROW_H)
    add_edge("e_bb_lp", "1", "node_bb", "node_lp", "", ARROW_H)
    add_edge("e_lp_feat", "1", "node_lp", "node_feat", "", ARROW_H)
    add_edge("e_feat_grid", "1", "node_feat", "node_grid", "", ARROW_H)

    # 1C. HSIC Representation Disentanglement (Clean Supervision Tags)
    add_node("lbl_hsic", "1", "<b>HSIC Representation Disentanglement:</b>", 60, 245, 300, 20, STYLE_PHASE_HEADER)
    
    node_lib = add_node("node_lib", "1", 
                        "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{ib}}`</b></span><br><font style=\"font-size:8.5px;\">`\\text{HSIC}(\\mathbf{V}_a, \\mathbf{z}_{\\text{img}})`</font>", 
                        60, 275, 170, 36, STYLE_LOSS_PILL_RED)
    node_lalign = add_node("node_lalign", "1", 
                           "<span style=\"color:#1b5e20;\"><b>`\\mathcal{L}_{\\text{align}}`</b></span><br><font style=\"font-size:8.5px;\">`-\\text{HSIC}(\\mathbf{V}_a, \\mathbf{Y}_a)`</font>", 
                           245, 275, 170, 36, STYLE_LOSS_PILL_GREEN)
    node_ldecorr = add_node("node_ldecorr", "1", 
                            "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{decorr}}`</b></span><br><font style=\"font-size:8.5px;\">`\\sum_{i \\ne j} \\text{HSIC}(\\mathbf{V}_a^i, \\mathbf{V}_a^j)`</font>", 
                            430, 275, 185, 36, STYLE_LOSS_PILL_RED)

    # Phase 1 Total Loss Formula Tag
    add_node("formula_p1", "1", 
             "<b>Phase 1 Loss:</b> `\\mathcal{L}_{\\text{phase1}} = \\mathcal{L}_{\\text{wa}} + \\gamma \\mathcal{L}_{\\text{we}} + \\lambda_{\\text{ib}} \\mathcal{L}_{\\text{ib}} + \\lambda_{\\text{align}} \\mathcal{L}_{\\text{align}} + \\lambda_{\\text{decorr}} \\mathcal{L}_{\\text{decorr}} + \\lambda_{\\text{con}} \\mathcal{L}_{\\text{con}}`", 
             60, 340, 555, 36, STYLE_FORMULA)

    # -------------------------------------------------------------------------
    # 1D. Multi-Branch Spatial Classifier (Right Side: X = 700 to 1350)
    # -------------------------------------------------------------------------
    add_node("title_multi", "1", "<b>Multi-Branch Spatial Heads &amp; FACS Prior</b>", 710, 35, 350, 20, STYLE_PHASE_HEADER)

    # --- AU Head (Rows at Center Y = 100, 142, 198) ---
    add_node("lbl_auhead", "1", "<b>AU Head (8 Branches)</b>", 710, 68, 180, 18, STYLE_SUBTITLE)

    # Row AU1 (Center Y = 100)
    add_node("cnn_au1", "1", "1D CNN<br>(AU1)", 710, 83, 65, 34, STYLE_CNN_AU)
    add_node("va_1", "1", "<b>`\\mathbf{V}_a^{(1)}`</b>", 800, 81, 42, 38, STYLE_CUBE_AU)
    add_node("lin_au1", "1", "Linear", 870, 85, 45, 30, STYLE_LINEAR)
    add_node("pau1", "1", "", 935, 83, 8, 34, STYLE_BAR_AU)

    add_edge("e_c1_v1", "1", "cnn_au1", "va_1", "", ARROW_H)
    add_edge("e_v1_l1", "1", "va_1", "lin_au1", "", ARROW_H)
    add_edge("e_l1_p1", "1", "lin_au1", "pau1", "", ARROW_H)

    # Row AU2 (Center Y = 142)
    add_node("cnn_au2", "1", "1D CNN<br>(AU2)", 710, 125, 65, 34, STYLE_CNN_AU)
    add_node("va_2", "1", "<b>`\\mathbf{V}_a^{(2)}`</b>", 800, 123, 42, 38, STYLE_CUBE_AU)
    add_node("lin_au2", "1", "Linear", 870, 127, 45, 30, STYLE_LINEAR)
    add_node("pau2", "1", "", 935, 125, 8, 34, STYLE_BAR_AU)

    add_edge("e_c2_v2", "1", "cnn_au2", "va_2", "", ARROW_H)
    add_edge("e_v2_l2", "1", "va_2", "lin_au2", "", ARROW_H)
    add_edge("e_l2_p2", "1", "lin_au2", "pau2", "", ARROW_H)

    # Dots
    add_node("dots_au", "1", "<b>&vellip;</b>", 735, 162, 20, 20, STYLE_PHASE_HEADER)

    # Row AU12 (Center Y = 198)
    add_node("cnn_au12", "1", "1D CNN<br>(AU12)", 710, 181, 65, 34, STYLE_CNN_AU)
    add_node("va_12", "1", "<b>`\\mathbf{V}_a^{(12)}`</b>", 800, 179, 42, 38, STYLE_CUBE_AU)
    add_node("lin_au12", "1", "Linear", 870, 183, 45, 30, STYLE_LINEAR)
    add_node("pau12", "1", "", 935, 181, 8, 34, STYLE_BAR_AU)

    add_edge("e_c12_v12", "1", "cnn_au12", "va_12", "", ARROW_H)
    add_edge("e_v12_l12", "1", "va_12", "lin_au12", "", ARROW_H)
    add_edge("e_l12_p12", "1", "lin_au12", "pau12", "", ARROW_H)

    # AU Loss & Prediction (Center Y = 142, Center X = 1030)
    add_node("lbl_pau", "1", "<b>`\\mathbf{P}_{\\text{AU}}`</b>", 955, 132, 35, 20, STYLE_SUBTITLE)
    node_lwa = add_node("node_lwa", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{wa}}`</b></span>", 1005, 126, 60, 32, STYLE_LOSS_CIRCLE_RED)
    add_edge("e_pau_lwa", "1", "pau2", "node_lwa", "", ARROW_H)

    # Output AU Embeddings (Center Y = 142)
    node_va_out = add_node("node_va_out", "1", "<b>`\\mathbf{V}_a`</b><br><font style=\"font-size:8px;\">`8 \\times 256`</font>", 1110, 121, 80, 42, STYLE_CUBE_PURPLE)

    # --- FACS Prior Matrix M_AE --- (Center X = 1035, Center Y = 225)
    node_mae = add_node("node_mae", "1", "<b>FACS Prior Matrix `\\mathbf{M}_{\\text{AE}}`</b><br><font style=\"font-size:7.5px;color:#555;\">`\\mathbf{Y}_e = \\operatorname{argmax}(\\mathbf{Y}_a \\mathbf{M}_{\\text{AE}})`</font>", 970, 207, 130, 36, STYLE_FORMULA)
    add_edge("e_lwa_mae", "1", "node_lwa", "node_mae", "", ARROW_DASH_V)

    # --- Emotion Head (Rows at Center Y = 280, 322, 378) ---
    add_node("lbl_exphead", "1", "<b>Expression Head (7 Branches)</b>", 710, 248, 200, 18, STYLE_SUBTITLE)

    # Row E1 (Center Y = 280)
    add_node("cnn_e1", "1", "1D CNN<br>(E1)", 710, 263, 65, 34, STYLE_CNN_EXP)
    add_node("ve_1", "1", "<b>`\\mathbf{V}_e^{(1)}`</b>", 800, 261, 42, 38, STYLE_CUBE_EXP)
    add_node("lin_e1", "1", "Linear", 870, 265, 45, 30, STYLE_LINEAR)
    add_node("pe1", "1", "", 935, 263, 8, 34, STYLE_BAR_EXP)

    add_edge("e_ce1_ve1", "1", "cnn_e1", "ve_1", "", ARROW_H)
    add_edge("e_ve1_le1", "1", "ve_1", "lin_e1", "", ARROW_H)
    add_edge("e_le1_pe1", "1", "lin_e1", "pe1", "", ARROW_H)

    # Row E2 (Center Y = 322)
    add_node("cnn_e2", "1", "1D CNN<br>(E2)", 710, 305, 65, 34, STYLE_CNN_EXP)
    add_node("ve_2", "1", "<b>`\\mathbf{V}_e^{(2)}`</b>", 800, 303, 42, 38, STYLE_CUBE_EXP)
    add_node("lin_e2", "1", "Linear", 870, 307, 45, 30, STYLE_LINEAR)
    add_node("pe2", "1", "", 935, 305, 8, 34, STYLE_BAR_EXP)

    add_edge("e_ce2_ve2", "1", "cnn_e2", "ve_2", "", ARROW_H)
    add_edge("e_ve2_le2", "1", "ve_2", "lin_e2", "", ARROW_H)
    add_edge("e_le2_pe2", "1", "lin_e2", "pe2", "", ARROW_H)

    # Dots
    add_node("dots_exp", "1", "<b>&vellip;</b>", 735, 342, 20, 20, STYLE_PHASE_HEADER)

    # Row E7 (Center Y = 378)
    add_node("cnn_e7", "1", "1D CNN<br>(E7)", 710, 361, 65, 34, STYLE_CNN_EXP)
    add_node("ve_7", "1", "<b>`\\mathbf{V}_e^{(7)}`</b>", 800, 359, 42, 38, STYLE_CUBE_EXP)
    add_node("lin_e7", "1", "Linear", 870, 363, 45, 30, STYLE_LINEAR)
    add_node("pe7", "1", "", 935, 361, 8, 34, STYLE_BAR_EXP)

    add_edge("e_ce7_ve7", "1", "cnn_e7", "ve_7", "", ARROW_H)
    add_edge("e_ve7_le7", "1", "ve_7", "lin_e7", "", ARROW_H)
    add_edge("e_le7_pe7", "1", "lin_e7", "pe7", "", ARROW_H)

    # Emotion Loss & Prediction (Center Y = 322, Center X = 1035)
    add_node("lbl_pe", "1", "<b>`\\mathbf{P}_{\\text{E}}`</b>", 955, 312, 35, 20, STYLE_SUBTITLE)
    node_lwe = add_node("node_lwe", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{we}}`</b></span>", 1005, 306, 60, 32, STYLE_LOSS_CIRCLE_RED)
    add_edge("e_pe_lwe", "1", "pe2", "node_lwe", "", ARROW_H)
    add_edge("e_mae_lwe", "1", "node_mae", "node_lwe", "", ARROW_DASH_V)

    # Output Emotion Embeddings (Center Y = 322)
    node_ve_out = add_node("node_ve_out", "1", "<b>`\\mathbf{V}_e`</b><br><font style=\"font-size:8px;\">`7 \\times 256`</font>", 1110, 301, 80, 42, STYLE_CUBE_EXP)

    # Connect Spatial Patches to Multi-branch CNNs with clean fork arrows
    add_edge("e_grid_au", "1", "node_grid", "cnn_au12", "", "edgeStyle=straightEdgeStyle;html=1;strokeColor=#37474f;strokeWidth=1.6;")
    add_edge("e_grid_emo", "1", "node_grid", "cnn_e1", "", "edgeStyle=straightEdgeStyle;html=1;strokeColor=#37474f;strokeWidth=1.6;")

    # =========================================================================
    # 2. PHASE 2: TWO-LEVEL CAUSAL GRAPH & COUNTERFACTUAL REASONING
    # =========================================================================
    add_node("banner_phase2", "1", 
        "<b>2. Two-Level Causal Graph Discovery &amp; Counterfactual Reasoning (Phase 2)</b> &nbsp;&nbsp;&nbsp;&nbsp; "
        "<span style=\"background-color:#ffebee;color:#c62828;padding:2px 8px;border-radius:3px;font-size:10px;border:1px solid #ef9a9a;\"><b>❄️ Frozen: ResNet-50 &amp; 1D CNN Heads</b></span>",
        50, 435, 1280, 28, BANNER_P2)

    # --- Stage 1: Level 1: AU-AU Causal Graph (X = 60 to 410, Center Y = 640) ---
    add_node("title_auau", "1", "<font style=\"font-size:13px;font-weight:bold;color:#1b5e20;\">Level 1: AU-AU Causal Graph</font>", 70, 480, 250, 20, STYLE_PHASE_HEADER)
    add_node("sub_auau", "1", "<font style=\"font-size:9px;color:#555;\">Discovers acyclic DAG `\\mathbf{A}_{\\text{AU-AU}}` over `\\mathbf{V}_a`</font>", 70, 500, 330, 16, STYLE_SUBTITLE)

    # AU Graph Nodes (Center Y = 548 for row 1, 608 for row 2)
    gau_1 = add_node("gau_1", "1", "AU1", 75, 534, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_2 = add_node("gau_2", "1", "AU2", 135, 534, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_4 = add_node("gau_4", "1", "AU4", 195, 534, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_6 = add_node("gau_6", "1", "AU6", 75, 594, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_12 = add_node("gau_12", "1", "AU12", 135, 594, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_25 = add_node("gau_25", "1", "AU25", 195, 594, 28, 28, STYLE_GRAPH_NODE_AU)

    add_edge("eg_1_2", "1", "gau_1", "gau_2", "", ARROW_GRAPH)
    add_edge("eg_2_4", "1", "gau_2", "gau_4", "", ARROW_GRAPH)
    add_edge("eg_6_12", "1", "gau_6", "gau_12", "", ARROW_GRAPH)
    add_edge("eg_12_25", "1", "gau_12", "gau_25", "", ARROW_GRAPH)
    add_edge("eg_2_12", "1", "gau_2", "gau_12", "", ARROW_GRAPH)

    # GAT and Losses in AU-AU (Center X = 320)
    # gau_4 center Y = 548. node_gat1 center Y = 528 + 20 = 548.
    node_gat1 = add_node("node_gat1", "1", "<b>GAT Layer</b><br><font style=\"font-size:8px;\">`\\mathbf{A}_{\\text{AU-AU}}`</font>", 265, 528, 110, 40, STYLE_LINEAR_BLOCK)
    node_ldag = add_node("node_ldag", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{DAG}}`</b></span>", 275, 590, 90, 32, STYLE_LOSS_CIRCLE_RED)
    node_lcau1 = add_node("node_lcau1", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{causal}}^{\\text{AU}}`</b></span>", 270, 642, 100, 32, STYLE_LOSS_CIRCLE_RED)
    node_pauau = add_node("node_pauau", "1", "<b>`\\mathbf{P}_{\\text{AU-AU}}`</b> &rarr; <span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{au\\_au}}`</b></span> (WAL)", 235, 694, 170, 38, STYLE_FORMULA)

    add_edge("eg_4_gat1", "1", "gau_4", "node_gat1", "", ARROW_H)
    add_edge("e_gat1_dag", "1", "node_gat1", "node_ldag", "", ARROW_V)
    add_edge("e_ldag_lcau", "1", "node_ldag", "node_lcau1", "", ARROW_V)
    add_edge("e_lcau_pauau", "1", "node_lcau1", "node_pauau", "", ARROW_V)

    # Output node (Updated AU) (Center Y = 780)
    node_up_au = add_node("node_up_au", "1", "<b>Updated AU Embeddings:</b> `\\mathbf{V}_a^* \\in \\mathbb{R}^{8 \\times 256}`", 90, 760, 270, 40, STYLE_CUBE_GREEN)

    # --- Stage 2: Level 2: AU -> Expression Graph (X = 470 to 880, Center Y = 640) ---
    add_node("title_auexp", "1", "<font style=\"font-size:13px;font-weight:bold;color:#1b5e20;\">Level 2: AU &rarr; Expression Graph</font>", 485, 480, 270, 20, STYLE_PHASE_HEADER)
    add_node("sub_auexp", "1", "<font style=\"font-size:9px;color:#555;\">Bipartite GAT `\\mathbf{A}_{\\text{AU-Exp}}` guided by FACS prior</font>", 485, 500, 350, 16, STYLE_SUBTITLE)

    # Cascade Arrow from Stage 1 to Stage 2: Center Y = 780.0
    add_edge("e_cascade_1_2", "1", "node_up_au", "node_mimp", "Cascade: `\\mathbf{V}_a^*`", ARROW_CASCADE)

    # Bipartite Network Nodes (AU on left, Emotion on right)
    bau_6 = add_node("bau_6", "1", "AU6", 485, 535, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_12 = add_node("bau_12", "1", "AU12", 485, 570, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_4 = add_node("bau_4", "1", "AU4", 485, 605, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_1 = add_node("bau_1", "1", "AU1", 485, 640, 26, 26, STYLE_GRAPH_NODE_AU)

    bemo_hap = add_node("bemo_hap", "1", "Happy", 575, 534, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_sad = add_node("bemo_sad", "1", "Sad", 575, 580, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_fea = add_node("bemo_fea", "1", "Fear", 575, 626, 28, 28, STYLE_GRAPH_NODE_EXP)

    add_edge("eb_6_h", "1", "bau_6", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_12_h", "1", "bau_12", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_4_s", "1", "bau_4", "bemo_sad", "", ARROW_GRAPH)
    add_edge("eb_1_f", "1", "bau_1", "bemo_fea", "", ARROW_GRAPH)

    # Bipartite GAT and Losses (Center X = 760)
    # bemo_hap center Y = 548. node_gat2 center Y = 528 + 20 = 548.
    node_gat2 = add_node("node_gat2", "1", "<b>Bipartite GAT</b><br><font style=\"font-size:8px;\">`\\mathbf{A}_{\\text{AU-Exp}}`</font>", 700, 528, 120, 40, STYLE_LINEAR_BLOCK)
    node_lcau2 = add_node("node_lcau2", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{causal}}^{\\text{Exp}}`</b></span>", 710, 590, 100, 32, STYLE_LOSS_CIRCLE_RED)
    node_lfacs = add_node("node_lfacs", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{facs}}^{\\text{Exp}}`</b></span>", 710, 642, 100, 32, STYLE_LOSS_CIRCLE_RED)
    node_pfinal = add_node("node_pfinal", "1", "Final: <b>`\\mathbf{P}_G^{\\text{AU}}, \\mathbf{P}_G^{\\text{Exp}}`</b> &rarr; <span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{wa}}^{\\text{AU}}, \\mathcal{L}_{\\text{we}}^{\\text{Exp}}`</b></span>", 665, 694, 190, 38, STYLE_FORMULA)

    add_edge("eb_hap_gat2", "1", "bemo_hap", "node_gat2", "", ARROW_H)
    add_edge("e_gat2_cau", "1", "node_gat2", "node_lcau2", "", ARROW_V)
    add_edge("e_lcau_facs", "1", "node_lcau2", "node_lfacs", "", ARROW_V)
    add_edge("e_lfacs_pfinal", "1", "node_lfacs", "node_pfinal", "", ARROW_V)

    # Output node (Importance Mask) (Center Y = 780)
    node_mimp = add_node("node_mimp", "1", "<b>Causal Importance Mask:</b> `\\mathbf{M}_{\\text{imp}} \\in \\mathbb{R}^{8 \\times 8}`", 505, 760, 275, 40, STYLE_CUBE_YELLOW)

    # --- Stage 3: Level 3: HiMod Counterfactual Intervention (X = 930 to 1350) ---
    add_node("title_himod", "1", "<font style=\"font-size:13px;font-weight:bold;color:#1b5e20;\">HiMod Counterfactual Reasoning</font>", 945, 480, 280, 20, STYLE_PHASE_HEADER)
    add_node("sub_himod", "1", "<font style=\"font-size:9px;color:#555;\">Intervention via importance mask `\\mathbf{M}_{\\text{imp}}` partition</font>", 945, 500, 350, 16, STYLE_SUBTITLE)

    # Cascade Arrow from Stage 2 to Stage 3: Center Y = 780.0
    add_edge("e_cascade_2_3", "1", "node_mimp", "node_p2_loss", "Cascade: `\\mathbf{M}_{\\text{imp}}`", ARROW_CASCADE)

    # Partition Selector Bar
    add_node("node_cf_part", "1", "<b>Causal Importance Partition:</b> `\\operatorname{diag}(\\mathbf{M}_{\\text{imp}}) > \\tau`", 945, 535, 360, 30, STYLE_FORMULA)
    
    # Path 1: Important AU Perturbation (Center Y = 605)
    add_node("node_cf_imp", "1", "<b>Perturb Important AU:</b><br>`\\mathbf{V}_a^* + \\epsilon \\cdot \\mathbf{M}_{\\text{imp}}`", 945, 586, 215, 38, STYLE_CUBE_PURPLE)
    add_node("node_lcf_imp", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{cf}}^{\\text{imp}}`</b></span><br><font style=\"font-size:7.5px;\">Invariance</font>", 1195, 586, 100, 38, STYLE_LOSS_CIRCLE_RED)
    add_edge("e_imp_loss", "1", "node_cf_imp", "node_lcf_imp", "", ARROW_H)

    # Path 2: Unimportant AU Perturbation (Center Y = 675)
    add_node("node_cf_unimp", "1", "<b>Perturb Unimportant AU:</b><br>`\\mathbf{V}_a^* + \\epsilon \\cdot (1 - \\mathbf{M}_{\\text{imp}})`", 945, 656, 215, 38, STYLE_CUBE_GREEN)
    add_node("node_lcf_unimp", "1", "<span style=\"color:#1b5e20;\"><b>`\\mathcal{L}_{\\text{cf}}^{\\text{unimp}}`</b></span><br><font style=\"font-size:7.5px;\">Consistency</font>", 1195, 656, 100, 38, STYLE_LOSS_CIRCLE_GREEN)
    add_edge("e_unimp_loss", "1", "node_cf_unimp", "node_lcf_unimp", "", ARROW_H)

    # Phase 2 Total loss formula
    add_node("node_p2_loss", "1", 
        "<b>Phase 2 Loss:</b> `\\mathcal{L}_{\\text{phase2}} = \\mathcal{L}_{\\text{wa}} + \\gamma \\mathcal{L}_{\\text{we}} + \\lambda_{\\text{dag}} \\mathcal{L}_{\\text{DAG}} + \\lambda_{\\text{cau}} \\mathcal{L}_{\\text{causal}} + \\lambda_{\\text{facs}} \\mathcal{L}_{\\text{facs}}^{\\text{Exp}} + \\lambda_{\\text{cf}} \\mathcal{L}_{\\text{cf}}`", 
        945, 760, 360, 40, STYLE_FORMULA)

    # =========================================================================
    # BUILD XML
    # =========================================================================
    mxfile = ET.Element('mxfile', host='app.diagrams.net')
    diagram = ET.SubElement(mxfile, 'diagram', name='CtrlAU Architecture', id='cQhNqvIMsQzl_Nc4XbVt')
    graph_model = ET.SubElement(diagram, 'mxGraphModel', 
                                grid='0', page='1', gridSize='10', 
                                guides='1', tooltips='1', connect='1', arrows='1', fold='1', pageScale='1', 
                                pageWidth='1400', pageHeight='950', math='1', shadow='0')
    root_el = ET.SubElement(graph_model, 'root')

    for c in cells:
        cell_el = ET.SubElement(root_el, 'mxCell')
        cell_el.set('id', c['id'])
        if c.get('parent') is not None:
            cell_el.set('parent', c['parent'])
        if c.get('value'):
            cell_el.set('value', c['value'])
        if c.get('style'):
            cell_el.set('style', c['style'])
            
        if c['type'] == 'node' and c.get('x') is not None:
            cell_el.set('vertex', '1')
            geom = ET.SubElement(cell_el, 'mxGeometry')
            geom.set('x', str(c['x']))
            geom.set('y', str(c['y']))
            geom.set('width', str(c['w']))
            geom.set('height', str(c['h']))
            geom.set('as', 'geometry')
        elif c['type'] == 'edge':
            cell_el.set('edge', '1')
            if c.get('source'):
                cell_el.set('source', c['source'])
            if c.get('target'):
                cell_el.set('target', c['target'])
            geom = ET.SubElement(cell_el, 'mxGeometry')
            geom.set('relative', '1')
            geom.set('as', 'geometry')
            if c.get('points'):
                arr = ET.SubElement(geom, 'Array')
                arr.set('as', 'points')
                for pt in c['points']:
                    mxpt = ET.SubElement(arr, 'mxPoint')
                    mxpt.set('x', str(pt[0]))
                    mxpt.set('y', str(pt[1]))

    out_xml = ET.tostring(mxfile, encoding='utf-8')
    with open('ctrlau.drawio', 'wb') as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(out_xml)
        
    print(f"Successfully generated pure paper figure ctrlau.drawio with {len(cells)} cells!")

if __name__ == '__main__':
    build_pure_paper_figure()
