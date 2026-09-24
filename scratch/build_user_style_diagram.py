import xml.etree.ElementTree as ET

def build_user_style_diagram():
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
    # STYLES (Clean Academic 3-Panel Paper Style - User Original Palette)
    # =========================================================================
    FONT = "fontFamily=Helvetica;"
    
    # 3 Main Panel Containers (Clean, light borders, NO heavy cards)
    PANEL_CONTAINER = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#b0bec5;strokeWidth=1.5;arcSize=2;{FONT}"
    PANEL_TITLE = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontStyle=1;fontSize=15;fontColor=#263238;{FONT}"
    SUBTITLE = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontColor=#546e7a;fontSize=9.5;{FONT}"

    # Visual Components (User's authentic color scheme)
    STYLE_BACKBONE = f"shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=10;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=1.5;fontColor=#0d47a1;fontSize=10.5;fontStyle=1;{FONT}"
    STYLE_LINEAR_BLOCK = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#9e9e9e;strokeWidth=1.2;fontColor=#212121;fontSize=10;{FONT}"
    STYLE_LINEAR_YELLOW = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;strokeWidth=1.2;fontColor=#212121;fontSize=10;{FONT}"
    STYLE_LINEAR_AU = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=1.2;fontColor=#311b92;fontSize=10;{FONT}"
    STYLE_LINEAR_EXP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=1.2;fontColor=#880e4f;fontSize=10;{FONT}"
    STYLE_CLIP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=1.5;fontColor=#311b92;fontSize=10;fontStyle=1;{FONT}"

    # 3D Tensor Cubes (User's exact style: clean cubes with black/subtle borders)
    STYLE_CUBE_FEAT = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=1.2;fontColor=#0d47a1;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_AU = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#ffffff;strokeColor=#4a148c;strokeWidth=1.2;fontColor=#4a148c;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_EXP = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#ffffff;strokeColor=#880e4f;strokeWidth=1.2;fontColor=#880e4f;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_GREEN = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#e8f5e9;strokeColor=#43a047;strokeWidth=1.5;fontColor=#1b5e20;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_YELLOW = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=1.5;fontColor=#f57f17;fontSize=9.5;fontStyle=1;{FONT}"

    # Multi-branch CNN blocks
    STYLE_CNN_AU = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=1.2;fontColor=#311b92;fontSize=10;fontStyle=1;{FONT}"
    STYLE_CNN_EXP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=1.2;fontColor=#880e4f;fontSize=10;fontStyle=1;{FONT}"

    # Prediction bars
    STYLE_BAR_AU = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ab47bc;strokeColor=#6a1b9a;strokeWidth=1;{FONT}"
    STYLE_BAR_EXP = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#e91e63;strokeColor=#ad1457;strokeWidth=1;{FONT}"

    # Pure Academic Loss Badges (Compact, elegant, never overflow)
    STYLE_LOSS_RED = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#e53935;strokeWidth=1.2;fontColor=#b71c1c;fontSize=9.5;{FONT}"
    STYLE_LOSS_GREEN = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#43a047;strokeWidth=1.2;fontColor=#1b5e20;fontSize=9.5;{FONT}"
    STYLE_FORMULA = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#b0bec5;strokeWidth=1;fontColor=#37474f;fontSize=9.5;{FONT}"

    # Graph Nodes
    STYLE_GRAPH_NODE_AU = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.5;fontColor=#4a148c;fontStyle=1;fontSize=10;{FONT}"
    STYLE_GRAPH_NODE_EXP = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#fce4ec;strokeColor=#d81b60;strokeWidth=1.5;fontColor=#880e4f;fontStyle=1;fontSize=10;{FONT}"

    # Connectors
    ARROW_H = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#37474f;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;{FONT}"
    ARROW_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#37474f;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_DASH_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#78909c;strokeWidth=1.2;dashed=1;dashPattern=4 3;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_CASCADE = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2e7d32;strokeWidth=1.8;exitX=1;exitY=0.5;entryX=0;entryY=0.5;fontColor=#1b5e20;fontStyle=1;fontSize=10;{FONT}"
    ARROW_GRAPH = f"edgeStyle=straightEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2e7d32;strokeWidth=1.5;{FONT}"

    # =========================================================================
    # 1. PANEL 1: FEATURE EXTRACTION (Top-Left: X=40, Y=40, W=620, H=320)
    # =========================================================================
    add_node("box_p1", "1", "", 40, 40, 620, 320, PANEL_CONTAINER)
    add_node("title_p1", "1", "<b>Feature extraction</b>", 55, 48, 200, 25, PANEL_TITLE)

    # 1A. CLIP Text Stream (Center Y = 95)
    node_txt = add_node("node_txt", "1", "<b>AU Descriptions</b><br><font style=\"font-size:8.5px;color:#666;\">(Text Prompts)</font>", 55, 75, 120, 40, STYLE_LINEAR_BLOCK)
    node_clip = add_node("node_clip", "1", "<b>CLIP</b><br><font style=\"font-size:8.5px;\">Text (❄️)</font>", 215, 72, 90, 46, STYLE_CLIP)
    node_temb = add_node("node_temb", "1", "<b>`\\mathbf{T}_a`</b>", 345, 75, 60, 40, STYLE_CUBE_AU)
    node_lcon = add_node("node_lcon", "1", "<span style=\"color:#b71c1c;\"><b>`L_{con}`</b></span>", 445, 78, 65, 34, STYLE_LOSS_RED)

    add_edge("e_txt_clip", "1", "node_txt", "node_clip", "", ARROW_H)
    add_edge("e_clip_temb", "1", "node_clip", "node_temb", "", ARROW_H)
    add_edge("e_temb_lcon", "1", "node_temb", "node_lcon", "", ARROW_H)

    # 1B. Visual Feature Stream (Center Y = 185)
    img_face = add_node("GjfUm5Ax_Uw3Q_51WUAJ-1", "1", "", 55, 150, 70, 70, image_style)
    node_bb = add_node("node_bb", "1", "<b>Backbone</b><br><font style=\"font-size:8.5px;\">(ResNet-50)</font>", 165, 155, 90, 60, STYLE_BACKBONE)
    node_lp = add_node("node_lp", "1", "<b>Linear</b><br><font style=\"font-size:8.5px;\">`2048 \\to 512`</font>", 290, 160, 80, 50, STYLE_LINEAR_BLOCK)
    node_feat = add_node("node_feat", "1", "<b>Feature map</b><br><font style=\"font-size:8px;\">`49 \\times 512`</font>", 405, 150, 80, 70, STYLE_CUBE_FEAT)
    node_grid = add_node("node_grid", "1", "<b>Spatial grid</b><br><font style=\"font-size:8px;\">`49 \\times 512`</font>", 520, 155, 85, 60, STYLE_CUBE_FEAT)

    add_edge("e_f_bb", "1", "GjfUm5Ax_Uw3Q_51WUAJ-1", "node_bb", "", ARROW_H)
    add_edge("e_bb_lp", "1", "node_bb", "node_lp", "", ARROW_H)
    add_edge("e_lp_feat", "1", "node_lp", "node_feat", "", ARROW_H)
    add_edge("e_feat_grid", "1", "node_feat", "node_grid", "", ARROW_H)

    # 1C. HSIC Disentanglement Losses (Center Y = 280)
    add_node("lbl_hsic", "1", "<b>HSIC Disentanglement:</b>", 55, 245, 200, 18, SUBTITLE)
    
    node_lib = add_node("node_lib", "1", 
                        "<span style=\"color:#b71c1c;\"><b>`L_{ib}`</b></span>: min HSIC(`\\mathbf{V}_a`, `\\mathbf{z}_{\\text{img}}`)", 
                        55, 265, 175, 34, STYLE_LOSS_RED)
    node_lalign = add_node("node_lalign", "1", 
                           "<span style=\"color:#1b5e20;\"><b>`L_{align}`</b></span>: max HSIC(`\\mathbf{V}_a`, `\\mathbf{Y}_a`)", 
                           240, 265, 175, 34, STYLE_LOSS_GREEN)
    node_ldecorr = add_node("node_ldecorr", "1", 
                            "<span style=\"color:#b71c1c;\"><b>`L_{decorr}`</b></span>: min &sum; HSIC(`\\mathbf{V}_a^i`, `\\mathbf{V}_a^j`)", 
                            425, 265, 195, 34, STYLE_LOSS_RED)

    # =========================================================================
    # 2. PANEL 2: MULTI-BRANCH CLASSIFICATION (Right Column: X=690, Y=40, W=570, H=720)
    # =========================================================================
    add_node("box_p2_right", "1", "", 690, 40, 570, 720, PANEL_CONTAINER)
    add_node("title_p2_right", "1", "<b>Multi-branch classification</b>", 705, 48, 300, 25, PANEL_TITLE)

    # Spatial Patches -> Multi-Branch Connection: Center Y = 185.0
    # box_p2_right Y=40, H=720. entryY = (185 - 40) / 720 = 0.2013889
    arrow_grid_multi = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#37474f;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.2013889;{FONT}"
    add_edge("e_grid_multi", "1", "node_grid", "box_p2_right", "Spatial Patches", arrow_grid_multi)

    # --- AU Head (Rows at Center Y = 115, 160, 220) ---
    add_node("lbl_auhead", "1", "<b>AU Head (8 Branches)</b>", 705, 82, 180, 18, SUBTITLE)

    # Row AU1 (Center Y = 115)
    add_node("cnn_au1", "1", "1D CNN<br>(AU1)", 705, 98, 65, 34, STYLE_CNN_AU)
    add_node("va_1", "1", "<b>`\\mathbf{V}_a^{(1)}`</b>", 790, 95, 40, 40, STYLE_CUBE_AU)
    add_node("lin_au1", "1", "Linear", 855, 100, 50, 30, STYLE_LINEAR_AU)
    add_node("pau1", "1", "", 925, 98, 8, 34, STYLE_BAR_AU)

    add_edge("e_c1_v1", "1", "cnn_au1", "va_1", "", ARROW_H)
    add_edge("e_v1_l1", "1", "va_1", "lin_au1", "", ARROW_H)
    add_edge("e_l1_p1", "1", "lin_au1", "pau1", "", ARROW_H)

    # Row AU2 (Center Y = 160)
    add_node("cnn_au2", "1", "1D CNN<br>(AU2)", 705, 143, 65, 34, STYLE_CNN_AU)
    add_node("va_2", "1", "<b>`\\mathbf{V}_a^{(2)}`</b>", 790, 140, 40, 40, STYLE_CUBE_AU)
    add_node("lin_au2", "1", "Linear", 855, 145, 50, 30, STYLE_LINEAR_AU)
    add_node("pau2", "1", "", 925, 143, 8, 34, STYLE_BAR_AU)

    add_edge("e_c2_v2", "1", "cnn_au2", "va_2", "", ARROW_H)
    add_edge("e_v2_l2", "1", "va_2", "lin_au2", "", ARROW_H)
    add_edge("e_l2_p2", "1", "lin_au2", "pau2", "", ARROW_H)

    # Dots
    add_node("dots_au", "1", "<b>&vellip;</b>", 725, 180, 20, 20, PANEL_TITLE)

    # Row AU12 (Center Y = 220)
    add_node("cnn_au12", "1", "1D CNN<br>(AU12)", 705, 203, 65, 34, STYLE_CNN_AU)
    add_node("va_12", "1", "<b>`\\mathbf{V}_a^{(12)}`</b>", 790, 200, 40, 40, STYLE_CUBE_AU)
    add_node("lin_au12", "1", "Linear", 855, 205, 50, 30, STYLE_LINEAR_AU)
    add_node("pau12", "1", "", 925, 203, 8, 34, STYLE_BAR_AU)

    add_edge("e_c12_v12", "1", "cnn_au12", "va_12", "", ARROW_H)
    add_edge("e_v12_l12", "1", "va_12", "lin_au12", "", ARROW_H)
    add_edge("e_l12_p12", "1", "lin_au12", "pau12", "", ARROW_H)

    # AU Loss & Prediction (Center Y = 160, Center X = 1030)
    add_node("lbl_pau", "1", "<b>`P_{AU}`</b>", 945, 150, 45, 20, SUBTITLE)
    node_lwa = add_node("node_lwa", "1", "<span style=\"color:#b71c1c;\"><b>`L_{wa}`</b></span>", 1000, 144, 60, 32, STYLE_LOSS_RED)
    add_edge("e_pau_lwa", "1", "pau2", "node_lwa", "", ARROW_H)

    # Transferred AU Embeddings (Center Y = 160)
    add_node("tag_va_trans", "1", "<b>`\\mathbf{V}_a`</b> `(8 \\times 256)`<br><font style=\"font-size:7.5px;color:#666;\">Transferred (❄️)</font>", 1085, 139, 145, 42, STYLE_FORMULA)

    # --- FACS Prior Matrix M_AE --- (Center X = 1030, Center Y = 250)
    node_mae = add_node("node_mae", "1", "<b>FACS Prior `M_{AE}`</b>", 975, 232, 110, 36, STYLE_FORMULA)
    add_edge("e_lwa_mae", "1", "node_lwa", "node_mae", "", ARROW_DASH_V)

    # --- Emotion Head (Rows at Center Y = 365, 410, 475) ---
    add_node("lbl_exphead", "1", "<b>Expression Head (7 Branches)</b>", 705, 330, 200, 18, SUBTITLE)

    # Row E1 (Center Y = 365)
    add_node("cnn_e1", "1", "1D CNN<br>(E1)", 705, 348, 65, 34, STYLE_CNN_EXP)
    add_node("ve_1", "1", "<b>`\\mathbf{V}_e^{(1)}`</b>", 790, 345, 40, 40, STYLE_CUBE_EXP)
    add_node("lin_e1", "1", "Linear", 855, 350, 50, 30, STYLE_LINEAR_EXP)
    add_node("pe1", "1", "", 925, 348, 8, 34, STYLE_BAR_EXP)

    add_edge("e_ce1_ve1", "1", "cnn_e1", "ve_1", "", ARROW_H)
    add_edge("e_ve1_le1", "1", "ve_1", "lin_e1", "", ARROW_H)
    add_edge("e_le1_pe1", "1", "lin_e1", "pe1", "", ARROW_H)

    # Row E2 (Center Y = 410)
    add_node("cnn_e2", "1", "1D CNN<br>(E2)", 705, 393, 65, 34, STYLE_CNN_EXP)
    add_node("ve_2", "1", "<b>`\\mathbf{V}_e^{(2)}`</b>", 790, 390, 40, 40, STYLE_CUBE_EXP)
    add_node("lin_e2", "1", "Linear", 855, 395, 50, 30, STYLE_LINEAR_EXP)
    add_node("pe2", "1", "", 925, 393, 8, 34, STYLE_BAR_EXP)

    add_edge("e_ce2_ve2", "1", "cnn_e2", "ve_2", "", ARROW_H)
    add_edge("e_ve2_le2", "1", "ve_2", "lin_e2", "", ARROW_H)
    add_edge("e_le2_pe2", "1", "lin_e2", "pe2", "", ARROW_H)

    # Dots
    add_node("dots_exp", "1", "<b>&vellip;</b>", 725, 435, 20, 20, PANEL_TITLE)

    # Row E7 (Center Y = 475)
    add_node("cnn_e7", "1", "1D CNN<br>(E7)", 705, 458, 65, 34, STYLE_CNN_EXP)
    add_node("ve_7", "1", "<b>`\\mathbf{V}_e^{(7)}`</b>", 790, 455, 40, 40, STYLE_CUBE_EXP)
    add_node("lin_e7", "1", "Linear", 855, 460, 50, 30, STYLE_LINEAR_EXP)
    add_node("pe7", "1", "", 925, 458, 8, 34, STYLE_BAR_EXP)

    add_edge("e_ce7_ve7", "1", "cnn_e7", "ve_7", "", ARROW_H)
    add_edge("e_ve7_le7", "1", "ve_7", "lin_e7", "", ARROW_H)
    add_edge("e_le7_pe7", "1", "lin_e7", "pe7", "", ARROW_H)

    # Emotion Loss & Prediction (Center Y = 410, Center X = 1030)
    add_node("lbl_pe", "1", "<b>`P_E`</b>", 945, 400, 45, 20, SUBTITLE)
    node_lwe = add_node("node_lwe", "1", "<span style=\"color:#b71c1c;\"><b>`L_{we}`</b></span>", 1000, 394, 60, 32, STYLE_LOSS_RED)
    add_edge("e_pe_lwe", "1", "pe2", "node_lwe", "", ARROW_H)
    add_edge("e_mae_lwe", "1", "node_mae", "node_lwe", "", ARROW_DASH_V)

    # Transferred Emotion Embeddings (Center Y = 410)
    add_node("tag_ve_trans", "1", "<b>`\\mathbf{V}_e`</b> `(7 \\times 256)`<br><font style=\"font-size:7.5px;color:#666;\">Transferred (❄️)</font>", 1085, 389, 145, 42, STYLE_FORMULA)

    # Phase 1 Loss Formula Tag (Bottom of Right Container)
    add_node("formula_p1", "1", 
             "<b>Phase 1 Loss:</b> `L_{\\text{phase1}} = L_{wa} + \\gamma L_{we} + \\lambda_{ib} L_{ib} + \\lambda_{align} L_{align} + \\lambda_{decorr} L_{decorr} + \\lambda_{con} L_{con}`", 
             705, 660, 540, 36, STYLE_FORMULA)

    # =========================================================================
    # 3. PANEL 3: CAUSAL DISCOVERY AND REASONING (Bottom-Left: X=40, Y=390, W=620, H=370)
    # =========================================================================
    add_node("box_p3", "1", "", 40, 390, 620, 370, PANEL_CONTAINER)
    add_node("title_p3", "1", 
        "<b>Causal discovery and reasoning (Phase 2)</b> &nbsp;&nbsp; "
        "<span style=\"background-color:#ffebee;color:#c62828;padding:1px 6px;border-radius:3px;font-size:9.5px;border:1px solid #ef9a9a;\"><b>❄️ Frozen Backbone &amp; Heads</b></span>", 
        55, 398, 590, 25, PANEL_TITLE)

    # --- 3A. Level 1: AU-AU Causal Graph (Left side of Panel 3: X=55 to 310) ---
    add_node("sub_lvl1", "1", "<b>Level 1: AU-AU Causal Graph</b><br><font style=\"font-size:8px;color:#666;\">Input: `\\mathbf{V}_a` (❄️) &rarr; learns `\\mathbf{A}_{\\text{AU-AU}}`</font>", 55, 430, 240, 24, SUBTITLE)

    # AU Graph Nodes (Center Y = 485 for row 1, 535 for row 2)
    gau_1 = add_node("gau_1", "1", "AU1", 60, 471, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_2 = add_node("gau_2", "1", "AU2", 115, 471, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_4 = add_node("gau_4", "1", "AU4", 170, 471, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_6 = add_node("gau_6", "1", "AU6", 60, 521, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_12 = add_node("gau_12", "1", "AU12", 115, 521, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_25 = add_node("gau_25", "1", "AU25", 170, 521, 28, 28, STYLE_GRAPH_NODE_AU)

    add_edge("eg_1_2", "1", "gau_1", "gau_2", "", ARROW_GRAPH)
    add_edge("eg_2_4", "1", "gau_2", "gau_4", "", ARROW_GRAPH)
    add_edge("eg_6_12", "1", "gau_6", "gau_12", "", ARROW_GRAPH)
    add_edge("eg_12_25", "1", "gau_12", "gau_25", "", ARROW_GRAPH)
    add_edge("eg_2_12", "1", "gau_2", "gau_12", "", ARROW_GRAPH)

    # GAT and Losses in AU-AU (All aligned at Center X = 265.0)
    # gau_4 center Y = 485. node_gat1 center Y = 467 + 18 = 485.
    node_gat1 = add_node("node_gat1", "1", "<b>GAT layer</b>", 225, 467, 80, 36, STYLE_LINEAR_BLOCK)
    node_ldag = add_node("node_ldag", "1", "<span style=\"color:#b71c1c;\"><b>`L_{DAG}`</b></span>", 230, 515, 70, 28, STYLE_LOSS_RED)
    node_lcau1 = add_node("node_lcau1", "1", "<span style=\"color:#b71c1c;\"><b>`L_{causal}^{AU}`</b></span>", 225, 553, 80, 28, STYLE_LOSS_RED)
    node_pauau = add_node("node_pauau", "1", "<b>`P_{AU-AU}`</b> &rarr; <span style=\"color:#b71c1c;\"><b>`L_{au\\_au}`</b></span>", 205, 591, 120, 32, STYLE_FORMULA)

    add_edge("eg_4_gat1", "1", "gau_4", "node_gat1", "", ARROW_H)
    add_edge("e_gat1_dag", "1", "node_gat1", "node_ldag", "", ARROW_V)
    add_edge("e_ldag_lcau", "1", "node_ldag", "node_lcau1", "", ARROW_V)
    add_edge("e_lcau_pauau", "1", "node_lcau1", "node_pauau", "", ARROW_V)

    # Updated AU Representation node (Center Y = 660)
    node_up_au = add_node("node_up_au", "1", "<b>Updated AU:</b> `\\mathbf{V}_a^* \\in \\mathbb{R}^{8 \\times 256}`", 55, 642, 250, 36, STYLE_CUBE_GREEN)

    # --- 3B. Level 2: AU -> Expression Graph (Right side of Panel 3: X=335 to 640) ---
    add_node("sub_lvl2", "1", "<b>Level 2: AU &rarr; Expression Graph</b><br><font style=\"font-size:8px;color:#666;\">Input: `\\mathbf{V}_a^*` &amp; `\\mathbf{V}_e` (❄️) &rarr; learns `\\mathbf{A}_{\\text{AU-Exp}}`</font>", 335, 430, 280, 24, SUBTITLE)

    # Cascade Arrow from Level 1 to Level 2 (Center Y = 660.0)
    add_edge("e_cascade_1_2", "1", "node_up_au", "node_mimp", "Cascade: `\\mathbf{V}_a^*`", ARROW_CASCADE)

    # Bipartite Network Nodes
    bau_6 = add_node("bau_6", "1", "AU6", 335, 470, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_12 = add_node("bau_12", "1", "AU12", 335, 505, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_4 = add_node("bau_4", "1", "AU4", 335, 540, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_1 = add_node("bau_1", "1", "AU1", 335, 575, 26, 26, STYLE_GRAPH_NODE_AU)

    bemo_hap = add_node("bemo_hap", "1", "Happy", 415, 471, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_sad = add_node("bemo_sad", "1", "Sad", 415, 515, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_fea = add_node("bemo_fea", "1", "Fear", 415, 559, 28, 28, STYLE_GRAPH_NODE_EXP)

    add_edge("eb_6_h", "1", "bau_6", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_12_h", "1", "bau_12", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_4_s", "1", "bau_4", "bemo_sad", "", ARROW_GRAPH)
    add_edge("eb_1_f", "1", "bau_1", "bemo_fea", "", ARROW_GRAPH)

    # Bipartite GAT and Losses (Center X = 535)
    # bemo_hap center Y = 485. node_gat2 center Y = 467 + 18 = 485.
    node_gat2 = add_node("node_gat2", "1", "<b>Bipartite GAT</b>", 485, 467, 100, 36, STYLE_LINEAR_BLOCK)
    node_lcau2 = add_node("node_lcau2", "1", "<span style=\"color:#b71c1c;\"><b>`L_{causal}^{Exp}`</b></span>", 495, 515, 80, 28, STYLE_LOSS_RED)
    node_lfacs = add_node("node_lfacs", "1", "<span style=\"color:#b71c1c;\"><b>`L_{facs}^{Exp}`</b></span>", 495, 553, 80, 28, STYLE_LOSS_RED)
    node_pfinal = add_node("node_pfinal", "1", "Final <b>`P_G`</b> &rarr; <span style=\"color:#b71c1c;\"><b>`L_{wa}^{AU}, L_{we}^{Exp}`</b></span>", 475, 591, 120, 32, STYLE_FORMULA)

    add_edge("eb_hap_gat2", "1", "bemo_hap", "node_gat2", "", ARROW_H)
    add_edge("e_gat2_cau", "1", "node_gat2", "node_lcau2", "", ARROW_V)
    add_edge("e_lcau_facs", "1", "node_lcau2", "node_lfacs", "", ARROW_V)
    add_edge("e_lfacs_pfinal", "1", "node_lfacs", "node_pfinal", "", ARROW_V)

    # Causal Importance Mask node (Center Y = 660)
    node_mimp = add_node("node_mimp", "1", "<b>Causal Importance Mask:</b> `\\mathbf{M}_{\\text{imp}} \\in \\mathbb{R}^{8 \\times 8}`", 335, 642, 305, 36, STYLE_CUBE_YELLOW)

    # --- 3C. HiMod Counterfactual Reasoning (Bottom of Panel 3, Y = 698) ---
    add_node("cf_imp_badge", "1", "<b>Perturb Important:</b> `\\mathbf{V}_a^* + \\epsilon \\mathbf{M}_{\\text{imp}}` &rarr; <span style=\"color:#b71c1c;\"><b>`L_{cf}^{imp}`</b></span>", 55, 698, 260, 36, STYLE_FORMULA)
    add_node("cf_unimp_badge", "1", "<b>Perturb Unimportant:</b> `\\mathbf{V}_a^* + \\epsilon (1-\\mathbf{M}_{\\text{imp}})` &rarr; <span style=\"color:#1b5e20;\"><b>`L_{cf}^{unimp}`</b></span>", 335, 698, 305, 36, STYLE_FORMULA)

    # =========================================================================
    # BUILD XML
    # =========================================================================
    mxfile = ET.Element('mxfile', host='app.diagrams.net')
    diagram = ET.SubElement(mxfile, 'diagram', name='CtrlAU Architecture', id='cQhNqvIMsQzl_Nc4XbVt')
    graph_model = ET.SubElement(diagram, 'mxGraphModel', 
                                grid='0', page='1', gridSize='10', 
                                guides='1', tooltips='1', connect='1', arrows='1', fold='1', pageScale='1', 
                                pageWidth='1350', pageHeight='820', math='1', shadow='0')
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
        
    print(f"Successfully generated authentic user-style ctrlau.drawio with {len(cells)} cells!")

if __name__ == '__main__':
    build_user_style_diagram()
