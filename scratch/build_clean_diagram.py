import xml.etree.ElementTree as ET

def build_clean_diagram():
    # 1. Extract base64 image style from backup
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
    # STYLES (Clean, Academic, Consistent)
    # =========================================================================
    FONT = "fontFamily=Helvetica;"
    BOX_P1 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=#b0bec5;strokeWidth=2;arcSize=4;{FONT}"
    BOX_P2 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f8e9;strokeColor=#81c784;strokeWidth=2;arcSize=4;{FONT}"
    
    CARD_WHITE = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#cfd8dc;strokeWidth=1.5;arcSize=6;{FONT}"
    CARD_YELLOW = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fffdf7;strokeColor=#ffe082;strokeWidth=1.5;arcSize=6;{FONT}"
    CARD_PURPLE = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#faf5ff;strokeColor=#d1c4e9;strokeWidth=1.5;arcSize=6;{FONT}"
    CARD_GREEN = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f9fbe7;strokeColor=#c5e1a5;strokeWidth=1.5;arcSize=6;{FONT}"

    TITLE_TEXT = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontStyle=1;fontSize=13;fontColor=#263238;{FONT}"
    BANNER_P1 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#37474f;strokeColor=none;fontColor=#ffffff;align=left;spacingLeft=15;fontStyle=1;fontSize=13;{FONT}"
    BANNER_P2 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#2e7d32;strokeColor=none;fontColor=#ffffff;align=left;spacingLeft=15;fontStyle=1;fontSize=13;{FONT}"

    STYLE_BACKBONE = f"shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=12;fillColor=#e3f2fd;strokeColor=#1976d2;strokeWidth=1.5;fontColor=#0d47a1;fontSize=11;fontStyle=1;{FONT}"
    STYLE_CUBE_AU = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.2;fontColor=#4a148c;fontSize=10;fontStyle=1;{FONT}"
    STYLE_CUBE_EXP = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#fce4ec;strokeColor=#c2185b;strokeWidth=1.2;fontColor=#880e4f;fontSize=10;fontStyle=1;{FONT}"
    STYLE_CUBE_FEAT = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#eceff1;strokeColor=#607d8b;strokeWidth=1.2;fontColor=#263238;fontSize=10;fontStyle=1;{FONT}"

    STYLE_CNN_AU = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.2;fontColor=#311b92;fontSize=10;fontStyle=1;{FONT}"
    STYLE_CNN_EXP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fce4ec;strokeColor=#e91e63;strokeWidth=1.2;fontColor=#880e4f;fontSize=10;fontStyle=1;{FONT}"
    STYLE_LINEAR = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#9e9e9e;strokeWidth=1.2;fontColor=#212121;fontSize=10;{FONT}"
    STYLE_CLIP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8bbd0;strokeColor=#c2185b;strokeWidth=1.5;fontColor=#880e4f;fontSize=11;fontStyle=1;{FONT}"

    STYLE_BAR_AU = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ab47bc;strokeColor=#6a1b9a;strokeWidth=1;{FONT}"
    STYLE_BAR_EXP = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ec407a;strokeColor=#ad1457;strokeWidth=1;{FONT}"

    STYLE_LOSS_RED = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#d32f2f;strokeWidth=1.5;fontColor=#b71c1c;fontSize=10;{FONT}"
    STYLE_LOSS_GREEN = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#388e3c;strokeWidth=1.5;fontColor=#1b5e20;fontSize=10;{FONT}"

    STYLE_GRAPH_NODE_AU = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.5;fontColor=#4a148c;fontStyle=1;fontSize=10;{FONT}"
    STYLE_GRAPH_NODE_EXP = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#fce4ec;strokeColor=#d81b60;strokeWidth=1.5;fontColor=#880e4f;fontStyle=1;fontSize=10;{FONT}"

    # Precision directional arrows (NO bend, laser straight!)
    ARROW_H = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#455a64;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;{FONT}"
    ARROW_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#455a64;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_DASH_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#78909c;strokeWidth=1.2;dashed=1;dashPattern=4 3;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_GRAPH = f"edgeStyle=straightEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2e7d32;strokeWidth=1.5;{FONT}"

    # =========================================================================
    # 1. TOP CONTAINER: PHASE 1 (Feature Extraction & Multi-Branch Heads)
    # =========================================================================
    # Container 1 (Left): Feature Extraction & Disentanglement
    add_node("box_p1_left", "1", "", 50, 40, 580, 430, BOX_P1)
    add_node("banner_p1_left", "1", "<b>Feature Extraction &amp; Representation Disentanglement</b>", 60, 48, 560, 28, BANNER_P1)

    # 1A. Visual Feature Extraction (Center Y = 120)
    img_face = add_node("GjfUm5Ax_Uw3Q_51WUAJ-1", "1", "", 75, 85, 70, 70, image_style)
    node_bb = add_node("node_bb", "1", "<b>ResNet-50</b><br><font style=\"font-size:9px;\">Backbone</font>", 180, 90, 85, 60, STYLE_BACKBONE)
    node_lp = add_node("node_lp", "1", "<b>LinearBlock</b><br><font style=\"font-size:9px;\">2048 &rarr; 512</font>", 300, 95, 80, 50, STYLE_LINEAR)
    node_feat = add_node("node_feat", "1", "<b>Feature Map</b><br><font style=\"font-size:9px;\">49&times;512</font>", 415, 90, 75, 60, STYLE_CUBE_FEAT)

    add_edge("e_f_bb", "1", "GjfUm5Ax_Uw3Q_51WUAJ-1", "node_bb", "", ARROW_H)
    add_edge("e_bb_lp", "1", "node_bb", "node_lp", "", ARROW_H)
    add_edge("e_lp_feat", "1", "node_lp", "node_feat", "", ARROW_H)

    # 1B. Vision-Language Alignment (CLIP) (Center Y = 195)
    node_txt = add_node("node_txt", "1", "<b>AU &amp; Expr Text</b><br><font style=\"font-size:9px;\">FACS descriptions</font>", 75, 175, 110, 40, STYLE_LINEAR)
    node_clip = add_node("node_clip", "1", "<b>CLIP</b><br><font style=\"font-size:9px;\">Text Encoder (❄️)</font>", 220, 172, 90, 46, STYLE_CLIP)
    node_temb = add_node("node_temb", "1", "<b>Text Emb</b><br><font style=\"font-size:9px;\">T_a, T_e</font>", 345, 175, 65, 40, STYLE_CUBE_EXP)
    node_lcon = add_node("node_lcon", "1", "<b style=\"color:#d32f2f;\">L_{con}</b> (InfoNCE)<br><font style=\"font-size:8px;\">Active samples (y=1)</font>", 445, 172, 105, 46, STYLE_LOSS_RED)

    add_edge("e_txt_clip", "1", "node_txt", "node_clip", "", ARROW_H)
    add_edge("e_clip_temb", "1", "node_clip", "node_temb", "", ARROW_H)
    add_edge("e_temb_lcon", "1", "node_temb", "node_lcon", "", ARROW_H)

    # 1C. HSIC Information Bottleneck & Disentanglement (Compact Unified Block)
    # node_lp center X is 300 + 40 = 340. box_hsic center X is 75 + 530/2 = 340.
    add_node("box_hsic", "1", 
        "<b>HSIC Disentanglement &amp; Information Bottleneck Module</b><br>"
        "<div style=\"text-align:left;padding:6px 12px;font-size:10px;line-height:1.6;\">"
        "&bull; <b style=\"color:#d32f2f;\">L_{ib}</b> = HSIC(V_a, z_{img}) &nbsp;&mdash;&nbsp; <i>Compresses identity &amp; background noise</i><br>"
        "&bull; <b style=\"color:#2e7d32;\">L_{align}</b> = -HSIC(V_a, Y_a) &nbsp;&mdash;&nbsp; <i>Maximizes mutual information with AU labels</i><br>"
        "&bull; <b style=\"color:#d32f2f;\">L_{decorr}</b> = &sum;_{i&ne;j} HSIC(V_a^i, V_a^j) &nbsp;&mdash;&nbsp; <i>Eliminates cross-AU feature collapse</i>"
        "</div>"
        "<div style=\"background:#fff9c4;padding:4px;border-radius:3px;font-size:9.5px;color:#f57f17;margin-top:2px;\">"
        "<b>L_{phase1} = L_{wa} + 0.05 L_{we} + &lambda;_{ib}L_{ib} + &lambda;_{align}L_{align} + &lambda;_{decorr}L_{decorr} + &lambda;_{con}L_{con}</b>"
        "</div>", 
        75, 245, 530, 135, CARD_PURPLE)

    # Clean vertical connection from LinearBlock down to HSIC (diff_x = 0)
    add_edge("e_lp_hsic", "1", "node_lp", "box_hsic", "", ARROW_DASH_V)

    # -------------------------------------------------------------------------
    # Container 2 (Right): Multi-Branch Classification & FACS Prior
    # -------------------------------------------------------------------------
    add_node("box_p1_right", "1", "", 650, 40, 630, 430, BOX_P1)
    add_node("banner_p1_right", "1", "<b>Multi-Branch Spatial Classification &amp; FACS Prior</b>", 660, 48, 610, 28, BANNER_P1)

    # Feature Map -> Box connection: enter exactly at y = 120 (40 + 430*0.1860465 = 120.0)
    arrow_feat_h = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#455a64;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.1860465;{FONT}"
    add_edge("e_feat_multi", "1", "node_feat", "box_p1_right", "Feature Map", arrow_feat_h)

    # --- AU Head (Rows at center Y = 115, 158, 215) ---
    add_node("lbl_auhead", "1", "<b>AU Head (8 Branches)</b>", 670, 85, 150, 20, TITLE_TEXT)

    # Row AU1 (Center Y = 115)
    add_node("cnn_au1", "1", "1D CNN<br>(AU1)", 675, 98, 65, 34, STYLE_CNN_AU)
    add_node("va_1", "1", "V_a^{(1)}", 760, 96, 38, 38, STYLE_CUBE_AU)
    add_node("lin_au1", "1", "Linear", 820, 100, 50, 30, STYLE_LINEAR)
    add_node("pau1", "1", "", 890, 98, 10, 34, STYLE_BAR_AU)

    add_edge("e_c1_v1", "1", "cnn_au1", "va_1", "", ARROW_H)
    add_edge("e_v1_l1", "1", "va_1", "lin_au1", "", ARROW_H)
    add_edge("e_l1_p1", "1", "lin_au1", "pau1", "", ARROW_H)

    # Row AU2 (Center Y = 158)
    add_node("cnn_au2", "1", "1D CNN<br>(AU2)", 675, 141, 65, 34, STYLE_CNN_AU)
    add_node("va_2", "1", "V_a^{(2)}", 760, 139, 38, 38, STYLE_CUBE_AU)
    add_node("lin_au2", "1", "Linear", 820, 143, 50, 30, STYLE_LINEAR)
    add_node("pau2", "1", "", 890, 141, 10, 34, STYLE_BAR_AU)

    add_edge("e_c2_v2", "1", "cnn_au2", "va_2", "", ARROW_H)
    add_edge("e_v2_l2", "1", "va_2", "lin_au2", "", ARROW_H)
    add_edge("e_l2_p2", "1", "lin_au2", "pau2", "", ARROW_H)

    # Dots
    add_node("dots_au", "1", "<b>&vellip;</b>", 700, 178, 20, 20, TITLE_TEXT)

    # Row AU12 (Center Y = 215)
    add_node("cnn_au12", "1", "1D CNN<br>(AU12)", 675, 198, 65, 34, STYLE_CNN_AU)
    add_node("va_12", "1", "V_a^{(12)}", 760, 196, 38, 38, STYLE_CUBE_AU)
    add_node("lin_au12", "1", "Linear", 820, 200, 50, 30, STYLE_LINEAR)
    add_node("pau12", "1", "", 890, 198, 10, 34, STYLE_BAR_AU)

    add_edge("e_c12_v12", "1", "cnn_au12", "va_12", "", ARROW_H)
    add_edge("e_v12_l12", "1", "va_12", "lin_au12", "", ARROW_H)
    add_edge("e_l12_p12", "1", "lin_au12", "pau12", "", ARROW_H)

    # AU Loss & Prediction (Center Y = 158, Center X = 1030)
    add_node("lbl_pau", "1", "<b>P_{AU}</b>", 910, 148, 35, 20, TITLE_TEXT)
    node_lwa = add_node("node_lwa", "1", "<b style=\"color:#d32f2f;font-size:12px;\">L_{wa}</b><br><font style=\"font-size:8.5px;\">Weighted Asymmetric Loss</font>", 955, 137, 150, 42, STYLE_LOSS_RED)
    add_edge("e_pau_lwa", "1", "pau2", "node_lwa", "", ARROW_H)

    # --- FACS Prior Matrix M_AE --- (Center X = 1030, Center Y = 238)
    node_mae = add_node("node_mae", "1", "<b>FACS Prior Matrix M_{AE}</b><br><font style=\"font-size:9px;\">Y_e = argmax(Y_a &middot; M_{AE})</font>", 955, 215, 150, 46, CARD_YELLOW)
    add_edge("e_lwa_mae", "1", "node_lwa", "node_mae", "", ARROW_DASH_V)

    # --- Emotion Head (Rows at center Y = 295, 338, 398) ---
    add_node("lbl_exphead", "1", "<b>Expression Head (7 Branches)</b>", 670, 260, 180, 20, TITLE_TEXT)

    # Row E1 (Center Y = 295)
    add_node("cnn_e1", "1", "1D CNN<br>(E1: Angry)", 675, 278, 75, 34, STYLE_CNN_EXP)
    add_node("ve_1", "1", "V_e^{(1)}", 765, 276, 38, 38, STYLE_CUBE_EXP)
    add_node("lin_e1", "1", "Linear", 820, 280, 50, 30, STYLE_LINEAR)
    add_node("pe1", "1", "", 890, 278, 10, 34, STYLE_BAR_EXP)

    add_edge("e_ce1_ve1", "1", "cnn_e1", "ve_1", "", ARROW_H)
    add_edge("e_ve1_le1", "1", "ve_1", "lin_e1", "", ARROW_H)
    add_edge("e_le1_pe1", "1", "lin_e1", "pe1", "", ARROW_H)

    # Row E2 (Center Y = 338)
    add_node("cnn_e2", "1", "1D CNN<br>(E2: Fear)", 675, 321, 75, 34, STYLE_CNN_EXP)
    add_node("ve_2", "1", "V_e^{(2)}", 765, 319, 38, 38, STYLE_CUBE_EXP)
    add_node("lin_e2", "1", "Linear", 820, 323, 50, 30, STYLE_LINEAR)
    add_node("pe2", "1", "", 890, 321, 10, 34, STYLE_BAR_EXP)

    add_edge("e_ce2_ve2", "1", "cnn_e2", "ve_2", "", ARROW_H)
    add_edge("e_ve2_le2", "1", "ve_2", "lin_e2", "", ARROW_H)
    add_edge("e_le2_pe2", "1", "lin_e2", "pe2", "", ARROW_H)

    # Dots
    add_node("dots_exp", "1", "<b>&vellip;</b>", 705, 358, 20, 20, TITLE_TEXT)

    # Row E7 (Center Y = 398)
    add_node("cnn_e7", "1", "1D CNN<br>(E7: Neutral)", 675, 381, 75, 34, STYLE_CNN_EXP)
    add_node("ve_7", "1", "V_e^{(7)}", 765, 379, 38, 38, STYLE_CUBE_EXP)
    add_node("lin_e7", "1", "Linear", 820, 383, 50, 30, STYLE_LINEAR)
    add_node("pe7", "1", "", 890, 381, 10, 34, STYLE_BAR_EXP)

    add_edge("e_ce7_ve7", "1", "cnn_e7", "ve_7", "", ARROW_H)
    add_edge("e_ve7_le7", "1", "ve_7", "lin_e7", "", ARROW_H)
    add_edge("e_le7_pe7", "1", "lin_e7", "pe7", "", ARROW_H)

    # Emotion Loss & Prediction (Center Y = 338, Center X = 1030)
    add_node("lbl_pe", "1", "<b>P_E</b>", 910, 328, 35, 20, TITLE_TEXT)
    node_lwe = add_node("node_lwe", "1", "<b style=\"color:#d32f2f;font-size:12px;\">L_{we}</b><br><font style=\"font-size:8.5px;\">Expression BCE Loss</font>", 955, 317, 150, 42, STYLE_LOSS_RED)
    add_edge("e_pe_lwe", "1", "pe2", "node_lwe", "", ARROW_H)
    add_edge("e_mae_lwe", "1", "node_mae", "node_lwe", "", ARROW_DASH_V)

    # =========================================================================
    # 2. BOTTOM CONTAINER: PHASE 2 (Two-Level Causal Graph & Counterfactual)
    # =========================================================================
    add_node("box_phase2", "1", "", 50, 490, 1230, 330, BOX_P2)
    add_node("banner_phase2", "1", 
        "<b>PHASE 2: TWO-LEVEL CAUSAL GRAPH DISCOVERY &amp; COUNTERFACTUAL REASONING</b> &nbsp;&nbsp;&nbsp;&nbsp; "
        "<span style=\"background-color:#ffebee;color:#c62828;padding:2px 8px;border-radius:3px;font-size:10px;border:1px solid #ef9a9a;\"><b>❄️ Frozen Representations: V_a, V_e</b></span>",
        60, 498, 1210, 28, BANNER_P2)

    # --- Module 1: AU-AU Causal Graph (Left) ---
    add_node("box_auau", "1", "", 70, 538, 355, 265, CARD_GREEN)
    add_node("title_auau", "1", "<b>Level 1: AU-AU Causal Graph</b>", 80, 545, 250, 20, TITLE_TEXT)

    # Compact Graph Visualization
    gau_1 = add_node("gau_1", "1", "AU1", 85, 580, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_2 = add_node("gau_2", "1", "AU2", 140, 580, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_4 = add_node("gau_4", "1", "AU4", 195, 580, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_6 = add_node("gau_6", "1", "AU6", 85, 630, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_12 = add_node("gau_12", "1", "AU12", 140, 630, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_25 = add_node("gau_25", "1", "AU25", 195, 630, 28, 28, STYLE_GRAPH_NODE_AU)

    add_edge("eg_1_2", "1", "gau_1", "gau_2", "", ARROW_GRAPH)
    add_edge("eg_2_4", "1", "gau_2", "gau_4", "", ARROW_GRAPH)
    add_edge("eg_6_12", "1", "gau_6", "gau_12", "", ARROW_GRAPH)
    add_edge("eg_12_25", "1", "gau_12", "gau_25", "", ARROW_GRAPH)
    add_edge("eg_2_12", "1", "gau_2", "gau_12", "", ARROW_GRAPH)

    # GAT and Losses in AU-AU (All aligned at Center X = 310)
    # gau_4 center Y = 594. node_gat1 center Y = 574 + 20 = 594.
    node_gat1 = add_node("node_gat1", "1", "<b>GAT Layer</b><br><font style=\"font-size:9px;\">Learns A_{AU-AU}</font>", 265, 574, 90, 40, CARD_WHITE)
    node_ldag = add_node("node_ldag", "1", "<b style=\"color:#d32f2f;\">L_{DAG}</b><br><font style=\"font-size:8px;\">tr(e^{A&comp;A}) - d = 0</font>", 265, 624, 90, 36, STYLE_LOSS_RED)
    node_lcau1 = add_node("node_lcau1", "1", "<b style=\"color:#d32f2f;\">L_{causal}^{AU}</b><br><font style=\"font-size:8px;\">Polarity &amp; Imp</font>", 265, 670, 90, 36, STYLE_LOSS_RED)
    node_pauau = add_node("node_pauau", "1", "Classifiers &rarr; <b>P_{AU-AU}</b><br><b style=\"color:#d32f2f;\">L_{au_au}</b> (WAL)", 225, 724, 170, 42, CARD_YELLOW)

    add_edge("eg_4_gat1", "1", "gau_4", "node_gat1", "", ARROW_H)
    add_edge("e_gat1_dag", "1", "node_gat1", "node_ldag", "", ARROW_V)
    add_edge("e_ldag_lcau", "1", "node_ldag", "node_lcau1", "", ARROW_V)
    add_edge("e_lcau_pauau", "1", "node_lcau1", "node_pauau", "", ARROW_V)

    # --- Module 2: AU-Expression Bipartite Graph (Middle) ---
    add_node("box_auexp", "1", "", 480, 538, 375, 265, CARD_YELLOW)
    add_node("title_auexp", "1", "<b>Level 2: AU &rarr; Expression Graph</b>", 490, 545, 250, 20, TITLE_TEXT)

    # Clean Cascade arrow from AU-AU to AU-EXP: 100% straight horizontal line at Y = 670.5
    add_edge("e_cascade_1_2", "1", "box_auau", "box_auexp", "updated_au", ARROW_H)

    # Bipartite Graph
    bau_6 = add_node("bau_6", "1", "AU6", 495, 585, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_12 = add_node("bau_12", "1", "AU12", 495, 620, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_4 = add_node("bau_4", "1", "AU4", 495, 655, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_1 = add_node("bau_1", "1", "AU1", 495, 690, 26, 26, STYLE_GRAPH_NODE_AU)

    bemo_hap = add_node("bemo_hap", "1", "Happy", 575, 595, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_sad = add_node("bemo_sad", "1", "Sad", 575, 635, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_fea = add_node("bemo_fea", "1", "Fear", 575, 675, 28, 28, STYLE_GRAPH_NODE_EXP)

    add_edge("eb_6_h", "1", "bau_6", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_12_h", "1", "bau_12", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_4_s", "1", "bau_4", "bemo_sad", "", ARROW_GRAPH)
    add_edge("eb_1_f", "1", "bau_1", "bemo_fea", "", ARROW_GRAPH)

    # Bipartite GAT and Losses (All aligned at Center X = 735)
    # bemo_hap center Y = 609. node_gat2 center Y = 589 + 20 = 609.
    node_gat2 = add_node("node_gat2", "1", "<b>Bipartite GAT</b><br><font style=\"font-size:9px;\">Learns A_{AU-Exp}</font>", 685, 589, 100, 40, CARD_WHITE)
    node_lcau2 = add_node("node_lcau2", "1", "<b style=\"color:#d32f2f;\">L_{causal}^{Exp}</b><br><font style=\"font-size:8px;\">Causal activation</font>", 685, 639, 100, 36, STYLE_LOSS_RED)
    node_lfacs = add_node("node_lfacs", "1", "<b style=\"color:#d32f2f;\">L_{facs}^{Exp}</b><br><font style=\"font-size:8px;\">FACS consistency</font>", 685, 685, 100, 36, STYLE_LOSS_RED)
    node_pfinal = add_node("node_pfinal", "1", "Final: <b>P_G^{AU}, P_G^{Exp}</b><br><b style=\"color:#d32f2f;\">L_{wa}^{AU}, L_{we}^{Exp}</b>", 635, 731, 200, 42, CARD_PURPLE)

    add_edge("eb_hap_gat2", "1", "bemo_hap", "node_gat2", "", ARROW_H)
    add_edge("e_gat2_cau", "1", "node_gat2", "node_lcau2", "", ARROW_V)
    add_edge("e_lcau_facs", "1", "node_lcau2", "node_lfacs", "", ARROW_V)
    add_edge("e_lfacs_pfinal", "1", "node_lfacs", "node_pfinal", "", ARROW_V)

    # --- Module 3: HiMod Counterfactual Intervention (Right) ---
    add_node("box_himod", "1", "", 905, 538, 360, 265, CARD_WHITE)
    add_node("title_himod", "1", "<b>HiMod Counterfactual Intervention</b>", 915, 545, 260, 20, TITLE_TEXT)

    # Straight horizontal connection from Level 2 to HiMod at Y = 670.5
    add_edge("e_cascade_2_3", "1", "box_auexp", "box_himod", "M_{imp}", ARROW_H)

    # Partition and Branches
    add_node("node_cf_part", "1", "<b>Causal Importance Partition:</b> diag(M_{imp}) &gt; &tau;", 920, 575, 330, 30, CARD_YELLOW)
    
    # Important (Center Y = 636)
    add_node("node_cf_imp", "1", "<b>Perturb Important:</b><br>V_a + &epsilon;&middot;M_{imp}", 920, 618, 195, 36, CARD_PURPLE)
    add_node("node_lcf_imp", "1", "<b style=\"color:#d32f2f;\">L_{cf}^{imp}</b> (Invariance)", 1130, 618, 120, 36, STYLE_LOSS_RED)
    add_edge("e_imp_loss", "1", "node_cf_imp", "node_lcf_imp", "", ARROW_H)

    # Unimportant (Center Y = 686)
    add_node("node_cf_unimp", "1", "<b>Perturb Unimportant:</b><br>V_a + &epsilon;&middot;(1-M_{imp})", 920, 668, 195, 36, CARD_GREEN)
    add_node("node_lcf_unimp", "1", "<b style=\"color:#2e7d32;\">L_{cf}^{unimp}</b> (Consistency)", 1130, 668, 120, 36, STYLE_LOSS_GREEN)
    add_edge("e_unimp_loss", "1", "node_cf_unimp", "node_lcf_unimp", "", ARROW_H)

    # Phase 2 Total loss formula
    add_node("node_p2_loss", "1", 
        "<b>L_{phase2} = L_{wa} + &gamma;L_{we} + &lambda;_{dag}L_{DAG} + &lambda;_{cau}L_{causal} + &lambda;_{facs}L_{facs}^{Exp} + &lambda;_{cf}L_{cf}</b>", 
        920, 724, 330, 42, CARD_YELLOW)

    # =========================================================================
    # BUILD XML
    # =========================================================================
    mxfile = ET.Element('mxfile', host='app.diagrams.net')
    diagram = ET.SubElement(mxfile, 'diagram', name='CtrlAU Architecture', id='cQhNqvIMsQzl_Nc4XbVt')
    graph_model = ET.SubElement(diagram, 'mxGraphModel', 
                                grid='0', page='1', gridSize='10', 
                                guides='1', tooltips='1', connect='1', arrows='1', fold='1', pageScale='1', 
                                pageWidth='1400', pageHeight='900', math='1', shadow='0')
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
        
    print(f"Successfully generated clean ctrlau.drawio with {len(cells)} cells!")

if __name__ == '__main__':
    build_clean_diagram()
