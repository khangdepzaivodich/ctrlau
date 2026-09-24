import xml.etree.ElementTree as ET

def build_paper_figure():
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
    # STYLES (Clean Academic Paper - CVPR/ICCV Aesthetic)
    # =========================================================================
    FONT = "fontFamily=Helvetica;"
    
    # Outer Containers
    BOX_CONTAINER_GRAY = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=#b0bec5;strokeWidth=1.5;arcSize=3;{FONT}"
    BOX_CONTAINER_YELLOW = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fffdf7;strokeColor=#ffe082;strokeWidth=1.5;arcSize=3;{FONT}"
    BOX_CONTAINER_GREEN = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f4fbf7;strokeColor=#a5d6a7;strokeWidth=1.5;arcSize=3;{FONT}"

    # Section Headers
    TITLE_MAIN = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontStyle=1;fontSize=14;fontColor=#263238;{FONT}"
    SUBTITLE = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontColor=#546e7a;fontSize=10;{FONT}"
    BANNER_P2 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#2e7d32;strokeColor=none;fontColor=#ffffff;align=left;spacingLeft=15;fontStyle=1;fontSize=12;{FONT}"

    # Architecture Components
    STYLE_BACKBONE = f"shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=10;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=1.5;fontColor=#0d47a1;fontSize=11;fontStyle=1;{FONT}"
    STYLE_LINEAR_BLOCK = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#9e9e9e;strokeWidth=1.2;fontColor=#212121;fontSize=10;{FONT}"
    STYLE_LINEAR = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;strokeWidth=1.2;fontColor=#212121;fontSize=10;{FONT}"
    STYLE_CLIP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;strokeWidth=1.5;fontColor=#1b5e20;fontSize=11;fontStyle=1;{FONT}"

    # 3D Tensor Cubes
    STYLE_CUBE_FEAT = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=1.2;fontColor=#1565c0;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_AU = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=1.2;fontColor=#4a148c;fontSize=9.5;fontStyle=1;{FONT}"
    STYLE_CUBE_EXP = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=1.2;fontColor=#880e4f;fontSize=9.5;fontStyle=1;{FONT}"

    # Multi-branch CNN blocks
    STYLE_CNN_AU = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=1.2;fontColor=#311b92;fontSize=10;fontStyle=1;{FONT}"
    STYLE_CNN_EXP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=1.2;fontColor=#880e4f;fontSize=10;fontStyle=1;{FONT}"

    # Prediction bars
    STYLE_BAR_AU = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ab47bc;strokeColor=#6a1b9a;strokeWidth=1;{FONT}"
    STYLE_BAR_EXP = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#e91e63;strokeColor=#ad1457;strokeWidth=1;{FONT}"

    # Scientific Loss Badges (Strictly LaTeX between backticks `...`)
    STYLE_LOSS_RED = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#ef5350;strokeWidth=1.2;fontColor=#b71c1c;fontSize=10;{FONT}"
    STYLE_LOSS_GREEN = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#66bb6a;strokeWidth=1.2;fontColor=#1b5e20;fontSize=10;{FONT}"
    STYLE_LOSS_PURPLE = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f3e5f5;strokeColor=#ab47bc;strokeWidth=1.2;fontColor=#4a148c;fontSize=10;{FONT}"
    STYLE_FORMULA_CARD = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#cfd8dc;strokeWidth=1;fontColor=#37474f;fontSize=9.5;{FONT}"

    # Graph Nodes & Panels
    STYLE_GRAPH_NODE_AU = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.5;fontColor=#4a148c;fontStyle=1;fontSize=10;{FONT}"
    STYLE_GRAPH_NODE_EXP = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#fce4ec;strokeColor=#d81b60;strokeWidth=1.5;fontColor=#880e4f;fontStyle=1;fontSize=10;{FONT}"
    STYLE_MODULE_CARD = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#cfd8dc;strokeWidth=1.2;arcSize=4;{FONT}"

    # Connectors (100% laser straight!)
    ARROW_H = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#455a64;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;{FONT}"
    ARROW_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#455a64;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_DASH_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#78909c;strokeWidth=1.2;dashed=1;dashPattern=4 3;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_GRAPH = f"edgeStyle=straightEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2e7d32;strokeWidth=1.5;{FONT}"

    # =========================================================================
    # 1. PHASE 1: CONTAINER LEFT - FEATURE EXTRACTION & CLIP ALIGNMENT
    # =========================================================================
    add_node("box_p1_left", "1", "", 40, 40, 615, 440, BOX_CONTAINER_GRAY)
    add_node("title_p1_left", "1", "<b>Feature Extraction &amp; Vision-Language Alignment</b>", 55, 48, 450, 25, TITLE_MAIN)

    # 1A. Vision-Language Alignment (CLIP) (Center Y = 105)
    node_txt = add_node("node_txt", "1", "<b>AU &amp; Expr Text</b><br><font style=\"font-size:9px;color:#555;\">FACS descriptions</font>", 65, 85, 115, 40, STYLE_LINEAR_BLOCK)
    node_clip = add_node("node_clip", "1", "<b>CLIP</b><br><font style=\"font-size:9px;\">Text Encoder (❄️)</font>", 220, 82, 95, 46, STYLE_CLIP)
    node_temb = add_node("node_temb", "1", "<b>`T_a, T_e`</b><br><font style=\"font-size:8.5px;\">Text Emb</font>", 355, 85, 70, 40, STYLE_CUBE_EXP)
    node_lcon = add_node("node_lcon", "1", "<span style=\"color:#b71c1c;\"><b>`L_{con}`</b></span><br><font style=\"font-size:8px;color:#555;\">InfoNCE</font>", 465, 82, 85, 46, STYLE_LOSS_RED)

    add_edge("e_txt_clip", "1", "node_txt", "node_clip", "", ARROW_H)
    add_edge("e_clip_temb", "1", "node_clip", "node_temb", "", ARROW_H)
    add_edge("e_temb_lcon", "1", "node_temb", "node_lcon", "", ARROW_H)

    # 1B. Visual Feature Extraction Pipeline (Center Y = 195)
    # Face -> ResNet50 -> LinearBlock -> Feature Map (49x512) -> Flatten Grid
    img_face = add_node("GjfUm5Ax_Uw3Q_51WUAJ-1", "1", "", 65, 160, 70, 70, image_style)
    node_bb = add_node("node_bb", "1", "<b>ResNet-50</b><br><font style=\"font-size:9px;\">Backbone</font>", 175, 165, 85, 60, STYLE_BACKBONE)
    node_lp = add_node("node_lp", "1", "<b>LinearBlock</b><br><font style=\"font-size:9px;\">2048 &rarr; 512</font>", 295, 170, 80, 50, STYLE_LINEAR_BLOCK)
    node_feat = add_node("node_feat", "1", "<b>Feature Map</b><br><font style=\"font-size:8.5px;\">`49 \\times 512`</font>", 405, 165, 80, 60, STYLE_CUBE_FEAT)
    node_grid = add_node("node_grid", "1", "<b>Flatten Grid</b><br><font style=\"font-size:8.5px;\">Spatial Patches</font>", 520, 165, 85, 60, STYLE_CUBE_FEAT)

    add_edge("e_f_bb", "1", "GjfUm5Ax_Uw3Q_51WUAJ-1", "node_bb", "", ARROW_H)
    add_edge("e_bb_lp", "1", "node_bb", "node_lp", "", ARROW_H)
    add_edge("e_lp_feat", "1", "node_lp", "node_feat", "", ARROW_H)
    add_edge("e_feat_grid", "1", "node_feat", "node_grid", "", ARROW_H)

    # 1C. HSIC Representation Disentanglement (Clean Academic Loss Supervision)
    add_node("lbl_hsic", "1", "<b>HSIC Representation Disentanglement</b>", 65, 275, 300, 20, TITLE_MAIN)
    add_node("sub_hsic", "1", "<font style=\"font-size:9.5px;color:#546e7a;\">Supervises AU Embeddings `V_a` via Hilbert-Schmidt Independence Criterion</font>", 65, 295, 450, 18, SUBTITLE)

    # 3 Distinct HSIC Loss Nodes (Center Y = 350)
    node_lib = add_node("node_lib", "1", 
                        "<span style=\"color:#b71c1c;\"><b>`L_{ib}`</b></span><br><font style=\"font-size:9px;color:#37474f;\">HSIC(`V_a`, `z_{img}`)</font><br><font style=\"font-size:8px;color:#78909c;\">Noise Compression</font>", 
                        65, 325, 135, 50, STYLE_LOSS_RED)
    node_lalign = add_node("node_lalign", "1", 
                           "<span style=\"color:#1b5e20;\"><b>`L_{align}`</b></span><br><font style=\"font-size:9px;color:#37474f;\">-HSIC(`V_a`, `Y_a`)</font><br><font style=\"font-size:8px;color:#78909c;\">Label Alignment</font>", 
                           225, 325, 135, 50, STYLE_LOSS_GREEN)
    node_ldecorr = add_node("node_ldecorr", "1", 
                            "<span style=\"color:#b71c1c;\"><b>`L_{decorr}`</b></span><br><font style=\"font-size:9px;color:#37474f;\">&sum; HSIC(`V_a^i`, `V_a^j`)</font><br><font style=\"font-size:8px;color:#78909c;\">AU Decorrelation</font>", 
                            385, 325, 140, 50, STYLE_LOSS_RED)

    # Phase 1 Total Loss Formula Pill
    add_node("formula_p1", "1", 
             "<b>Phase 1 Loss:</b> `L_{\\text{phase1}} = L_{wa} + \\gamma L_{we} + \\lambda_{ib} L_{ib} + \\lambda_{align} L_{align} + \\lambda_{decorr} L_{decorr} + \\lambda_{con} L_{con}`", 
             65, 405, 560, 32, STYLE_FORMULA_CARD)

    # =========================================================================
    # 2. PHASE 1: CONTAINER RIGHT - MULTI-BRANCH CLASSIFIER & FACS PRIOR
    # =========================================================================
    add_node("box_p1_right", "1", "", 680, 40, 630, 440, BOX_CONTAINER_YELLOW)
    add_node("title_p1_right", "1", "<b>Multi-Branch Classifier &amp; FACS Prior</b>", 695, 48, 450, 25, TITLE_MAIN)

    # Connection from Spatial Grid into Multi-Branch Classifier (Center Y = 195)
    # node_grid center Y is 195. box_p1_right entryY = (195 - 40) / 440 = 0.3522727
    arrow_grid_multi = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#455a64;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.3522727;{FONT}"
    add_edge("e_grid_multi", "1", "node_grid", "box_p1_right", "Spatial Patches", arrow_grid_multi)

    # --- AU Head (Rows at Center Y = 115, 158, 215) ---
    add_node("lbl_auhead", "1", "<b>AU Head (8 Branches)</b>", 695, 82, 180, 20, TITLE_MAIN)

    # Row AU1 (Center Y = 115)
    add_node("cnn_au1", "1", "1D CNN<br>(AU1)", 695, 98, 65, 34, STYLE_CNN_AU)
    add_node("va_1", "1", "<b>`V_a^{(1)}`</b>", 780, 96, 40, 38, STYLE_CUBE_AU)
    add_node("lin_au1", "1", "Linear", 845, 100, 50, 30, STYLE_LINEAR)
    add_node("pau1", "1", "", 915, 98, 10, 34, STYLE_BAR_AU)

    add_edge("e_c1_v1", "1", "cnn_au1", "va_1", "", ARROW_H)
    add_edge("e_v1_l1", "1", "va_1", "lin_au1", "", ARROW_H)
    add_edge("e_l1_p1", "1", "lin_au1", "pau1", "", ARROW_H)

    # Row AU2 (Center Y = 158)
    add_node("cnn_au2", "1", "1D CNN<br>(AU2)", 695, 141, 65, 34, STYLE_CNN_AU)
    add_node("va_2", "1", "<b>`V_a^{(2)}`</b>", 780, 139, 40, 38, STYLE_CUBE_AU)
    add_node("lin_au2", "1", "Linear", 845, 143, 50, 30, STYLE_LINEAR)
    add_node("pau2", "1", "", 915, 141, 10, 34, STYLE_BAR_AU)

    add_edge("e_c2_v2", "1", "cnn_au2", "va_2", "", ARROW_H)
    add_edge("e_v2_l2", "1", "va_2", "lin_au2", "", ARROW_H)
    add_edge("e_l2_p2", "1", "lin_au2", "pau2", "", ARROW_H)

    # Dots
    add_node("dots_au", "1", "<b>&vellip;</b>", 720, 178, 20, 20, TITLE_MAIN)

    # Row AU12 (Center Y = 215)
    add_node("cnn_au12", "1", "1D CNN<br>(AU12)", 695, 198, 65, 34, STYLE_CNN_AU)
    add_node("va_12", "1", "<b>`V_a^{(12)}`</b>", 780, 196, 40, 38, STYLE_CUBE_AU)
    add_node("lin_au12", "1", "Linear", 845, 200, 50, 30, STYLE_LINEAR)
    add_node("pau12", "1", "", 915, 198, 10, 34, STYLE_BAR_AU)

    add_edge("e_c12_v12", "1", "cnn_au12", "va_12", "", ARROW_H)
    add_edge("e_v12_l12", "1", "va_12", "lin_au12", "", ARROW_H)
    add_edge("e_l12_p12", "1", "lin_au12", "pau12", "", ARROW_H)

    # AU Loss & Prediction (Center Y = 158, Center X = 1055)
    add_node("lbl_pau", "1", "<b>`P_{AU}`</b>", 935, 148, 35, 20, TITLE_MAIN)
    node_lwa = add_node("node_lwa", "1", "<span style=\"color:#b71c1c;\"><b>`L_{wa}`</b></span><br><font style=\"font-size:8.5px;color:#555;\">Weighted Asymmetric Loss</font>", 985, 137, 140, 42, STYLE_LOSS_RED)
    add_edge("e_pau_lwa", "1", "pau2", "node_lwa", "", ARROW_H)

    # --- FACS Prior Matrix M_AE --- (Center X = 1055, Center Y = 238)
    node_mae = add_node("node_mae", "1", "<b>FACS Prior Matrix `M_{AE}`</b><br><font style=\"font-size:8.5px;color:#555;\">`Y_e = \\text{argmax}(Y_a \\cdot M_{AE})`</font>", 985, 215, 140, 46, STYLE_FORMULA_CARD)
    add_edge("e_lwa_mae", "1", "node_lwa", "node_mae", "", ARROW_DASH_V)

    # --- Emotion Head (Rows at Center Y = 295, 338, 398) ---
    add_node("lbl_exphead", "1", "<b>Expression Head (7 Branches)</b>", 695, 260, 200, 20, TITLE_MAIN)

    # Row E1 (Center Y = 295)
    add_node("cnn_e1", "1", "1D CNN<br>(E1)", 695, 278, 65, 34, STYLE_CNN_EXP)
    add_node("ve_1", "1", "<b>`V_e^{(1)}`</b>", 780, 276, 40, 38, STYLE_CUBE_EXP)
    add_node("lin_e1", "1", "Linear", 845, 280, 50, 30, STYLE_LINEAR)
    add_node("pe1", "1", "", 915, 278, 10, 34, STYLE_BAR_EXP)

    add_edge("e_ce1_ve1", "1", "cnn_e1", "ve_1", "", ARROW_H)
    add_edge("e_ve1_le1", "1", "ve_1", "lin_e1", "", ARROW_H)
    add_edge("e_le1_pe1", "1", "lin_e1", "pe1", "", ARROW_H)

    # Row E2 (Center Y = 338)
    add_node("cnn_e2", "1", "1D CNN<br>(E2)", 695, 321, 65, 34, STYLE_CNN_EXP)
    add_node("ve_2", "1", "<b>`V_e^{(2)}`</b>", 780, 319, 40, 38, STYLE_CUBE_EXP)
    add_node("lin_e2", "1", "Linear", 845, 323, 50, 30, STYLE_LINEAR)
    add_node("pe2", "1", "", 915, 321, 10, 34, STYLE_BAR_EXP)

    add_edge("e_ce2_ve2", "1", "cnn_e2", "ve_2", "", ARROW_H)
    add_edge("e_ve2_le2", "1", "ve_2", "lin_e2", "", ARROW_H)
    add_edge("e_le2_pe2", "1", "lin_e2", "pe2", "", ARROW_H)

    # Dots
    add_node("dots_exp", "1", "<b>&vellip;</b>", 720, 358, 20, 20, TITLE_MAIN)

    # Row E7 (Center Y = 398)
    add_node("cnn_e7", "1", "1D CNN<br>(E7)", 695, 381, 65, 34, STYLE_CNN_EXP)
    add_node("ve_7", "1", "<b>`V_e^{(7)}`</b>", 780, 379, 40, 38, STYLE_CUBE_EXP)
    add_node("lin_e7", "1", "Linear", 845, 383, 50, 30, STYLE_LINEAR)
    add_node("pe7", "1", "", 915, 381, 10, 34, STYLE_BAR_EXP)

    add_edge("e_ce7_ve7", "1", "cnn_e7", "ve_7", "", ARROW_H)
    add_edge("e_ve7_le7", "1", "ve_7", "lin_e7", "", ARROW_H)
    add_edge("e_le7_pe7", "1", "lin_e7", "pe7", "", ARROW_H)

    # Emotion Loss & Prediction (Center Y = 338, Center X = 1055)
    add_node("lbl_pe", "1", "<b>`P_E`</b>", 935, 328, 35, 20, TITLE_MAIN)
    node_lwe = add_node("node_lwe", "1", "<span style=\"color:#b71c1c;\"><b>`L_{we}`</b></span><br><font style=\"font-size:8.5px;color:#555;\">Expression BCE Loss</font>", 985, 317, 140, 42, STYLE_LOSS_RED)
    add_edge("e_pe_lwe", "1", "pe2", "node_lwe", "", ARROW_H)
    add_edge("e_mae_lwe", "1", "node_mae", "node_lwe", "", ARROW_DASH_V)

    # =========================================================================
    # 3. PHASE 2: CAUSAL DISCOVERY & REASONING (Bottom Container)
    # =========================================================================
    add_node("box_phase2", "1", "", 40, 510, 1270, 340, BOX_CONTAINER_GREEN)
    add_node("banner_phase2", "1", 
        "<b>PHASE 2: TWO-LEVEL CAUSAL GRAPH DISCOVERY &amp; COUNTERFACTUAL REASONING</b> &nbsp;&nbsp;&nbsp;&nbsp; "
        "<span style=\"background-color:#ffebee;color:#c62828;padding:2px 8px;border-radius:3px;font-size:10px;border:1px solid #ef9a9a;\"><b>❄️ Frozen: ResNet-50 &amp; 1D CNN Heads</b></span>",
        50, 518, 1250, 28, BANNER_P2)

    # --- Module 1: AU-AU Causal Graph (Left: X=60, Y=560, W=360, H=270) ---
    add_node("box_auau", "1", "", 60, 560, 360, 270, STYLE_MODULE_CARD)
    add_node("title_auau", "1", "<b>Level 1: AU-AU Causal Graph</b>", 70, 566, 250, 20, TITLE_MAIN)
    add_node("sub_auau", "1", "<font style=\"font-size:9px;color:#555;\">Discovers causal DAG `A_{AU-AU}` over AU representations `V_a`</font>", 70, 584, 340, 16, SUBTITLE)

    # Graph Nodes (Center Y = 615 for row 1, 665 for row 2)
    gau_1 = add_node("gau_1", "1", "AU1", 75, 601, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_2 = add_node("gau_2", "1", "AU2", 130, 601, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_4 = add_node("gau_4", "1", "AU4", 185, 601, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_6 = add_node("gau_6", "1", "AU6", 75, 651, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_12 = add_node("gau_12", "1", "AU12", 130, 651, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_25 = add_node("gau_25", "1", "AU25", 185, 651, 28, 28, STYLE_GRAPH_NODE_AU)

    add_edge("eg_1_2", "1", "gau_1", "gau_2", "", ARROW_GRAPH)
    add_edge("eg_2_4", "1", "gau_2", "gau_4", "", ARROW_GRAPH)
    add_edge("eg_6_12", "1", "gau_6", "gau_12", "", ARROW_GRAPH)
    add_edge("eg_12_25", "1", "gau_12", "gau_25", "", ARROW_GRAPH)
    add_edge("eg_2_12", "1", "gau_2", "gau_12", "", ARROW_GRAPH)

    # GAT and Losses in AU-AU (All aligned at Center X = 310)
    # gau_4 center Y = 615. node_gat1 center Y = 595 + 20 = 615.
    node_gat1 = add_node("node_gat1", "1", "<b>GAT Layer</b><br><font style=\"font-size:8.5px;\">Learns `A_{AU-AU}`</font>", 260, 595, 100, 40, STYLE_LINEAR_BLOCK)
    node_ldag = add_node("node_ldag", "1", "<span style=\"color:#b71c1c;\"><b>`L_{DAG}`</b></span><br><font style=\"font-size:8px;\">tr(e^{A&comp;A}) - d = 0</font>", 260, 647, 100, 34, STYLE_LOSS_RED)
    node_lcau1 = add_node("node_lcau1", "1", "<span style=\"color:#b71c1c;\"><b>`L_{causal}^{AU}`</b></span><br><font style=\"font-size:8px;\">Polarity &amp; Imp</font>", 260, 691, 100, 34, STYLE_LOSS_RED)
    node_pauau = add_node("node_pauau", "1", "Classifier &rarr; <b>`P_{AU-AU}`</b><br><span style=\"color:#b71c1c;\"><b>`L_{au\_au}`</b></span> (WAL)", 225, 740, 170, 42, STYLE_FORMULA_CARD)

    add_edge("eg_4_gat1", "1", "gau_4", "node_gat1", "", ARROW_H)
    add_edge("e_gat1_dag", "1", "node_gat1", "node_ldag", "", ARROW_V)
    add_edge("e_ldag_lcau", "1", "node_ldag", "node_lcau1", "", ARROW_V)
    add_edge("e_lcau_pauau", "1", "node_lcau1", "node_pauau", "", ARROW_V)

    # --- Module 2: AU-Expression Bipartite Graph (Middle: X=480, Y=560, W=380, H=270) ---
    add_node("box_auexp", "1", "", 480, 560, 380, 270, STYLE_MODULE_CARD)
    add_node("title_auexp", "1", "<b>Level 2: AU &rarr; Expression Graph</b>", 490, 566, 250, 20, TITLE_MAIN)
    add_node("sub_auexp", "1", "<font style=\"font-size:9px;color:#555;\">Bipartite GAT `A_{AU-Exp}` guided by FACS prior</font>", 490, 584, 350, 16, SUBTITLE)

    # Clean Cascade arrow from AU-AU to AU-EXP: 100% straight horizontal line at Y = 695.0
    add_edge("e_cascade_1_2", "1", "box_auau", "box_auexp", "`updated_au` (`V_a^*`)", ARROW_H)

    # Bipartite Graph Nodes (Center Y = 615 for Happy, 655 for Sad, 695 for Fear)
    bau_6 = add_node("bau_6", "1", "AU6", 495, 602, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_12 = add_node("bau_12", "1", "AU12", 495, 637, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_4 = add_node("bau_4", "1", "AU4", 495, 672, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_1 = add_node("bau_1", "1", "AU1", 495, 707, 26, 26, STYLE_GRAPH_NODE_AU)

    bemo_hap = add_node("bemo_hap", "1", "Happy", 575, 601, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_sad = add_node("bemo_sad", "1", "Sad", 575, 641, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_fea = add_node("bemo_fea", "1", "Fear", 575, 681, 28, 28, STYLE_GRAPH_NODE_EXP)

    add_edge("eb_6_h", "1", "bau_6", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_12_h", "1", "bau_12", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_4_s", "1", "bau_4", "bemo_sad", "", ARROW_GRAPH)
    add_edge("eb_1_f", "1", "bau_1", "bemo_fea", "", ARROW_GRAPH)

    # Bipartite GAT and Losses (All aligned at Center X = 740)
    # bemo_hap center Y = 615. node_gat2 center Y = 595 + 20 = 615.
    node_gat2 = add_node("node_gat2", "1", "<b>Bipartite GAT</b><br><font style=\"font-size:8.5px;\">Learns `A_{AU-Exp}`</font>", 685, 595, 110, 40, STYLE_LINEAR_BLOCK)
    node_lcau2 = add_node("node_lcau2", "1", "<span style=\"color:#b71c1c;\"><b>`L_{causal}^{Exp}`</b></span><br><font style=\"font-size:8px;\">Causal Activation</font>", 685, 647, 110, 34, STYLE_LOSS_RED)
    node_lfacs = add_node("node_lfacs", "1", "<span style=\"color:#b71c1c;\"><b>`L_{facs}^{Exp}`</b></span><br><font style=\"font-size:8px;\">FACS Consistency</font>", 685, 691, 110, 34, STYLE_LOSS_RED)
    node_pfinal = add_node("node_pfinal", "1", "Final: <b>`P_G^{AU}, P_G^{Exp}`</b><br><span style=\"color:#b71c1c;\"><b>`L_{wa}^{AU}, L_{we}^{Exp}`</b></span>", 640, 740, 200, 42, STYLE_FORMULA_CARD)

    add_edge("eb_hap_gat2", "1", "bemo_hap", "node_gat2", "", ARROW_H)
    add_edge("e_gat2_cau", "1", "node_gat2", "node_lcau2", "", ARROW_V)
    add_edge("e_lcau_facs", "1", "node_lcau2", "node_lfacs", "", ARROW_V)
    add_edge("e_lfacs_pfinal", "1", "node_lfacs", "node_pfinal", "", ARROW_V)

    # --- Module 3: HiMod Counterfactual Intervention (Right: X=910, Y=560, W=380, H=270) ---
    add_node("box_himod", "1", "", 910, 560, 380, 270, STYLE_MODULE_CARD)
    add_node("title_himod", "1", "<b>HiMod Counterfactual Reasoning</b>", 920, 566, 260, 20, TITLE_MAIN)
    add_node("sub_himod", "1", "<font style=\"font-size:9px;color:#555;\">Intervention via importance mask `M_{imp}` partition</font>", 920, 584, 350, 16, SUBTITLE)

    # Cascade Arrow from Level 2 to HiMod at Y = 695.0
    add_edge("e_cascade_2_3", "1", "box_auexp", "box_himod", "`M_{imp}`", ARROW_H)

    # Partition Header Card
    add_node("node_cf_part", "1", "<b>Causal Importance Partition:</b> `\\text{diag}(M_{imp}) > \\tau`", 925, 603, 350, 28, STYLE_FORMULA_CARD)
    
    # Important Branch (Center Y = 657)
    add_node("node_cf_imp", "1", "<b>Perturb Important:</b><br>`V_a^* + \\epsilon \\cdot M_{imp}`", 925, 640, 205, 34, STYLE_CUBE_AU)
    add_node("node_lcf_imp", "1", "<span style=\"color:#b71c1c;\"><b>`L_{cf}^{imp}`</b></span><br><font style=\"font-size:8px;\">Invariance</font>", 1145, 640, 130, 34, STYLE_LOSS_RED)
    add_edge("e_imp_loss", "1", "node_cf_imp", "node_lcf_imp", "", ARROW_H)

    # Unimportant Branch (Center Y = 705)
    add_node("node_cf_unimp", "1", "<b>Perturb Unimportant:</b><br>`V_a^* + \\epsilon \\cdot (1 - M_{imp})`", 925, 688, 205, 34, STYLE_CUBE_AU)
    add_node("node_lcf_unimp", "1", "<span style=\"color:#1b5e20;\"><b>`L_{cf}^{unimp}`</b></span><br><font style=\"font-size:8px;\">Consistency</font>", 1145, 688, 130, 34, STYLE_LOSS_GREEN)
    add_edge("e_unimp_loss", "1", "node_cf_unimp", "node_lcf_unimp", "", ARROW_H)

    # Phase 2 Total loss formula
    add_node("node_p2_loss", "1", 
        "<b>Phase 2 Loss:</b> `L_{\\text{phase2}} = L_{wa} + \\gamma L_{we} + \\lambda_{dag} L_{DAG} + \\lambda_{cau} L_{causal} + \\lambda_{facs} L_{facs}^{Exp} + \\lambda_{cf} L_{cf}`", 
        925, 742, 350, 38, STYLE_FORMULA_CARD)

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
        
    print(f"Successfully generated paper figure ctrlau.drawio with {len(cells)} cells!")

if __name__ == '__main__':
    build_paper_figure()
