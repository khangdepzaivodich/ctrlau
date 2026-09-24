import xml.etree.ElementTree as ET

def build_creative_figure():
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
    # STYLES (Clean Academic Paper - CVPR/ICCV Aesthetic)
    # =========================================================================
    FONT = "fontFamily=Helvetica;"
    
    # Outer Sections (Subtle tint, clean borders)
    PANEL_P1 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fafbfc;strokeColor=#cfd8dc;strokeWidth=1.5;arcSize=2;{FONT}"
    PANEL_P2 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f6fbf8;strokeColor=#a5d6a7;strokeWidth=1.5;arcSize=2;{FONT}"
    PANEL_YELLOW = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fffdf7;strokeColor=#ffe082;strokeWidth=1.5;arcSize=3;{FONT}"
    PANEL_WHITE = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#cfd8dc;strokeWidth=1.2;arcSize=3;{FONT}"

    # Typography
    TITLE_MAIN = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontStyle=1;fontSize=13;fontColor=#263238;{FONT}"
    SUBTITLE = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontColor=#546e7a;fontSize=9.5;{FONT}"
    BANNER_P2 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#2e7d32;strokeColor=none;fontColor=#ffffff;align=left;spacingLeft=15;fontStyle=1;fontSize=12;{FONT}"

    # Architecture Components
    STYLE_BACKBONE = f"shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=10;fillColor=#e3f2fd;strokeColor=#1976d2;strokeWidth=1.5;fontColor=#0d47a1;fontSize=11;fontStyle=1;{FONT}"
    STYLE_LINEAR_BLOCK = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#9e9e9e;strokeWidth=1.2;fontColor=#212121;fontSize=10;{FONT}"
    STYLE_LINEAR = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;strokeWidth=1.2;fontColor=#212121;fontSize=10;{FONT}"
    STYLE_CLIP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#43a047;strokeWidth=1.5;fontColor=#1b5e20;fontSize=11;fontStyle=1;{FONT}"

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

    # Scientific Loss Badges (LaTeX between backticks)
    STYLE_LOSS_RED = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#ef5350;strokeWidth=1.2;fontColor=#b71c1c;fontSize=9.5;{FONT}"
    STYLE_LOSS_GREEN = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#66bb6a;strokeWidth=1.2;fontColor=#1b5e20;fontSize=9.5;{FONT}"
    STYLE_FORMULA = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#b0bec5;strokeWidth=1;fontColor=#37474f;fontSize=9.5;{FONT}"

    # Graph Nodes & Panels
    STYLE_GRAPH_NODE_AU = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.5;fontColor=#4a148c;fontStyle=1;fontSize=10;{FONT}"
    STYLE_GRAPH_NODE_EXP = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#fce4ec;strokeColor=#d81b60;strokeWidth=1.5;fontColor=#880e4f;fontStyle=1;fontSize=10;{FONT}"

    # Laser-straight connectives
    ARROW_H = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#455a64;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;{FONT}"
    ARROW_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#455a64;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_DASH_V = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#78909c;strokeWidth=1.2;dashed=1;dashPattern=4 3;exitX=0.5;exitY=1;entryX=0.5;entryY=0;{FONT}"
    ARROW_GRAPH = f"edgeStyle=straightEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2e7d32;strokeWidth=1.5;{FONT}"

    # =========================================================================
    # 1. TOP REGION: PHASE 1 (Feature Extraction & Multi-Branch Heads)
    # =========================================================================
    add_node("panel_phase1", "1", "", 40, 30, 1300, 430, PANEL_P1)
    add_node("title_phase1", "1", "<b>PHASE 1: FEATURE EXTRACTION &amp; MULTI-BRANCH REPRESENTATION LEARNING</b>", 55, 38, 650, 25, TITLE_MAIN)

    # 1A. Vision-Language Alignment (CLIP Text Stream) (Center Y = 95)
    node_txt = add_node("node_txt", "1", "<b>FACS Descriptions</b><br><font style=\"font-size:9px;color:#666;\">AU &amp; Emotion text</font>", 60, 75, 130, 40, STYLE_LINEAR_BLOCK)
    node_clip = add_node("node_clip", "1", "<b>CLIP Text Encoder</b><br><font style=\"font-size:9px;color:#2e7d32;\">Frozen (❄️)</font>", 230, 72, 115, 46, STYLE_CLIP)
    node_temb = add_node("node_temb", "1", "<b>`\\mathbf{T}_a, \\mathbf{T}_e`</b><br><font style=\"font-size:8.5px;\">Text Emb</font>", 385, 75, 75, 40, STYLE_CUBE_EXP)
    node_lcon = add_node("node_lcon", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{con}}`</b></span><br><font style=\"font-size:8px;color:#555;\">InfoNCE</font>", 500, 73, 90, 44, STYLE_LOSS_RED)

    add_edge("e_txt_clip", "1", "node_txt", "node_clip", "", ARROW_H)
    add_edge("e_clip_temb", "1", "node_clip", "node_temb", "", ARROW_H)
    add_edge("e_temb_lcon", "1", "node_temb", "node_lcon", "", ARROW_H)

    # 1B. Visual Feature Extraction Pipeline (Center Y = 195)
    img_face = add_node("GjfUm5Ax_Uw3Q_51WUAJ-1", "1", "", 60, 160, 70, 70, image_style)
    node_bb = add_node("node_bb", "1", "<b>ResNet-50</b><br><font style=\"font-size:9px;\">Backbone</font>", 170, 165, 90, 60, STYLE_BACKBONE)
    node_lp = add_node("node_lp", "1", "<b>LinearBlock</b><br><font style=\"font-size:9px;\">`2048 \\to 512`</font>", 300, 170, 85, 50, STYLE_LINEAR_BLOCK)
    node_feat = add_node("node_feat", "1", "<b>Feature Map</b><br><font style=\"font-size:8.5px;\">`49 \\times 512`</font>", 425, 165, 80, 60, STYLE_CUBE_FEAT)
    node_grid = add_node("node_grid", "1", "<b>Spatial Patches</b><br><font style=\"font-size:8.5px;\">`49 \\times 512`</font>", 545, 165, 85, 60, STYLE_CUBE_FEAT)

    add_edge("e_f_bb", "1", "GjfUm5Ax_Uw3Q_51WUAJ-1", "node_bb", "", ARROW_H)
    add_edge("e_bb_lp", "1", "node_bb", "node_lp", "", ARROW_H)
    add_edge("e_lp_feat", "1", "node_lp", "node_feat", "", ARROW_H)
    add_edge("e_feat_grid", "1", "node_feat", "node_grid", "", ARROW_H)

    # 1C. HSIC Representation Disentanglement (Clean Supervision Tags)
    add_node("lbl_hsic", "1", "<b>HSIC Representation Disentanglement:</b>", 60, 275, 300, 20, TITLE_MAIN)
    
    # 3 Distinct HSIC Loss Badges (Center Y = 330)
    node_lib = add_node("node_lib", "1", 
                        "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{ib}}`</b></span><br><font style=\"font-size:9px;\">`\\text{HSIC}(\\mathbf{V}_a, \\mathbf{z}_{\\text{img}})`</font><br><font style=\"font-size:8px;color:#78909c;\">Noise Compression</font>", 
                        60, 308, 165, 44, STYLE_LOSS_RED)
    node_lalign = add_node("node_lalign", "1", 
                           "<span style=\"color:#1b5e20;\"><b>`\\mathcal{L}_{\\text{align}}`</b></span><br><font style=\"font-size:9px;\">`-\\text{HSIC}(\\mathbf{V}_a, \\mathbf{Y}_a)`</font><br><font style=\"font-size:8px;color:#78909c;\">Label Alignment</font>", 
                           240, 308, 165, 44, STYLE_LOSS_GREEN)
    node_ldecorr = add_node("node_ldecorr", "1", 
                            "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{decorr}}`</b></span><br><font style=\"font-size:9px;\">`\\sum_{i \\ne j} \\text{HSIC}(\\mathbf{V}_a^i, \\mathbf{V}_a^j)`</font><br><font style=\"font-size:8px;color:#78909c;\">Cross-AU Decorrelation</font>", 
                            420, 308, 180, 44, STYLE_LOSS_RED)

    # Phase 1 Total Loss Formula Tag
    add_node("formula_p1", "1", 
             "<b>Phase 1 Loss:</b> `\\mathcal{L}_{\\text{phase1}} = \\mathcal{L}_{\\text{wa}} + \\gamma \\mathcal{L}_{\\text{we}} + \\lambda_{\\text{ib}} \\mathcal{L}_{\\text{ib}} + \\lambda_{\\text{align}} \\mathcal{L}_{\\text{align}} + \\lambda_{\\text{decorr}} \\mathcal{L}_{\\text{decorr}} + \\lambda_{\\text{con}} \\mathcal{L}_{\\text{con}}`", 
             60, 385, 540, 34, STYLE_FORMULA)

    # -------------------------------------------------------------------------
    # 1D. Multi-Branch Spatial Classifier (Right Side of Phase 1)
    # -------------------------------------------------------------------------
    add_node("box_multibranch", "1", "", 670, 55, 650, 385, PANEL_YELLOW)
    add_node("title_multi", "1", "<b>Multi-Branch Spatial Classifier &amp; FACS Prior</b>", 685, 63, 350, 20, TITLE_MAIN)

    # Spatial Patches -> Multi-Branch input: 100% straight horizontal at Center Y = 195
    # box_multibranch Y=55, H=385. entryY = (195 - 55) / 385 = 0.3636364
    arrow_grid_multi = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#455a64;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.3636364;{FONT}"
    add_edge("e_grid_multi", "1", "node_grid", "box_multibranch", "Spatial Patches", arrow_grid_multi)

    # --- AU Head (Rows at Center Y = 120, 162, 218) ---
    add_node("lbl_auhead", "1", "<b>AU Head (8 Branches)</b>", 685, 88, 180, 18, TITLE_MAIN)

    # Row AU1 (Center Y = 120)
    add_node("cnn_au1", "1", "1D CNN<br>(AU1)", 685, 103, 65, 34, STYLE_CNN_AU)
    add_node("va_1", "1", "<b>`\\mathbf{V}_a^{(1)}`</b>", 770, 101, 42, 38, STYLE_CUBE_AU)
    add_node("lin_au1", "1", "Linear", 835, 105, 50, 30, STYLE_LINEAR)
    add_node("pau1", "1", "", 905, 103, 10, 34, STYLE_BAR_AU)

    add_edge("e_c1_v1", "1", "cnn_au1", "va_1", "", ARROW_H)
    add_edge("e_v1_l1", "1", "va_1", "lin_au1", "", ARROW_H)
    add_edge("e_l1_p1", "1", "lin_au1", "pau1", "", ARROW_H)

    # Row AU2 (Center Y = 162)
    add_node("cnn_au2", "1", "1D CNN<br>(AU2)", 685, 145, 65, 34, STYLE_CNN_AU)
    add_node("va_2", "1", "<b>`\\mathbf{V}_a^{(2)}`</b>", 770, 143, 42, 38, STYLE_CUBE_AU)
    add_node("lin_au2", "1", "Linear", 835, 147, 50, 30, STYLE_LINEAR)
    add_node("pau2", "1", "", 905, 145, 10, 34, STYLE_BAR_AU)

    add_edge("e_c2_v2", "1", "cnn_au2", "va_2", "", ARROW_H)
    add_edge("e_v2_l2", "1", "va_2", "lin_au2", "", ARROW_H)
    add_edge("e_l2_p2", "1", "lin_au2", "pau2", "", ARROW_H)

    # Dots
    add_node("dots_au", "1", "<b>&vellip;</b>", 710, 182, 20, 20, TITLE_MAIN)

    # Row AU12 (Center Y = 218)
    add_node("cnn_au12", "1", "1D CNN<br>(AU12)", 685, 201, 65, 34, STYLE_CNN_AU)
    add_node("va_12", "1", "<b>`\\mathbf{V}_a^{(12)}`</b>", 770, 199, 42, 38, STYLE_CUBE_AU)
    add_node("lin_au12", "1", "Linear", 835, 203, 50, 30, STYLE_LINEAR)
    add_node("pau12", "1", "", 905, 201, 10, 34, STYLE_BAR_AU)

    add_edge("e_c12_v12", "1", "cnn_au12", "va_12", "", ARROW_H)
    add_edge("e_v12_l12", "1", "va_12", "lin_au12", "", ARROW_H)
    add_edge("e_l12_p12", "1", "lin_au12", "pau12", "", ARROW_H)

    # AU Loss & Prediction (Center Y = 162, Center X = 1045)
    add_node("lbl_pau", "1", "<b>`\\mathbf{P}_{\\text{AU}}`</b>", 925, 152, 40, 20, TITLE_MAIN)
    node_lwa = add_node("node_lwa", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{wa}}`</b></span><br><font style=\"font-size:8px;color:#78909c;\">Weighted Asymmetric</font>", 980, 141, 130, 42, STYLE_LOSS_RED)
    add_edge("e_pau_lwa", "1", "pau2", "node_lwa", "", ARROW_H)

    # --- FACS Prior Matrix M_AE --- (Center X = 1045, Center Y = 248)
    node_mae = add_node("node_mae", "1", "<b>FACS Prior Matrix `\\mathbf{M}_{\\text{AE}}`</b><br><font style=\"font-size:8px;color:#555;\">`\\mathbf{Y}_e = \\operatorname{argmax}(\\mathbf{Y}_a \\mathbf{M}_{\\text{AE}})`</font>", 980, 224, 130, 48, STYLE_FORMULA)
    add_edge("e_lwa_mae", "1", "node_lwa", "node_mae", "", ARROW_DASH_V)

    # --- Emotion Head (Rows at Center Y = 292, 334, 390) ---
    add_node("lbl_exphead", "1", "<b>Expression Head (7 Branches)</b>", 685, 258, 200, 18, TITLE_MAIN)

    # Row E1 (Center Y = 292)
    add_node("cnn_e1", "1", "1D CNN<br>(E1)", 685, 275, 65, 34, STYLE_CNN_EXP)
    add_node("ve_1", "1", "<b>`\\mathbf{V}_e^{(1)}`</b>", 770, 273, 42, 38, STYLE_CUBE_EXP)
    add_node("lin_e1", "1", "Linear", 835, 277, 50, 30, STYLE_LINEAR)
    add_node("pe1", "1", "", 905, 275, 10, 34, STYLE_BAR_EXP)

    add_edge("e_ce1_ve1", "1", "cnn_e1", "ve_1", "", ARROW_H)
    add_edge("e_ve1_le1", "1", "ve_1", "lin_e1", "", ARROW_H)
    add_edge("e_le1_pe1", "1", "lin_e1", "pe1", "", ARROW_H)

    # Row E2 (Center Y = 334)
    add_node("cnn_e2", "1", "1D CNN<br>(E2)", 685, 317, 65, 34, STYLE_CNN_EXP)
    add_node("ve_2", "1", "<b>`\\mathbf{V}_e^{(2)}`</b>", 770, 315, 42, 38, STYLE_CUBE_EXP)
    add_node("lin_e2", "1", "Linear", 835, 319, 50, 30, STYLE_LINEAR)
    add_node("pe2", "1", "", 905, 317, 10, 34, STYLE_BAR_EXP)

    add_edge("e_ce2_ve2", "1", "cnn_e2", "ve_2", "", ARROW_H)
    add_edge("e_ve2_le2", "1", "ve_2", "lin_e2", "", ARROW_H)
    add_edge("e_le2_pe2", "1", "lin_e2", "pe2", "", ARROW_H)

    # Dots
    add_node("dots_exp", "1", "<b>&vellip;</b>", 710, 354, 20, 20, TITLE_MAIN)

    # Row E7 (Center Y = 390)
    add_node("cnn_e7", "1", "1D CNN<br>(E7)", 685, 373, 65, 34, STYLE_CNN_EXP)
    add_node("ve_7", "1", "<b>`\\mathbf{V}_e^{(7)}`</b>", 770, 371, 42, 38, STYLE_CUBE_EXP)
    add_node("lin_e7", "1", "Linear", 835, 375, 50, 30, STYLE_LINEAR)
    add_node("pe7", "1", "", 905, 373, 10, 34, STYLE_BAR_EXP)

    add_edge("e_ce7_ve7", "1", "cnn_e7", "ve_7", "", ARROW_H)
    add_edge("e_ve7_le7", "1", "ve_7", "lin_e7", "", ARROW_H)
    add_edge("e_le7_pe7", "1", "lin_e7", "pe7", "", ARROW_H)

    # Emotion Loss & Prediction (Center Y = 334, Center X = 1045)
    add_node("lbl_pe", "1", "<b>`\\mathbf{P}_{\\text{E}}`</b>", 925, 324, 40, 20, TITLE_MAIN)
    node_lwe = add_node("node_lwe", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{we}}`</b></span><br><font style=\"font-size:8px;color:#78909c;\">Expression BCE</font>", 980, 313, 130, 42, STYLE_LOSS_RED)
    add_edge("e_pe_lwe", "1", "pe2", "node_lwe", "", ARROW_H)
    add_edge("e_mae_lwe", "1", "node_mae", "node_lwe", "", ARROW_DASH_V)

    # Output Feature Indicators (Right of Multi-branch)
    add_node("tag_va_out", "1", "<b>Transferred Embeddings:</b><br>`\\mathbf{V}_a \\in \\mathbb{R}^{8 \\times 256}`<br><font style=\"font-size:8px;color:#78909c;\">To Level 1 &amp; 2 (❄️)</font>", 1135, 141, 165, 42, STYLE_FORMULA)
    add_node("tag_ve_out", "1", "<b>Transferred Embeddings:</b><br>`\\mathbf{V}_e \\in \\mathbb{R}^{7 \\times 256}`<br><font style=\"font-size:8px;color:#78909c;\">To Level 2 (❄️)</font>", 1135, 313, 165, 42, STYLE_FORMULA)

    # =========================================================================
    # 2. BOTTOM REGION: PHASE 2 (Two-Level Causal Graph & Counterfactual)
    # =========================================================================
    add_node("panel_phase2", "1", "", 40, 480, 1300, 395, PANEL_P2)
    add_node("banner_phase2", "1", 
        "<b>PHASE 2: TWO-LEVEL CAUSAL GRAPH DISCOVERY &amp; COUNTERFACTUAL REASONING</b> &nbsp;&nbsp;&nbsp;&nbsp; "
        "<span style=\"background-color:#ffebee;color:#c62828;padding:2px 8px;border-radius:3px;font-size:10px;border:1px solid #ef9a9a;\"><b>❄️ Frozen: ResNet-50 Backbone &amp; 1D CNN Heads</b></span>",
        50, 488, 1280, 28, BANNER_P2)

    # --- Module 1: AU-AU Causal Graph (Left: X=60, Y=530, W=360, H=330) ---
    add_node("box_auau", "1", "", 60, 530, 360, 330, PANEL_WHITE)
    add_node("title_auau", "1", "<b>Level 1: AU-AU Causal Graph</b>", 75, 538, 250, 20, TITLE_MAIN)
    add_node("sub_auau", "1", "<font style=\"font-size:9px;color:#555;\">Discovers causal DAG `\\mathbf{A}_{\\text{AU-AU}}` over `\\mathbf{V}_a`</font>", 75, 556, 330, 16, SUBTITLE)

    # Graph Nodes (Center Y = 604 for row 1, 659 for row 2)
    gau_1 = add_node("gau_1", "1", "AU1", 75, 590, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_2 = add_node("gau_2", "1", "AU2", 130, 590, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_4 = add_node("gau_4", "1", "AU4", 185, 590, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_6 = add_node("gau_6", "1", "AU6", 75, 645, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_12 = add_node("gau_12", "1", "AU12", 130, 645, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_25 = add_node("gau_25", "1", "AU25", 185, 645, 28, 28, STYLE_GRAPH_NODE_AU)

    add_edge("eg_1_2", "1", "gau_1", "gau_2", "", ARROW_GRAPH)
    add_edge("eg_2_4", "1", "gau_2", "gau_4", "", ARROW_GRAPH)
    add_edge("eg_6_12", "1", "gau_6", "gau_12", "", ARROW_GRAPH)
    add_edge("eg_12_25", "1", "gau_12", "gau_25", "", ARROW_GRAPH)
    add_edge("eg_2_12", "1", "gau_2", "gau_12", "", ARROW_GRAPH)

    # GAT and Losses in AU-AU (All aligned at Center X = 310)
    # gau_4 center Y = 604. node_gat1 center Y = 584 + 20 = 604.
    node_gat1 = add_node("node_gat1", "1", "<b>GAT Layer</b><br><font style=\"font-size:8.5px;\">Learns `\\mathbf{A}_{\\text{AU-AU}}`</font>", 255, 584, 110, 40, STYLE_LINEAR_BLOCK)
    node_ldag = add_node("node_ldag", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{DAG}}`</b></span><br><font style=\"font-size:7.5px;color:#555;\">`\\text{tr}(e^{\\mathbf{A} \\circ \\mathbf{A}}) - d = 0`</font>", 255, 642, 110, 34, STYLE_LOSS_RED)
    node_lcau1 = add_node("node_lcau1", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{causal}}^{\\text{AU}}`</b></span><br><font style=\"font-size:7.5px;color:#555;\">Polarity &amp; Importance</font>", 255, 694, 110, 34, STYLE_LOSS_RED)
    node_pauau = add_node("node_pauau", "1", "Classifier &rarr; <b>`\\mathbf{P}_{\\text{AU-AU}}`</b><br><span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{au\\_au}}`</b></span> (WAL)", 225, 746, 170, 44, STYLE_FORMULA)

    add_edge("eg_4_gat1", "1", "gau_4", "node_gat1", "", ARROW_H)
    add_edge("e_gat1_dag", "1", "node_gat1", "node_ldag", "", ARROW_V)
    add_edge("e_ldag_lcau", "1", "node_ldag", "node_lcau1", "", ARROW_V)
    add_edge("e_lcau_pauau", "1", "node_lcau1", "node_pauau", "", ARROW_V)

    # Output badge at bottom
    add_node("tag_auau_out", "1", "<b>Output:</b> Updated AU Embeddings `\\mathbf{V}_a^* \\in \\mathbb{R}^{8 \\times 256}`", 75, 810, 330, 34, STYLE_FORMULA)

    # --- Module 2: AU-Expression Bipartite Graph (Middle: X=475, Y=530, W=385, H=330) ---
    add_node("box_auexp", "1", "", 475, 530, 385, 330, PANEL_WHITE)
    add_node("title_auexp", "1", "<b>Level 2: AU &rarr; Expression Graph</b>", 490, 538, 270, 20, TITLE_MAIN)
    add_node("sub_auexp", "1", "<font style=\"font-size:9px;color:#555;\">Bipartite GAT `\\mathbf{A}_{\\text{AU-Exp}}` guided by FACS prior</font>", 490, 556, 350, 16, SUBTITLE)

    # Cascade Arrow from AU-AU to AU-EXP: 100% straight horizontal line at Y = 695.0
    add_edge("e_cascade_1_2", "1", "box_auau", "box_auexp", "`\\mathbf{V}_a^*`", ARROW_H)

    # Bipartite Graph Nodes (Center Y = 604 for Happy, 650 for Sad, 696 for Fear)
    bau_6 = add_node("bau_6", "1", "AU6", 490, 591, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_12 = add_node("bau_12", "1", "AU12", 490, 626, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_4 = add_node("bau_4", "1", "AU4", 490, 661, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_1 = add_node("bau_1", "1", "AU1", 490, 696, 26, 26, STYLE_GRAPH_NODE_AU)

    bemo_hap = add_node("bemo_hap", "1", "Happy", 575, 590, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_sad = add_node("bemo_sad", "1", "Sad", 575, 636, 28, 28, STYLE_GRAPH_NODE_EXP)
    bemo_fea = add_node("bemo_fea", "1", "Fear", 575, 682, 28, 28, STYLE_GRAPH_NODE_EXP)

    add_edge("eb_6_h", "1", "bau_6", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_12_h", "1", "bau_12", "bemo_hap", "", ARROW_GRAPH)
    add_edge("eb_4_s", "1", "bau_4", "bemo_sad", "", ARROW_GRAPH)
    add_edge("eb_1_f", "1", "bau_1", "bemo_fea", "", ARROW_GRAPH)

    # Bipartite GAT and Losses (All aligned at Center X = 745)
    # bemo_hap center Y = 604. node_gat2 center Y = 584 + 20 = 604.
    node_gat2 = add_node("node_gat2", "1", "<b>Bipartite GAT</b><br><font style=\"font-size:8.5px;\">Learns `\\mathbf{A}_{\\text{AU-Exp}}`</font>", 685, 584, 120, 40, STYLE_LINEAR_BLOCK)
    node_lcau2 = add_node("node_lcau2", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{causal}}^{\\text{Exp}}`</b></span><br><font style=\"font-size:7.5px;color:#555;\">Causal Activation</font>", 685, 642, 120, 34, STYLE_LOSS_RED)
    node_lfacs = add_node("node_lfacs", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{facs}}^{\\text{Exp}}`</b></span><br><font style=\"font-size:7.5px;color:#555;\">FACS Consistency</font>", 685, 694, 120, 34, STYLE_LOSS_RED)
    node_pfinal = add_node("node_pfinal", "1", "Final: <b>`\\mathbf{P}_G^{\\text{AU}}, \\mathbf{P}_G^{\\text{Exp}}`</b><br><span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{wa}}^{\\text{AU}}, \\mathcal{L}_{\\text{we}}^{\\text{Exp}}`</b></span>", 650, 746, 190, 44, STYLE_FORMULA)

    add_edge("eb_hap_gat2", "1", "bemo_hap", "node_gat2", "", ARROW_H)
    add_edge("e_gat2_cau", "1", "node_gat2", "node_lcau2", "", ARROW_V)
    add_edge("e_lcau_facs", "1", "node_lcau2", "node_lfacs", "", ARROW_V)
    add_edge("e_lfacs_pfinal", "1", "node_lfacs", "node_pfinal", "", ARROW_V)

    # Output badge at bottom
    add_node("tag_auexp_out", "1", "<b>Output:</b> Causal Importance Mask `\\mathbf{M}_{\\text{imp}} \\in \\mathbb{R}^{8 \\times 8}`", 490, 810, 355, 34, STYLE_FORMULA)

    # --- Module 3: HiMod Counterfactual Intervention (Right: X=915, Y=530, W=405, H=330) ---
    add_node("box_himod", "1", "", 915, 530, 405, 330, PANEL_WHITE)
    add_node("title_himod", "1", "<b>HiMod Counterfactual Reasoning</b>", 930, 538, 280, 20, TITLE_MAIN)
    add_node("sub_himod", "1", "<font style=\"font-size:9px;color:#555;\">Intervention via importance mask `\\mathbf{M}_{\\text{imp}}` partition</font>", 930, 556, 370, 16, SUBTITLE)

    # Cascade Arrow from Level 2 to HiMod at Y = 695.0
    add_edge("e_cascade_2_3", "1", "box_auexp", "box_himod", "`\\mathbf{M}_{\\text{imp}}`", ARROW_H)

    # Partition Bar
    add_node("node_cf_part", "1", "<b>Importance Partition:</b> `\\operatorname{diag}(\\mathbf{M}_{\\text{imp}}) > \\tau`", 930, 584, 375, 30, STYLE_FORMULA)
    
    # Important Branch (Center Y = 647)
    add_node("node_cf_imp", "1", "<b>Perturb Important AU:</b><br>`\\mathbf{V}_a^* + \\epsilon \\cdot \\mathbf{M}_{\\text{imp}}`", 930, 628, 215, 38, STYLE_CUBE_AU)
    add_node("node_lcf_imp", "1", "<span style=\"color:#b71c1c;\"><b>`\\mathcal{L}_{\\text{cf}}^{\\text{imp}}`</b></span><br><font style=\"font-size:8px;color:#555;\">(Invariance Loss)</font>", 1170, 628, 135, 38, STYLE_LOSS_RED)
    add_edge("e_imp_loss", "1", "node_cf_imp", "node_lcf_imp", "", ARROW_H)

    # Unimportant Branch (Center Y = 705)
    add_node("node_cf_unimp", "1", "<b>Perturb Unimportant AU:</b><br>`\\mathbf{V}_a^* + \\epsilon \\cdot (1 - \\mathbf{M}_{\\text{imp}})`", 930, 686, 215, 38, STYLE_CUBE_AU)
    add_node("node_lcf_unimp", "1", "<span style=\"color:#1b5e20;\"><b>`\\mathcal{L}_{\\text{cf}}^{\\text{unimp}}`</b></span><br><font style=\"font-size:8px;color:#555;\">(Consistency Loss)</font>", 1170, 686, 135, 38, STYLE_LOSS_GREEN)
    add_edge("e_unimp_loss", "1", "node_cf_unimp", "node_lcf_unimp", "", ARROW_H)

    # Phase 2 Total loss formula
    add_node("node_p2_loss", "1", 
        "<b>Phase 2 Loss:</b> `\\mathcal{L}_{\\text{phase2}} = \\mathcal{L}_{\\text{wa}} + \\gamma \\mathcal{L}_{\\text{we}} + \\lambda_{\\text{dag}} \\mathcal{L}_{\\text{DAG}} + \\lambda_{\\text{cau}} \\mathcal{L}_{\\text{causal}} + \\lambda_{\\text{facs}} \\mathcal{L}_{\\text{facs}}^{\\text{Exp}} + \\lambda_{\\text{cf}} \\mathcal{L}_{\\text{cf}}`", 
        930, 750, 375, 38, STYLE_FORMULA)

    # =========================================================================
    # BUILD XML
    # =========================================================================
    mxfile = ET.Element('mxfile', host='app.diagrams.net')
    diagram = ET.SubElement(mxfile, 'diagram', name='CtrlAU Architecture', id='cQhNqvIMsQzl_Nc4XbVt')
    graph_model = ET.SubElement(diagram, 'mxGraphModel', 
                                grid='0', page='1', gridSize='10', 
                                guides='1', tooltips='1', connect='1', arrows='1', fold='1', pageScale='1', 
                                pageWidth='1400', pageHeight='920', math='1', shadow='0')
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
        
    print(f"Successfully generated creative paper figure ctrlau.drawio with {len(cells)} cells!")

if __name__ == '__main__':
    build_creative_figure()
