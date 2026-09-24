import xml.etree.ElementTree as ET
import os

def build_diagram():
    # 1. Read existing file to extract the exact base64 image style
    tree = ET.parse('ctrlau.drawio')
    root = tree.getroot()
    image_style = None
    for c in root.iter('mxCell'):
        if c.get('id') == 'GjfUm5Ax_Uw3Q_51WUAJ-1':
            image_style = c.get('style')
            break
            
    if not image_style:
        raise ValueError("Could not find image cell GjfUm5Ax_Uw3Q_51WUAJ-1 in ctrlau.drawio")

    print(f"Extracted image style of length {len(image_style)}")

    # 2. Build the new mxGraphModel XML
    # Using clean ID generation
    cells = []
    cell_id_counter = 100

    def get_id(prefix="cell"):
        nonlocal cell_id_counter
        cell_id_counter += 1
        return f"{prefix}_{cell_id_counter}"

    def add_node(cid, parent, val, x, y, w, h, style):
        cells.append({
            'type': 'node',
            'id': cid,
            'parent': parent,
            'value': val,
            'x': x, 'y': y, 'w': w, 'h': h,
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

    # Root layers
    cells.append({'type': 'node', 'id': '0', 'parent': None, 'value': '', 'style': ''})
    cells.append({'type': 'node', 'id': '1', 'parent': '0', 'value': '', 'style': ''})

    # =========================================================================
    # STYLES DEFINITION
    # =========================================================================
    FONT = "fontFamily=Helvetica;"
    STYLE_PHASE1_BOX = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=#b0bec5;strokeWidth=2;arcSize=4;{FONT}"
    STYLE_PHASE2_BOX = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f1f8e9;strokeColor=#81c784;strokeWidth=2;arcSize=4;{FONT}"
    
    STYLE_SUBBOX = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#cfd8dc;strokeWidth=1.5;arcSize=6;{FONT}"
    STYLE_SUBBOX_YELLOW = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fffdf7;strokeColor=#ffe082;strokeWidth=1.5;arcSize=6;{FONT}"
    STYLE_SUBBOX_GREEN = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f9fbe7;strokeColor=#c5e1a5;strokeWidth=1.5;arcSize=6;{FONT}"
    STYLE_SUBBOX_PURPLE = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#faf5ff;strokeColor=#d1c4e9;strokeWidth=1.5;arcSize=6;{FONT}"

    STYLE_TITLE = f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;{FONT}"
    STYLE_HEADER_BANNER = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#37474f;strokeColor=none;fontColor=#ffffff;align=left;spacingLeft=15;{FONT}"
    STYLE_HEADER_BANNER_P2 = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#2e7d32;strokeColor=none;fontColor=#ffffff;align=left;spacingLeft=15;{FONT}"

    STYLE_BACKBONE = f"shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=15;fillColor=#e3f2fd;strokeColor=#1976d2;strokeWidth=1.5;fontColor=#0d47a1;{FONT}"
    STYLE_CUBE = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.2;fontColor=#4a148c;{FONT}"
    STYLE_CUBE_PINK = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#fce4ec;strokeColor=#c2185b;strokeWidth=1.2;fontColor=#880e4f;{FONT}"
    STYLE_CUBE_GRAY = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#eceff1;strokeColor=#607d8b;strokeWidth=1.2;fontColor=#263238;{FONT}"
    STYLE_CUBE_AMBER = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.15;fillColor=#fff8e1;strokeColor=#ffa000;strokeWidth=1.2;fontColor=#ff6f00;{FONT}"

    STYLE_CNN_AU = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.2;fontColor=#311b92;fontSize=11;{FONT}"
    STYLE_CNN_EXP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fce4ec;strokeColor=#e91e63;strokeWidth=1.2;fontColor=#880e4f;fontSize=11;{FONT}"
    STYLE_LINEAR = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#9e9e9e;strokeWidth=1.2;fontColor=#212121;fontSize=11;{FONT}"
    STYLE_CLIP = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f8bbd0;strokeColor=#c2185b;strokeWidth=1.5;fontColor=#880e4f;fontSize=12;{FONT}"
    STYLE_FACS_MAT = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fff9c4;strokeColor=#fbc02d;strokeWidth=1.5;fontColor=#f57f17;fontSize=11;{FONT}"

    STYLE_BAR_AU = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ab47bc;strokeColor=#6a1b9a;strokeWidth=1;{FONT}"
    STYLE_BAR_EXP = f"rounded=0;whiteSpace=wrap;html=1;fillColor=#ec407a;strokeColor=#ad1457;strokeWidth=1;{FONT}"

    STYLE_LOSS_BADGE = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#d32f2f;strokeWidth=1.5;fontColor=#b71c1c;fontSize=11;{FONT}"
    STYLE_LOSS_BADGE_GREEN = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f5e9;strokeColor=#388e3c;strokeWidth=1.5;fontColor=#1b5e20;fontSize=11;{FONT}"
    STYLE_LOSS_BADGE_AMBER = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fffde7;strokeColor=#fbc02d;strokeWidth=1.5;fontColor=#f57f17;fontSize=11;{FONT}"
    STYLE_LOSS_BADGE_BLUE = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e1f5fe;strokeColor=#0288d1;strokeWidth=1.5;fontColor=#01579b;fontSize=11;{FONT}"

    STYLE_GRAPH_NODE_AU = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ede7f6;strokeColor=#7e57c2;strokeWidth=1.5;fontColor=#4a148c;fontStyle=1;fontSize=10;{FONT}"
    STYLE_GRAPH_NODE_EXP = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#fce4ec;strokeColor=#d81b60;strokeWidth=1.5;fontColor=#880e4f;fontStyle=1;fontSize=10;{FONT}"
    STYLE_GRAPH_BOX = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#a5d6a7;strokeWidth=1.5;{FONT}"

    STYLE_ARROW = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#546e7a;strokeWidth=1.5;entryX=0;entryY=0.5;exitX=1;exitY=0.5;"
    STYLE_ARROW_DASH = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#78909c;strokeWidth=1.2;dashed=1;dashPattern=4 3;"
    STYLE_ARROW_RED = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#d32f2f;strokeWidth=1.5;"
    STYLE_ARROW_GRAPH = "edgeStyle=straightEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#388e3c;strokeWidth=1.5;"

    # =========================================================================
    # 1. PHASE 1 CONTAINER & TITLE
    # =========================================================================
    # Box spans x: -730 to 710 (width 1440), y: -30 to 455 (height 485)
    box_p1 = add_node("box_phase1", "1", "", -730, -30, 1440, 485, STYLE_PHASE1_BOX)
    banner_p1 = add_node("banner_phase1", "1", 
        "<b>PHASE 1: MULTI-BRANCH FEATURE EXTRACTION, DISENTANGLEMENT & VISION-LANGUAGE GUIDANCE</b>",
        -715, -18, 1410, 32, STYLE_HEADER_BANNER)

    # -------------------------------------------------------------------------
    # (A) Visual Feature Extraction
    # -------------------------------------------------------------------------
    box_vis = add_node("box_vis_extract", "1", "", -715, 25, 410, 240, STYLE_SUBBOX)
    add_node("title_vis", "1", "<b>Visual Feature Extraction</b>", -700, 30, 200, 25, STYLE_TITLE)

    # Face Image (retaining exact base64 data)
    img_face = add_node("GjfUm5Ax_Uw3Q_51WUAJ-1", "1", "", -700, 85, 75, 75, image_style)
    add_node("lbl_face", "1", "Face Image<br><font style=\"font-size:9px;color:#666;\">3&times;224&times;224</font>", -705, 165, 85, 30, STYLE_TITLE)

    # Backbone ResNet-50
    node_backbone = add_node("node_backbone", "1", "<b>ResNet-50</b><br><font style=\"font-size:10px;\">Backbone</font>", -595, 90, 85, 65, STYLE_BACKBONE)
    
    # Feature Map z_img
    node_zimg = add_node("node_zimg", "1", "<b>z_{img}</b><br><font style=\"font-size:9px;\">49&times;2048</font>", -480, 80, 50, 75, STYLE_CUBE_GRAY)
    add_node("lbl_zimg", "1", "Spatial Grid<br><font style=\"font-size:9px;color:#666;\">7&times;7 Patches</font>", -490, 160, 80, 30, STYLE_TITLE)

    # LinearBlock (2048 -> 512)
    node_linear_proj = add_node("node_linear_proj", "1", "<b>LinearBlock</b><br><font style=\"font-size:9px;\">2048 &rarr; 512</font>", -400, 95, 75, 50, STYLE_LINEAR)

    # Flatten spatial grid feat (49x512)
    node_feat = add_node("node_feat", "1", "<b>feat</b><br><font style=\"font-size:9px;\">49&times;512</font>", -295, 88, 48, 65, STYLE_CUBE)
    add_node("lbl_feat", "1", "Patch Tokens", -295, 158, 70, 25, STYLE_TITLE)

    # Connections in Visual branch
    add_edge("e_face_bb", "1", "GjfUm5Ax_Uw3Q_51WUAJ-1", "node_backbone", "", STYLE_ARROW)
    add_edge("e_bb_zimg", "1", "node_backbone", "node_zimg", "", STYLE_ARROW)
    add_edge("e_zimg_lp", "1", "node_zimg", "node_linear_proj", "", STYLE_ARROW)
    add_edge("e_lp_feat", "1", "node_linear_proj", "node_feat", "", STYLE_ARROW)

    # -------------------------------------------------------------------------
    # (B) Vision-Language Guidance (CLIP)
    # -------------------------------------------------------------------------
    box_clip = add_node("box_clip_area", "1", "", -715, 275, 410, 165, STYLE_SUBBOX_PURPLE)
    add_node("title_clip", "1", "<b>Vision-Language Semantic Guidance</b>", -700, 280, 250, 25, STYLE_TITLE)

    # Text descriptions
    node_txt_au = add_node("node_txt_au", "1", "<b>AU Text Descriptions</b><br><font style=\"font-size:9px;\"><i>\"Inner Brow Raiser...\"</i></font>", -700, 310, 125, 42, STYLE_LINEAR)
    node_txt_exp = add_node("node_txt_exp", "1", "<b>Expr Text Descriptions</b><br><font style=\"font-size:9px;\"><i>\"Happiness, Sadness...\"</i></font>", -700, 375, 125, 42, STYLE_LINEAR)

    # CLIP Text Encoder
    node_clip_enc = add_node("node_clip_enc", "1", "<b>CLIP</b><br><font style=\"font-size:10px;\">Text Encoder<br><i>(Frozen ❄️)</i></font>", -545, 335, 75, 60, STYLE_CLIP)

    # Text Embeddings
    node_temb_au = add_node("node_temb_au", "1", "<b>T_a</b>", -440, 312, 38, 42, STYLE_CUBE_PINK)
    node_temb_exp = add_node("node_temb_exp", "1", "<b>T_e</b>", -440, 377, 38, 42, STYLE_CUBE_PINK)

    # Projection & Contrastive Badges
    node_loss_con_au = add_node("node_loss_con_au", "1", "Proj & InfoNCE<br><b style=\"color:#d32f2f;\">L_{con}^{AU}</b> <font style=\"font-size:8px;\">(y=1)</font>", -375, 310, 95, 44, STYLE_LOSS_BADGE)
    node_loss_con_exp = add_node("node_loss_con_exp", "1", "Proj & InfoNCE<br><b style=\"color:#d32f2f;\">L_{con}^{Exp}</b> <font style=\"font-size:8px;\">(y=1)</font>", -375, 375, 95, 44, STYLE_LOSS_BADGE)

    add_edge("e_tau_clip", "1", "node_txt_au", "node_clip_enc", "", STYLE_ARROW)
    add_edge("e_texp_clip", "1", "node_txt_exp", "node_clip_enc", "", STYLE_ARROW)
    add_edge("e_clip_tembau", "1", "node_clip_enc", "node_temb_au", "", STYLE_ARROW)
    add_edge("e_clip_tembexp", "1", "node_clip_enc", "node_temb_exp", "", STYLE_ARROW)
    add_edge("e_tembau_loss", "1", "node_temb_au", "node_loss_con_au", "", STYLE_ARROW)
    add_edge("e_tembexp_loss", "1", "node_temb_exp", "node_loss_con_exp", "", STYLE_ARROW)

    # -------------------------------------------------------------------------
    # (C) HSIC Disentanglement & Information Bottleneck Module
    # -------------------------------------------------------------------------
    box_hsic = add_node("box_hsic", "1", "", -290, 25, 420, 415, STYLE_SUBBOX_PURPLE)
    add_node("title_hsic", "1", "<b>HSIC Disentanglement & Bottleneck</b>", -275, 32, 280, 25, STYLE_TITLE)
    add_node("sub_hsic", "1", "<font style=\"font-size:10px;color:#555;\">Orthogonal Representation Learning via Hilbert-Schmidt Independence Criterion</font>", -275, 52, 390, 20, STYLE_TITLE)

    # 3 Distinct HSIC loss blocks
    node_lib = add_node("node_lib", "1", 
        "<b>Information Bottleneck:</b> &nbsp; <b style=\"color:#d32f2f;font-size:13px;\">L_{ib}</b><br>"
        "<font style=\"font-size:10px;\">min &nbsp; <b>HSIC(V_a, z_{img})</b></font><br>"
        "<font style=\"font-size:9px;color:#555;\">&bull; Compresses identity, head-pose & illumination noise</font>",
        -275, 80, 390, 65, STYLE_LOSS_BADGE_BLUE)

    node_lalign = add_node("node_lalign", "1", 
        "<b>Label Alignment:</b> &nbsp; <b style=\"color:#d32f2f;font-size:13px;\">L_{align}</b><br>"
        "<font style=\"font-size:10px;\">max &nbsp; <b>HSIC(V_a, Y_a)</b></font><br>"
        "<font style=\"font-size:9px;color:#555;\">&bull; Maximizes mutual information with AU ground truth</font>",
        -275, 160, 390, 65, STYLE_LOSS_BADGE_GREEN)

    node_ldecorr = add_node("node_ldecorr", "1", 
        "<b>Cross-AU Decorrelation:</b> &nbsp; <b style=\"color:#d32f2f;font-size:13px;\">L_{decorr}</b><br>"
        "<font style=\"font-size:10px;\">min &nbsp; <b>(2/N&sup2;) &sum;_{i &lt; j} HSIC(V_a^{(i)}, V_a^{(j)})</b></font><br>"
        "<font style=\"font-size:9px;color:#555;\">&bull; Eliminates feature collapse across distinct muscle units</font>",
        -275, 240, 390, 75, STYLE_LOSS_BADGE_AMBER)

    # Note on Phase 1 total loss
    add_node("node_p1_sum", "1", 
        "<b>Phase 1 Joint Feature Loss:</b><br>"
        "<font style=\"font-size:10px;\"><b>L_{phase1} = L_{wa} + &gamma;L_{we} + &lambda;_{ib}L_{ib} + &lambda;_{align}L_{align} + &lambda;_{decorr}L_{decorr} + &lambda;_{con}L_{con}</b></font>",
        -275, 335, 390, 50, STYLE_SUBBOX_YELLOW)

    # Connect z_img to L_ib
    add_edge("e_zimg_lib", "1", "node_zimg", "node_lib", "", STYLE_ARROW_DASH, [(-455, 175), (-455, 112), (-275, 112)])

    # -------------------------------------------------------------------------
    # (D) Multi-Branch Spatial Classification & FACS Prior
    # -------------------------------------------------------------------------
    box_multi = add_node("box_multibranch", "1", "", 145, 25, 555, 415, STYLE_SUBBOX_YELLOW)
    add_node("title_multi", "1", "<b>Multi-Branch Spatial Heads & FACS Prior</b>", 160, 30, 320, 25, STYLE_TITLE)

    # --- AU Branches ---
    add_node("lbl_au_head", "1", "<b>AU Classification Head (8 Branches)</b>", 160, 55, 250, 20, STYLE_TITLE)
    
    node_cnn_au1 = add_node("cnn_au1", "1", "1D CNN<br>(AU1)", 160, 80, 68, 36, STYLE_CNN_AU)
    node_cnn_au2 = add_node("cnn_au2", "1", "1D CNN<br>(AU2)", 160, 122, 68, 36, STYLE_CNN_AU)
    add_node("dots_au", "1", "<b>&vellip;</b>", 188, 158, 20, 20, STYLE_TITLE)
    node_cnn_au12 = add_node("cnn_au12", "1", "1D CNN<br>(AU12)", 160, 180, 68, 36, STYLE_CNN_AU)

    # AU Embeddings V_a (cubes)
    node_va1 = add_node("va_1", "1", "<b>V_a^{(1)}</b>", 252, 78, 42, 42, STYLE_CUBE)
    node_va2 = add_node("va_2", "1", "<b>V_a^{(2)}</b>", 252, 120, 42, 42, STYLE_CUBE)
    node_va12 = add_node("va_12", "1", "<b>V_a^{(12)}</b>", 252, 178, 42, 42, STYLE_CUBE)

    # AU Linear Classifiers
    node_lin_au1 = add_node("lin_au1", "1", "Linear", 318, 82, 52, 32, STYLE_LINEAR)
    node_lin_au2 = add_node("lin_au2", "1", "Linear", 318, 124, 52, 32, STYLE_LINEAR)
    node_lin_au12 = add_node("lin_au12", "1", "Linear", 318, 182, 52, 32, STYLE_LINEAR)

    # AU Prediction bars
    node_pau1 = add_node("pau1", "1", "", 392, 80, 10, 36, STYLE_BAR_AU)
    node_pau2 = add_node("pau2", "1", "", 392, 122, 10, 36, STYLE_BAR_AU)
    node_pau12 = add_node("pau12", "1", "", 392, 180, 10, 36, STYLE_BAR_AU)
    add_node("lbl_pau", "1", "<b>P_{AU}</b>", 410, 130, 40, 25, STYLE_TITLE)

    # Loss L_wa (Weighted Asymmetric Loss)
    node_loss_wa = add_node("node_loss_wa", "1", "<b style=\"color:#d32f2f;font-size:12px;\">L_{wa}</b><br><font style=\"font-size:9px;\">Weighted Asymmetric Loss (MultiviewSymAU)</font>", 460, 100, 125, 45, STYLE_LOSS_BADGE)

    # Connections AU branch
    add_edge("e_feat_au1", "1", "node_feat", "cnn_au1", "", STYLE_ARROW, [(-230, 120), (-230, 70), (140, 70), (140, 98), (160, 98)])
    add_edge("e_feat_au2", "1", "node_feat", "cnn_au2", "", STYLE_ARROW, [(-230, 120), (-230, 70), (140, 70), (140, 140), (160, 140)])
    add_edge("e_feat_au12", "1", "node_feat", "cnn_au12", "", STYLE_ARROW, [(-230, 120), (-230, 70), (140, 70), (140, 198), (160, 198)])

    add_edge("e_c1_v1", "1", "cnn_au1", "va_1", "", STYLE_ARROW)
    add_edge("e_c2_v2", "1", "cnn_au2", "va_2", "", STYLE_ARROW)
    add_edge("e_c12_v12", "1", "cnn_au12", "va_12", "", STYLE_ARROW)

    add_edge("e_v1_l1", "1", "va_1", "lin_au1", "", STYLE_ARROW)
    add_edge("e_v2_l2", "1", "va_2", "lin_au2", "", STYLE_ARROW)
    add_edge("e_v12_l12", "1", "va_12", "lin_au12", "", STYLE_ARROW)

    add_edge("e_l1_p1", "1", "lin_au1", "pau1", "", STYLE_ARROW)
    add_edge("e_l2_p2", "1", "lin_au2", "pau2", "", STYLE_ARROW)
    add_edge("e_l12_p12", "1", "lin_au12", "pau12", "", STYLE_ARROW)

    add_edge("e_pau_lwa", "1", "pau2", "node_loss_wa", "", STYLE_ARROW)

    # Connect V_a to HSIC Disentanglement box & CLIP AU contrastive loss
    add_edge("e_va_hsic", "1", "va_2", "node_ldecorr", "", STYLE_ARROW_DASH, [(235, 140), (235, 275), (115, 275)])
    add_edge("e_va_clip", "1", "va_12", "node_loss_con_au", "", STYLE_ARROW_DASH, [(270, 225), (270, 260), (-265, 260), (-265, 332), (-280, 332)])

    # --- FACS Knowledge Prior Matrix M_AE ---
    node_mae = add_node("node_mae", "1", 
        "<b>FACS Prior Matrix M_{AE} &isin; R^{8&times;7}</b><br>"
        "<font style=\"font-size:10px;\">Y_e = argmax(Y_a &middot; M_{AE}) &nbsp; (Neutral if &sum;Y_a = 0)</font>",
        430, 195, 255, 45, STYLE_FACS_MAT)
    add_edge("e_pau_mae", "1", "node_loss_wa", "node_mae", "", STYLE_ARROW_DASH)

    # --- Emotion Branches ---
    add_node("lbl_exp_head", "1", "<b>Expression Classification Head (7 Branches)</b>", 160, 240, 260, 20, STYLE_TITLE)

    node_cnn_e1 = add_node("cnn_e1", "1", "1D CNN<br>(E1: Angry)", 160, 265, 75, 36, STYLE_CNN_EXP)
    node_cnn_e2 = add_node("cnn_e2", "1", "1D CNN<br>(E2: Fear)", 160, 307, 75, 36, STYLE_CNN_EXP)
    add_node("dots_exp", "1", "<b>&vellip;</b>", 188, 343, 20, 20, STYLE_TITLE)
    node_cnn_e7 = add_node("cnn_e7", "1", "1D CNN<br>(E7: Neutral)", 160, 365, 75, 36, STYLE_CNN_EXP)

    # Emotion Embeddings V_e (cubes)
    node_ve1 = add_node("ve_1", "1", "<b>V_e^{(1)}</b>", 258, 263, 42, 42, STYLE_CUBE_PINK)
    node_ve2 = add_node("ve_2", "1", "<b>V_e^{(2)}</b>", 258, 305, 42, 42, STYLE_CUBE_PINK)
    node_ve7 = add_node("ve_7", "1", "<b>V_e^{(7)}</b>", 258, 363, 42, 42, STYLE_CUBE_PINK)

    # Emotion Linear Classifiers
    node_lin_e1 = add_node("lin_e1", "1", "Linear", 325, 267, 52, 32, STYLE_LINEAR)
    node_lin_e2 = add_node("lin_e2", "1", "Linear", 325, 309, 52, 32, STYLE_LINEAR)
    node_lin_e7 = add_node("lin_e7", "1", "Linear", 325, 367, 52, 32, STYLE_LINEAR)

    # Emotion Prediction bars
    node_pe1 = add_node("pe1", "1", "", 395, 265, 10, 36, STYLE_BAR_EXP)
    node_pe2 = add_node("pe2", "1", "", 395, 307, 10, 36, STYLE_BAR_EXP)
    node_pe7 = add_node("pe7", "1", "", 395, 365, 10, 36, STYLE_BAR_EXP)
    add_node("lbl_pe", "1", "<b>P_E</b>", 412, 315, 35, 25, STYLE_TITLE)

    # Loss L_we (Expression BCE Loss)
    node_loss_we = add_node("node_loss_we", "1", "<b style=\"color:#d32f2f;font-size:12px;\">L_{we}</b><br><font style=\"font-size:9px;\">Expression BCE Loss (FACS Pseudo-labels)</font>", 460, 305, 125, 45, STYLE_LOSS_BADGE)

    # Connections Emotion branch
    add_edge("e_feat_e1", "1", "node_feat", "cnn_e1", "", STYLE_ARROW, [(-230, 120), (-230, 230), (140, 230), (140, 283), (160, 283)])
    add_edge("e_feat_e2", "1", "node_feat", "cnn_e2", "", STYLE_ARROW, [(-230, 120), (-230, 230), (140, 230), (140, 325), (160, 325)])
    add_edge("e_feat_e7", "1", "node_feat", "cnn_e7", "", STYLE_ARROW, [(-230, 120), (-230, 230), (140, 230), (140, 383), (160, 383)])

    add_edge("e_ce1_ve1", "1", "cnn_e1", "ve_1", "", STYLE_ARROW)
    add_edge("e_ce2_ve2", "1", "cnn_e2", "ve_2", "", STYLE_ARROW)
    add_edge("e_ce7_ve7", "1", "cnn_e7", "ve_7", "", STYLE_ARROW)

    add_edge("e_ve1_le1", "1", "ve_1", "lin_e1", "", STYLE_ARROW)
    add_edge("e_ve2_le2", "1", "ve_2", "lin_e2", "", STYLE_ARROW)
    add_edge("e_ve7_le7", "1", "ve_7", "lin_e7", "", STYLE_ARROW)

    add_edge("e_le1_pe1", "1", "lin_e1", "pe1", "", STYLE_ARROW)
    add_edge("e_le2_pe2", "1", "lin_e2", "pe2", "", STYLE_ARROW)
    add_edge("e_le7_pe7", "1", "lin_e7", "pe7", "", STYLE_ARROW)

    add_edge("e_pe_lwe", "1", "pe2", "node_loss_we", "", STYLE_ARROW)
    add_edge("e_mae_lwe", "1", "node_mae", "node_loss_we", "", STYLE_ARROW)

    # Connect V_e to CLIP Emotion contrastive loss
    add_edge("e_ve_clip", "1", "ve_7", "node_loss_con_exp", "", STYLE_ARROW_DASH, [(280, 415), (-265, 415), (-265, 397), (-280, 397)])

    # =========================================================================
    # 2. PHASE 2 CONTAINER & TITLE
    # =========================================================================
    # Box spans x: -730 to 710 (width 1440), y: 470 to 815 (height 345)
    box_p2 = add_node("box_phase2", "1", "", -730, 470, 1440, 345, STYLE_PHASE2_BOX)
    banner_p2 = add_node("banner_phase2", "1", 
        "<b>PHASE 2: TWO-LEVEL CAUSAL GRAPH DISCOVERY & COUNTERFACTUAL REASONING</b> &nbsp;&nbsp;&nbsp;&nbsp; "
        "<span style=\"background-color:#ffebee;color:#c62828;padding:2px 8px;border-radius:3px;font-size:11px;border:1px solid #ef9a9a;\"><b>❄️ Backbone &amp; CNN Heads Frozen</b></span>",
        -715, 482, 1410, 32, STYLE_HEADER_BANNER_P2)

    # Transfer indicator from Phase 1 to Phase 2
    add_node("transfer_p1_p2", "1", 
        "<b>Transferred Embeddings: V_a &isin; R^{B&times;8&times;256} &nbsp;&amp;&nbsp; V_e &isin; R^{B&times;7&times;256}</b>",
        -480, 452, 450, 20, "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=#e8f5e9;fontColor=#2e7d32;fontSize=10;fontStyle=1;align=center;")

    # -------------------------------------------------------------------------
    # (E) Level 1: AU-AU Causal Graph Discovery
    # -------------------------------------------------------------------------
    box_au_au = add_node("box_au_au", "1", "", -715, 525, 455, 275, STYLE_SUBBOX_GREEN)
    add_node("title_au_au", "1", "<b>Level 1: AU-AU Causal Graph Discovery</b>", -700, 532, 280, 22, STYLE_TITLE)

    # Sub-graph visual box
    add_node("graph_box_au", "1", "", -700, 560, 215, 175, STYLE_GRAPH_BOX)
    add_node("lbl_graph_topo", "1", "<font style=\"font-size:9px;color:#555;\">Directed Causal DAG Topology</font>", -695, 565, 160, 15, STYLE_TITLE)

    # AU Nodes in circular/DAG topology
    gau_1 = add_node("gau_1", "1", "AU1", -685, 595, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_2 = add_node("gau_2", "1", "AU2", -635, 580, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_4 = add_node("gau_4", "1", "AU4", -565, 580, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_6 = add_node("gau_6", "1", "AU6", -685, 680, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_12 = add_node("gau_12", "1", "AU12", -615, 680, 28, 28, STYLE_GRAPH_NODE_AU)
    gau_25 = add_node("gau_25", "1", "AU25", -535, 645, 28, 28, STYLE_GRAPH_NODE_AU)

    # Directed causal edges (causal hierarchy)
    add_edge("eg_1_2", "1", "gau_1", "gau_2", "", STYLE_ARROW_GRAPH)
    add_edge("eg_2_4", "1", "gau_2", "gau_4", "", STYLE_ARROW_GRAPH)
    add_edge("eg_6_12", "1", "gau_6", "gau_12", "", STYLE_ARROW_GRAPH)
    add_edge("eg_12_25", "1", "gau_12", "gau_25", "", STYLE_ARROW_GRAPH)
    add_edge("eg_2_12", "1", "gau_2", "gau_12", "", STYLE_ARROW_GRAPH)

    # GAT Layer 1
    node_gat1 = add_node("node_gat1", "1", "<b>GAT Layer</b><br><font style=\"font-size:9px;\">Learns A_{AU-AU}</font>", -470, 560, 95, 42, STYLE_SUBBOX_YELLOW)

    # Soft DAG Acyclicity Loss
    node_loss_dag = add_node("node_loss_dag", "1", 
        "<b style=\"color:#d32f2f;font-size:12px;\">L_{DAG}</b><br>"
        "<font style=\"font-size:9px;\">tr(e^{A &comp; A}) - d = 0<br>(Acyclicity Constraint)</font>",
        -470, 612, 100, 48, STYLE_LOSS_BADGE)

    # Polarity & Importance Masks
    node_masks = add_node("node_masks", "1", 
        "<b>MaskModule</b><br><font style=\"font-size:9px;\">M_{pol} &nbsp;(Sign &plusmn;)<br>M_{imp} &nbsp;(Edge &bull;)</font>", 
        -470, 670, 95, 48, STYLE_SUBBOX_PURPLE)

    # Causal Violation Loss AU-AU
    node_loss_cau_au = add_node("node_loss_cau_au", "1", 
        "<b style=\"color:#d32f2f;font-size:12px;\">L_{causal}^{AU}</b><br>"
        "<font style=\"font-size:9px;\">Causal activation &amp;<br>polarity violation</font>", 
        -360, 560, 90, 45, STYLE_LOSS_BADGE)

    # Intermediate AU Prediction
    node_p_auau = add_node("node_p_auau", "1", 
        "Linear Classifiers<br>&rarr; <b>P_{AU-AU}</b><br>"
        "<b style=\"color:#d32f2f;\">L_{AU-AU}</b> (WAL)", 
        -360, 625, 90, 52, STYLE_SUBBOX_YELLOW)

    add_edge("e_gbox_gat1", "1", "graph_box_au", "node_gat1", "", STYLE_ARROW)
    add_edge("e_gat1_dag", "1", "node_gat1", "node_loss_dag", "", STYLE_ARROW)
    add_edge("e_gat1_masks", "1", "node_gat1", "node_masks", "", STYLE_ARROW)
    add_edge("e_masks_cauau", "1", "node_masks", "node_loss_cau_au", "", STYLE_ARROW)
    add_edge("e_gat1_pauau", "1", "node_gat1", "node_p_auau", "", STYLE_ARROW)

    # -------------------------------------------------------------------------
    # (F) Level 2: AU-Expression Bipartite Causal Graph
    # -------------------------------------------------------------------------
    box_bipartite = add_node("box_bipartite", "1", "", -245, 525, 480, 275, STYLE_SUBBOX_YELLOW)
    add_node("title_bipartite", "1", "<b>Level 2: AU &rarr; Expression Heterogeneous Graph</b>", -230, 532, 330, 22, STYLE_TITLE)

    # Bipartite Graph Box
    add_node("graph_box_exp", "1", "", -230, 560, 205, 175, STYLE_GRAPH_BOX)
    add_node("lbl_bipartite_topo", "1", "<font style=\"font-size:9px;color:#555;\">Bipartite AU &rarr; Expression Mapping</font>", -225, 565, 180, 15, STYLE_TITLE)

    # Left: AU Nodes
    bau_6 = add_node("bau_6", "1", "AU6", -220, 590, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_12 = add_node("bau_12", "1", "AU12", -220, 625, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_4 = add_node("bau_4", "1", "AU4", -220, 660, 26, 26, STYLE_GRAPH_NODE_AU)
    bau_1 = add_node("bau_1", "1", "AU1", -220, 695, 26, 26, STYLE_GRAPH_NODE_AU)

    # Right: Emotion Nodes
    bemo_hap = add_node("bemo_hap", "1", "Happy", -90, 595, 30, 30, STYLE_GRAPH_NODE_EXP)
    bemo_sad = add_node("bemo_sad", "1", "Sad", -90, 635, 30, 30, STYLE_GRAPH_NODE_EXP)
    bemo_fea = add_node("bemo_fea", "1", "Fear", -90, 675, 30, 30, STYLE_GRAPH_NODE_EXP)

    # Bipartite directed edges (FACS prototypes)
    add_edge("eb_6_hap", "1", "bau_6", "bemo_hap", "", STYLE_ARROW_GRAPH)
    add_edge("eb_12_hap", "1", "bau_12", "bemo_hap", "", STYLE_ARROW_GRAPH)
    add_edge("eb_4_sad", "1", "bau_4", "bemo_sad", "", STYLE_ARROW_GRAPH)
    add_edge("eb_1_sad", "1", "bau_1", "bemo_sad", "", STYLE_ARROW_GRAPH)
    add_edge("eb_1_fea", "1", "bau_1", "bemo_fea", "", STYLE_ARROW_GRAPH)

    # Bipartite GAT
    node_gat2 = add_node("node_gat2", "1", "<b>Bipartite GAT</b><br><font style=\"font-size:9px;\">Learns A_{AU-Exp}</font>", -10, 560, 100, 42, STYLE_SUBBOX_GREEN)

    # Causal Violation Loss AU-Exp
    node_loss_cau_exp = add_node("node_loss_cau_exp", "1", 
        "<b style=\"color:#d32f2f;font-size:12px;\">L_{causal}^{Exp}</b><br>"
        "<font style=\"font-size:9px;\">AU&rarr;Expr causal rule violation</font>", 
        -10, 612, 105, 48, STYLE_LOSS_BADGE)

    # FACS Emotion Violation Loss
    node_loss_facs_exp = add_node("node_loss_facs_exp", "1", 
        "<b style=\"color:#d32f2f;font-size:12px;\">L_{facs}^{Exp}</b><br>"
        "<font style=\"font-size:9px;\">Anatomical FACS rule consistency</font>", 
        -10, 670, 105, 48, STYLE_LOSS_BADGE_AMBER)

    # Final Graph Predictions P_G^{AU} and P_G^{Exp}
    node_p_final = add_node("node_p_final", "1", 
        "Graph Classifiers<br>"
        "<b>P_G^{AU}, P_G^{Exp}</b><br>"
        "<b style=\"color:#d32f2f;\">L_{wa}^{AU}, L_{we}^{Exp}</b>", 
        110, 615, 115, 60, STYLE_SUBBOX_PURPLE)

    add_edge("e_gbox_gat2", "1", "graph_box_exp", "node_gat2", "", STYLE_ARROW)
    add_edge("e_gat2_cauexp", "1", "node_gat2", "node_loss_cau_exp", "", STYLE_ARROW)
    add_edge("e_gat2_facsexp", "1", "node_gat2", "node_loss_facs_exp", "", STYLE_ARROW)
    add_edge("e_gat2_final", "1", "node_gat2", "node_p_final", "", STYLE_ARROW)

    # Connect AU-AU output to Bipartite Graph
    add_edge("e_pauau_gat2", "1", "node_p_auau", "node_gat2", "", STYLE_ARROW, [(-260, 650), (-260, 545), (-10, 545)])

    # -------------------------------------------------------------------------
    # (G) HiMod Feature-Level Counterfactual Intervention
    # -------------------------------------------------------------------------
    box_himod = add_node("box_himod", "1", "", 250, 525, 450, 275, STYLE_SUBBOX)
    add_node("title_himod", "1", "<b>HiMod Feature-Level Counterfactual Intervention</b>", 265, 532, 360, 22, STYLE_TITLE)

    # Partition based on importance
    node_cf_part = add_node("node_cf_part", "1", 
        "<b>Graph-Guided Causal Importance Partition</b><br>"
        "<font style=\"font-size:10px;\">diag(M_{imp}) &gt; &tau; &nbsp; &rarr; &nbsp; <b>Important AUs</b> vs <b>Unimportant AUs</b></font>", 
        265, 560, 420, 38, STYLE_SUBBOX_YELLOW)

    # Important branch
    node_cf_imp_box = add_node("node_cf_imp_box", "1", 
        "<b>Perturb Important Nodes:</b><br>"
        "<font style=\"font-size:9px;\">V_a^{imp} = V_a + &epsilon; &middot; M_{imp} &nbsp; (&epsilon; &sim; N(0,&sigma;&sup2;))</font>", 
        265, 612, 230, 42, STYLE_SUBBOX_PURPLE)

    node_loss_cf_imp = add_node("node_loss_cf_imp", "1", 
        "<b style=\"color:#d32f2f;font-size:12px;\">L_{cf}^{imp}</b> (Invariance Violation)<br>"
        "<font style=\"font-size:9px;\">Predictions <b>MUST change</b> when key causal triggers are corrupted</font>", 
        508, 608, 175, 48, STYLE_LOSS_BADGE)

    # Unimportant branch
    node_cf_unimp_box = add_node("node_cf_unimp_box", "1", 
        "<b>Perturb Unimportant Nodes:</b><br>"
        "<font style=\"font-size:9px;\">V_a^{unimp} = V_a + &epsilon; &middot; (1 - M_{imp})</font>", 
        265, 672, 230, 42, STYLE_SUBBOX_GREEN)

    node_loss_cf_unimp = add_node("node_loss_cf_unimp", "1", 
        "<b style=\"color:#2e7d32;font-size:12px;\">L_{cf}^{unimp}</b> (Consistency)<br>"
        "<font style=\"font-size:9px;\">Predictions <b>MUST stay robust</b> under non-causal perturbations</font>", 
        508, 668, 175, 48, STYLE_LOSS_BADGE_GREEN)

    # Summary of Phase 2 total loss
    add_node("node_p2_sum", "1", 
        "<b>Phase 2 Total Objective:</b> &nbsp; "
        "<font style=\"font-size:10px;\"><b>L_{phase2} = &lambda;_{dag}L_{DAG} + &lambda;_{cau}L_{causal} + &lambda;_{facs}L_{facs}^{Exp} + &lambda;_{cf}L_{cf} + &lambda;_G L_G</b></font>",
        265, 735, 420, 35, STYLE_SUBBOX_YELLOW)

    add_edge("e_part_imp", "1", "node_cf_part", "node_cf_imp_box", "", STYLE_ARROW)
    add_edge("e_part_unimp", "1", "node_cf_part", "node_cf_unimp_box", "", STYLE_ARROW)
    add_edge("e_imp_loss", "1", "node_cf_imp_box", "node_loss_cf_imp", "", STYLE_ARROW)
    add_edge("e_unimp_loss", "1", "node_cf_unimp_box", "node_loss_cf_unimp", "", STYLE_ARROW)

    # Connect Masks from AU-AU to HiMod
    add_edge("e_masks_himod", "1", "node_masks", "node_cf_part", "", STYLE_ARROW_DASH, [(-375, 694), (-375, 510), (475, 510), (475, 560)])

    # =========================================================================
    # BUILD FINAL XML
    # =========================================================================
    offset_x = 770
    offset_y = 50

    mxfile = ET.Element('mxfile', host='app.diagrams.net')
    diagram = ET.SubElement(mxfile, 'diagram', name='CtrlAU Architecture', id='cQhNqvIMsQzl_Nc4XbVt')
    graph_model = ET.SubElement(diagram, 'mxGraphModel', 
                                grid='0', page='1', gridSize='10', 
                                guides='1', tooltips='1', connect='1', arrows='1', fold='1', pageScale='1', 
                                pageWidth='1600', pageHeight='1000', math='1', shadow='0')
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
            geom.set('x', str(round(c['x'] + offset_x, 1)))
            geom.set('y', str(round(c['y'] + offset_y, 1)))
            geom.set('width', str(round(c['w'], 1)))
            geom.set('height', str(round(c['h'], 1)))
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
                    mxpt.set('x', str(round(pt[0] + offset_x, 1)))
                    mxpt.set('y', str(round(pt[1] + offset_y, 1)))

    out_xml = ET.tostring(mxfile, encoding='utf-8')
    with open('ctrlau.drawio', 'wb') as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(out_xml)
        
    print(f"Successfully generated ctrlau.drawio with {len(cells)} cells and valid XML!")

if __name__ == '__main__':
    build_diagram()
