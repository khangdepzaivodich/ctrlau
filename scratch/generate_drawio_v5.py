import xml.etree.ElementTree as ET

def create_cell(root, id, value, style, x, y, w, h, parent="1"):
    cell = ET.SubElement(root, "mxCell", id=id, value=value, style=style, vertex="1", parent=parent)
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
    return cell

def create_edge(root, id, src, tgt, style, waypoints=None, parent="1"):
    cell = ET.SubElement(root, "mxCell", id=id, style=style, edge="1", parent=parent, source=src, target=tgt)
    geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
    if waypoints:
        arr = ET.SubElement(geo, "Array", **{"as": "points"})
        for wx, wy in waypoints:
            ET.SubElement(arr, "mxPoint", x=str(wx), y=str(wy))
    return cell

def generate_drawio():
    mxfile = ET.Element("mxfile", version="14.6.11")
    diagram = ET.SubElement(mxfile, "diagram", id="ctrlau_master", name="CtrlAU Master")
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="2000", dy="2000", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="3000", pageHeight="2000", math="1", shadow="0")
    root = ET.SubElement(mxGraphModel, "root")
    
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    # ================= STYLES =================
    s_bg = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8f9fa;strokeColor=#cfd8dc;strokeWidth=2;dashed=1;verticalAlign=top;align=left;spacingLeft=15;spacingTop=15;fontSize=22;fontStyle=1;"
    s_mod = "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#37474f;strokeWidth=2;fontStyle=1;fontSize=14;"
    s_tensor = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.1;darkOpacity2=0.1;fillColor=#e3f2fd;strokeColor=#1565c0;strokeWidth=2;fontSize=14;"
    s_tensor_g = "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.1;darkOpacity2=0.1;fillColor=#e8f5e9;strokeColor=#2e7d32;strokeWidth=2;fontSize=14;"
    s_op = "shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#ffecb3;strokeColor=#ff8f00;strokeWidth=2;fontSize=20;fontStyle=1;"
    s_loss = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffebee;strokeColor=#c62828;strokeWidth=2;dashed=1;fontColor=#b71c1c;fontSize=14;fontStyle=1;"
    s_text = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=16;fontStyle=1;"
    s_arrow = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#263238;endArrow=block;endFill=1;"
    s_arrow_dash = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;strokeColor=#c62828;endArrow=block;endFill=1;dashed=1;"
    s_arrow_straight = "edgeStyle=none;html=1;strokeWidth=2;strokeColor=#263238;endArrow=block;endFill=1;"

    # ================= 1. MAIN FLOW (TOP) =================
    create_cell(root, "bg_main", "MAIN PIPELINE FLOW", s_bg, 50, 50, 2400, 250)
    
    create_cell(root, "m_img", "Input Image\n$X$", s_mod, 100, 150, 120, 60)
    create_cell(root, "m_res", "ResNet-50\n(Backbone)", s_mod, 300, 150, 120, 60)
    create_cell(root, "m_feat", "$Z$", s_tensor, 500, 140, 80, 80)
    create_cell(root, "m_heads", "Multiview Heads\n(See Module A)", s_mod + "fillColor=#fff9c4;", 660, 150, 140, 60)
    create_cell(root, "m_basev", "$V_{base}$", s_tensor_g, 880, 140, 80, 80)
    
    create_cell(root, "m_clip", "CLIP Text\nEncoder", s_mod, 860, 220, 120, 60)
    
    create_cell(root, "m_null", "Semantic Nulling\n(See Module B)", s_mod + "fillColor=#fff9c4;", 1060, 150, 140, 60)
    create_cell(root, "m_nullv", "$\hat{V}$", s_tensor_g, 1280, 140, 80, 80)
    
    create_cell(root, "m_gat", "Dense Graph Conv\n(See Module C)", s_mod + "fillColor=#fff9c4;", 1460, 150, 140, 60)
    create_cell(root, "m_gatv", "$V_{out}$", s_tensor_g, 1680, 140, 80, 80)
    
    create_cell(root, "m_class", "Classifiers", s_mod, 1840, 150, 120, 60)
    create_cell(root, "m_prob", "$\hat{Y}$", s_tensor_g, 2040, 140, 80, 80)
    
    create_cell(root, "m_abd", "Abductive Inference\n(See Module D)", s_mod + "fillColor=#fff9c4;", 2220, 150, 140, 60)

    # Connections Main Flow
    create_edge(root, "em1", "m_img", "m_res", s_arrow_straight)
    create_edge(root, "em2", "m_res", "m_feat", s_arrow_straight)
    create_edge(root, "em3", "m_feat", "m_heads", s_arrow_straight)
    create_edge(root, "em4", "m_heads", "m_basev", s_arrow_straight)
    create_edge(root, "em5", "m_basev", "m_null", s_arrow_straight)
    create_edge(root, "em_clip_null", "m_clip", "m_null", s_arrow)
    create_edge(root, "em6", "m_null", "m_nullv", s_arrow_straight)
    create_edge(root, "em7", "m_nullv", "m_gat", s_arrow_straight)
    create_edge(root, "em8", "m_gat", "m_gatv", s_arrow_straight)
    create_edge(root, "em9", "m_gatv", "m_class", s_arrow_straight)
    create_edge(root, "em10", "m_class", "m_prob", s_arrow_straight)
    create_edge(root, "em11", "m_prob", "m_abd", s_arrow_straight)

    # ================= 2. MODULE A: AU & EMOTION HEADS =================
    create_cell(root, "bg_moda", "MODULE A: 1000% Detailed Multiview Extraction Heads (SymAU / SymExpr)", s_bg, 50, 350, 1150, 600)
    
    create_cell(root, "ma_in", "Input Features $Z$\n$C \\times H \\times W$", s_tensor, 100, 580, 100, 100)
    
    # 8 Parallel Branches
    create_cell(root, "ma_split", "", "endArrow=none;html=1;strokeWidth=3;", 280, 450, 0, 300)
    create_edge(root, "ema_in", "ma_in", "ma_split", s_arrow_straight)
    
    def draw_branch(y_pos, lbl, idx):
        create_edge(root, f"ema_b{idx}", "ma_split", f"ma_conv_{idx}", s_arrow_straight, waypoints=[(280, y_pos+30)])
        create_cell(root, f"ma_conv_{idx}", "1x1 Conv\nSpatial Attn", s_mod, 320, y_pos, 100, 60)
        create_cell(root, f"ma_sig_{idx}", "Sigmoid", s_mod + "fillColor=#e1bee7;", 460, y_pos, 80, 60)
        create_cell(root, f"ma_mul_{idx}", "$\otimes$", s_op, 580, y_pos+10, 40, 40)
        create_cell(root, f"ma_gap_{idx}", "Global Avg\nPool (GAP)", s_mod, 660, y_pos, 100, 60)
        create_cell(root, f"ma_out_{idx}", f"Branch {idx}\n$V_{{{idx}}}$", s_tensor_g, 800, y_pos, 80, 60)
        
        create_edge(root, f"ema_c{idx}", f"ma_conv_{idx}", f"ma_sig_{idx}", s_arrow_straight)
        create_edge(root, f"ema_s{idx}", f"ma_sig_{idx}", f"ma_mul_{idx}", s_arrow_straight)
        # Residual connection from Z to multiply
        create_edge(root, f"ema_res{idx}", "ma_in", f"ma_mul_{idx}", s_arrow, waypoints=[(240, 580), (240, y_pos+80), (600, y_pos+80)])
        create_edge(root, f"ema_m{idx}", f"ma_mul_{idx}", f"ma_gap_{idx}", s_arrow_straight)
        create_edge(root, f"ema_g{idx}", f"ma_gap_{idx}", f"ma_out_{idx}", s_arrow_straight)

    draw_branch(420, "Branch 1", 1)
    create_cell(root, "ma_dots", "...", s_text, 500, 520, 60, 40)
    draw_branch(600, "Branch 8", 8)
    
    create_cell(root, "ma_concat", "Concat / Stack\n$[V_1, V_2, ..., V_8]$", s_mod, 940, 500, 140, 80)
    create_edge(root, "ema_o1", "ma_out_1", "ma_concat", s_arrow, waypoints=[(900, 450), (900, 540)])
    create_edge(root, "ema_o8", "ma_out_8", "ma_concat", s_arrow, waypoints=[(900, 630), (900, 540)])
    
    create_cell(root, "ma_final", "Final AU\nEmbeddings $V_{AU}$", s_tensor_g, 1120, 500, 80, 80)
    create_edge(root, "ema_fin", "ma_concat", "ma_final", s_arrow_straight)

    # ================= 3. MODULE B: SEMANTIC NULLING =================
    create_cell(root, "bg_modb", "MODULE B: 1000% Detailed Semantic Nulling (Idea 1 & 1.1)", s_bg, 1250, 350, 1200, 600)
    
    create_cell(root, "mb_traw", "Raw Text $T_{raw}$\n(from CLIP)", s_tensor, 1300, 450, 80, 80)
    
    # Gram Schmidt Loop box
    create_cell(root, "mb_gs_box", "Iterative Gram-Schmidt Orthogonalization", s_mod + "dashed=1;fillColor=none;align=left;verticalAlign=top;spacingLeft=10;", 1450, 400, 400, 200)
    create_cell(root, "mb_gs_eq1", "$$proj_{u}(v) = \\frac{v \\cdot u}{u \\cdot u} u$$", s_text, 1480, 450, 150, 40)
    create_cell(root, "mb_gs_eq2", "$$u_i = v_i - \sum proj_{u_j}(v_i)$$", s_text, 1480, 510, 150, 40)
    create_cell(root, "mb_gs_loop", "For $i=1...N$", "shape=step;fillColor=#e3f2fd;", 1700, 470, 100, 60)
    
    create_cell(root, "mb_tortho", "Orthogonal Basis\n$T_{ortho}$", s_tensor_g, 1920, 450, 80, 80)
    
    create_edge(root, "emb_1", "mb_traw", "mb_gs_box", s_arrow_straight)
    create_edge(root, "emb_2", "mb_gs_box", "mb_tortho", s_arrow_straight)
    
    # Projection and subtraction
    create_cell(root, "mb_v", "Input $V$\n(From Mod A)", s_tensor, 1300, 700, 80, 80)
    
    create_cell(root, "mb_dot", "Dot Product\n$(V \cdot T_{ortho})$", s_op, 1600, 680, 80, 80)
    create_edge(root, "emb_3", "mb_v", "mb_dot", s_arrow)
    create_edge(root, "emb_4", "mb_tortho", "mb_dot", s_arrow, waypoints=[(1640, 490)])
    
    create_cell(root, "mb_scale", "Scalar Multiply\n$* T_{ortho}$", s_op, 1750, 680, 80, 80)
    create_edge(root, "emb_5", "mb_dot", "mb_scale", s_arrow_straight)
    create_edge(root, "emb_6", "mb_tortho", "mb_scale", s_arrow, waypoints=[(1790, 490)])
    
    create_cell(root, "mb_sub", "Subtract\n$\ominus$", s_op, 1950, 680, 80, 80)
    create_edge(root, "emb_7", "mb_v", "mb_sub", s_arrow, waypoints=[(1340, 820), (1990, 820)])
    create_edge(root, "emb_8", "mb_scale", "mb_sub", s_arrow_straight)
    
    create_cell(root, "mb_out", "Nulled Subspace\n$\hat{V}$", s_tensor_g, 2150, 700, 80, 80)
    create_edge(root, "emb_9", "mb_sub", "mb_out", s_arrow_straight)

    # ================= 4. MODULE C: DYNAMIC GRAPH =================
    create_cell(root, "bg_modc", "MODULE C: 1000% Detailed Dense Dynamic Graph Conv & DAG (Idea 2)", s_bg, 50, 1000, 1150, 700)
    
    create_cell(root, "mc_in", "Input $\hat{V}$", s_tensor, 100, 1300, 80, 80)
    
    create_cell(root, "mc_wq", "Linear $W_Q$", s_mod, 260, 1150, 100, 60)
    create_cell(root, "mc_wk", "Linear $W_K$", s_mod, 260, 1250, 100, 60)
    create_cell(root, "mc_wv", "Linear $W_V$", s_mod, 260, 1450, 100, 60)
    
    create_edge(root, "emc_1", "mc_in", "mc_wq", s_arrow, waypoints=[(220, 1340), (220, 1180)])
    create_edge(root, "emc_2", "mc_in", "mc_wk", s_arrow, waypoints=[(220, 1340), (220, 1280)])
    create_edge(root, "emc_3", "mc_in", "mc_wv", s_arrow, waypoints=[(220, 1340), (220, 1480)])
    
    create_cell(root, "mc_q", "$Q$", s_tensor, 420, 1140, 60, 80)
    create_cell(root, "mc_k", "$K^T$", s_tensor, 420, 1240, 60, 80)
    create_edge(root, "emc_4", "mc_wq", "mc_q", s_arrow_straight)
    create_edge(root, "emc_5", "mc_wk", "mc_k", s_arrow_straight)
    
    create_cell(root, "mc_matmul1", "MatMul\n$\otimes$", s_op, 540, 1170, 80, 80)
    create_edge(root, "emc_6", "mc_q", "mc_matmul1", s_arrow)
    create_edge(root, "emc_7", "mc_k", "mc_matmul1", s_arrow)
    
    create_cell(root, "mc_sig", "Sigmoid\n$\sigma$", s_mod + "fillColor=#e1bee7;", 680, 1180, 80, 60)
    create_edge(root, "emc_8", "mc_matmul1", "mc_sig", s_arrow_straight)
    
    create_cell(root, "mc_gdyn", "Dynamic Matrix\n$\mathcal{G}_{dyn}$", s_tensor_g, 820, 1170, 80, 80)
    create_edge(root, "emc_9", "mc_sig", "mc_gdyn", s_arrow_straight)
    
    # Global Matrix
    create_cell(root, "mc_ginv", "Learnable Matrix\n$\mathcal{G}_{inv}$", s_tensor_g + "fillColor=#fff9c4;", 820, 1350, 80, 80)
    create_cell(root, "mc_dag", "DAG Regularization\n$Tr(e^{\mathcal{G} \circ \mathcal{G}}) - d$", s_loss, 600, 1370, 150, 40)
    create_edge(root, "emc_dag", "mc_dag", "mc_ginv", s_arrow_dash)
    
    # Mask Modulation
    create_cell(root, "mc_mask", "Importance Mask\n$M_{imp}$", s_tensor, 820, 1530, 80, 80)
    
    create_cell(root, "mc_mul_all", "Element-wise Multiply\n$\odot$", s_op, 1000, 1350, 80, 80)
    create_edge(root, "emc_10", "mc_gdyn", "mc_mul_all", s_arrow)
    create_edge(root, "emc_11", "mc_ginv", "mc_mul_all", s_arrow_straight)
    create_edge(root, "emc_12", "mc_mask", "mc_mul_all", s_arrow)
    
    create_cell(root, "mc_gfinal", "Final Graph\n$\mathcal{G}_{final}$", s_tensor_g, 1140, 1350, 80, 80)
    create_edge(root, "emc_13", "mc_mul_all", "mc_gfinal", s_arrow_straight)

    # ================= 5. MODULE D: ABDUCTIVE INFERENCE =================
    create_cell(root, "bg_modd", "MODULE D: 1000% Detailed Abductive Inference & Cycle Consistency", s_bg, 1250, 1000, 1200, 700)
    
    create_cell(root, "md_pau", "Initial AU Probs\n$P_{AU}$", s_tensor, 1300, 1150, 80, 80)
    create_cell(root, "md_pexp", "Initial Exp Probs\n$P_{Exp}$", s_tensor, 1300, 1450, 80, 80)
    
    # Cycle Consistency
    create_cell(root, "md_mae", "Prior Projection\n$M_{AE}$ Matrix", s_mod + "fillColor=#fff9c4;", 1460, 1550, 120, 60)
    create_cell(root, "md_proj_op", "MatMul\n$\otimes$", s_op, 1640, 1450, 80, 80)
    create_edge(root, "emd_1", "md_pexp", "md_proj_op", s_arrow_straight)
    create_edge(root, "emd_2", "md_mae", "md_proj_op", s_arrow)
    
    create_cell(root, "md_p_pseudo", "Expected AUs\n$P_{pseudo}$", s_tensor_g, 1780, 1450, 80, 80)
    create_edge(root, "emd_3", "md_proj_op", "md_p_pseudo", s_arrow_straight)
    
    create_cell(root, "md_mse", "MSE Loss\n$L_{Cycle}$", s_loss, 1950, 1300, 100, 60)
    create_edge(root, "emd_4", "md_p_pseudo", "md_mse", s_arrow)
    create_edge(root, "emd_5", "md_pau", "md_mse", s_arrow, waypoints=[(1400, 1190), (2000, 1190)])
    
    # FACS Rules
    create_cell(root, "md_facs_box", "FACS Logic Constraints\n(e.g. Happiness $\implies$ AU6 + AU12)", s_mod, 1640, 1150, 200, 60)
    create_cell(root, "md_facs_loss", "$L_{FACS}$", s_loss, 1950, 1150, 100, 60)
    create_edge(root, "emd_6", "md_pau", "md_facs_box", s_arrow_straight)
    create_edge(root, "emd_7", "md_pexp", "md_facs_box", s_arrow, waypoints=[(1500, 1490), (1500, 1180)])
    create_edge(root, "emd_8", "md_facs_box", "md_facs_loss", s_arrow_straight)
    
    # Optimizer Loop
    create_cell(root, "md_opt", "Test-Time Adam\nOptimizer\n$\min_P (L_{FACS} + L_{Cycle})$", s_op + "fillColor=#dcedc8;strokeColor=#689f38;", 2150, 1200, 150, 150)
    create_edge(root, "emd_9", "md_mse", "md_opt", s_arrow_dash)
    create_edge(root, "emd_10", "md_facs_loss", "md_opt", s_arrow_dash)
    
    # Update arrow
    create_edge(root, "emd_11", "md_opt", "md_pau", "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=4;strokeColor=#689f38;endArrow=block;endFill=1;", waypoints=[(2225, 1050), (1340, 1050)])
    create_edge(root, "emd_12", "md_opt", "md_pexp", "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=4;strokeColor=#689f38;endArrow=block;endFill=1;", waypoints=[(2225, 1050), (1340, 1050)])
    
    # Save to file
    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ", level=0)
    tree.write("ctrlau_1000_percent_detailed.drawio", encoding="utf-8", xml_declaration=True)
    print("Master Drawio generated!")

if __name__ == "__main__":
    generate_drawio()
