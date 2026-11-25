"""Page 4 - training curriculum: what trains when and which losses apply."""
import math
from shapes import *
from page1 import minigraph, chev, rmat

W, H = 2400, 1100
PH = [(BLUE, "P1", "Feature learning"), (PURP, "P2", "Graph learning"), (CRIM, "P3", "Counterfactual"),
      (GREEN, "P4", "End-to-end")]
GREEN_D = "#2E6B31"
PY, PHGT = 130, 705


def headers(p, xs, y, w=132):
    for (c, tag, name), cx in zip(PH, xs):
        chev(p, cx - w / 2, y, w, 44, c, tag, fs=15)
        p.t(cx, y + 62, name, fs=12.5, bold=True, color=c if c != GREEN else GREEN_D)


# ---------------------------------------------------------------- module icons
def ic_backbone(p, cx, cy):
    trap(p, cx, cy, 40, 40, mix("#FFFFFF", SLATE, 0.6), SLATE, narrow="right", inset=10, grad=SLATE)


def ic_branches(p, cx, cy):
    cube(p, cx - 22, cy - 18, 20, 36, 8, mix("#FFFFFF", BLUE, 0.3), BLUE)
    cube(p, cx + 2, cy - 18, 20, 36, 8, mix("#FFFFFF", TEAL, 0.3), TEAL)


def ic_fc(color):
    def f(p, cx, cy):
        trap(p, cx - 10, cy, 16, 34, mix("#FFFFFF", color, 0.45), color, narrow="right", inset=7)
        op(p, cx + 14, cy, "\u03c3", d=18, fs=10, color=color, stroke=color, sw=1.3)
    return f


def ic_clip(p, cx, cy):
    trap(p, cx, cy, 40, 40, mix("#FFFFFF", AMBER, 0.7), AMBER, narrow="right", inset=10, grad=AMBER)


def ic_proj(p, cx, cy):
    trap(p, cx - 12, cy, 18, 36, mix("#FFFFFF", BLUE, 0.45), BLUE, narrow="right", inset=8)
    trap(p, cx + 12, cy, 18, 36, mix("#FFFFFF", AMBER, 0.5), AMBER, narrow="left", inset=8)


def ic_graph(p, cx, cy):
    minigraph(p, cx, cy, PURP, seed=4, n=5, rx=15, ry=13, r=5.5, ew=1.3)


def ic_mask(p, cx, cy):
    heat(p, cx - 16, cy - 16, 8, [[1, 0.1, 0.1, 1], [0.1, 1, 0.1, 0.1], [0.1, 0.1, 1, 0.1], [1, 0.1, 0.1, 1]],
         CRIM, lo="#FDF3F5", gap=0.6)


MODULES = [
    ("ResNet-50 + linear block", "backbone and 2048\u2192512 projection", ic_backbone, "TFFT"),
    ("Branch convolutions", "AU head and expression head trunks", ic_branches, "TFFT"),
    ("Base classifiers", "FC + \u03c3 on V<sub>a</sub> and V<sub>e</sub>", ic_fc(BLUE), "TFFF"),
    ("CLIP text encoder", "ViT-B/32, always frozen", ic_clip, "FFFF"),
    ("Contrastive projections", "visual / text heads into the shared space", ic_proj, "TFFF"),
    ("AU\u2013AU and AU\u2192Expr graphs", "dense dynamic graph layers", ic_graph, "FTTT"),
    ("Graph classifiers", "AU\u2013AU, graph-AU, graph-expression", ic_fc(PURP), "FTTT"),
    ("Masks and cutoffs \u03c4", "importance masks, learnable thresholds", ic_mask, "FFTT"),
]
LR = ["1\u00d710<sup>\u22124</sup>", "1\u00d710<sup>\u22124</sup>", "2\u00d710<sup>\u22125</sup>", "1\u00d710<sup>\u22125</sup>"]


def trainable_panel(p):
    panel(p, 30, PY, 1150, PHGT, BLUE, "What trains when", tag="set_phase()")
    xs = [610, 760, 910, 1060]
    headers(p, xs, 172)
    y0, dy = 262, 58
    for k, (name, sub_, icon, pat) in enumerate(MODULES):
        y = y0 + k * dy + 29
        if k % 2 == 0:
            box(p, 46, y - 29, 1118, 58, fill=mix("#FFFFFF", BLUE, 0.035), stroke="none", sw=0, arc=8, layer=1)
        icon(p, 84, y)
        p.t(124, y - 10, name, fs=14, bold=True, color=INK, align="left")
        p.t(124, y + 11, sub_, fs=11.5, color=MUTE, align="left")
        for (c, _, _), cx, s in zip(PH, xs, pat):
            if s == "T":
                ell(p, cx, y, 30, c, "#FFFFFF", sw=2, value="\u2713", fs=15, bold=True, color="#FFFFFF",
                    grad=mix(c, "#000000", 0.2))
            else:
                ell(p, cx, y, 30, "#E9F3FB", "#8DB7DC", sw=1.6, value="\u2744", fs=15, color="#4C8CC0")
    # learning-rate row
    y = y0 + len(MODULES) * dy + 14
    p.e([(50, y - 22), (1160, y - 22)], color=GREY_M, w=1.4, head=None, rounded=False, dashed=True)
    p.t(124, y - 8, "Initial learning rate", fs=14, bold=True, color=INK, align="left")
    p.t(124, y + 13, "AdamW with per-batch cosine decay", fs=11.5, color=MUTE, align="left")
    for (c, _, _), cx, v in zip(PH, xs, LR):
        pill(p, cx, y + 2, v, color=c if c != GREEN else GREEN_D, fill=mix("#FFFFFF", c, 0.16), fs=13, h=28, w=84,
             bold=True, stroke=c)
    ly = PY + PHGT - 26
    ell(p, 70, ly, 20, BLUE, "#FFFFFF", sw=1.5, value="\u2713", fs=11, bold=True, color="#FFFFFF")
    p.t(86, ly, "trainable", fs=12, color=MUTE, align="left")
    ell(p, 170, ly, 20, "#E9F3FB", "#8DB7DC", sw=1.4, value="\u2744", fs=11, color="#4C8CC0")
    p.t(186, ly, "frozen", fs=12, color=MUTE, align="left")
    p.t(1164, ly, "Phase 4 keeps CLIP, projections and base classifiers frozen; BatchNorm statistics can be frozen too.",
        fs=11.5, color=MUTE, align="right", italic=True)


# ---------------------------------------------------------------- loss matrix
LOSSES = [
    (Lc("wa"), "weighted asymmetric loss, AU head", ["1.0", "", "", ""]),
    (Lc("we"), "expression BCE vs M<sub>A\u2013E</sub> pseudo-label", ["0.05", "", "", ""]),
    (Lc("ib") + " \u00b7 " + Lc("align") + " \u00b7 " + Lc("decorr"), "HSIC disentanglement of AU embeddings",
     ["0.01", "", "", ""]),
    (Lc("con,AU") + " \u00b7 " + Lc("con,emo"), "CLIP text\u2013visual contrastive, active labels", ["0.05", "", "", ""]),
    (Lc("au-au"), "AU\u2013AU graph output", ["", "1.0", "1.0", "1.0"]),
    (Lc("graph-au") + " \u00b7 " + Lc("graph-emo"), "final graph classifiers", ["", "1.0", "1.0", "1.0"]),
    (Lc("dag"), "acyclicity of the AU\u2013AU adjacency", ["", "0.1", "0.1", "0.1"]),
    (Lc("facs"), "FACS rule violations on probabilities", ["", "0.05", "0.05", "0.05"]),
    (Lc("causal-au") + " \u00b7 " + Lc("causal-exp"), "edge polarity / importance violations", ["", "", "0.05", "0.05"]),
    (Lc("cf-au,imp") + " \u00b7 " + Lc("cf-au,unimp") + " \u00b7 " + Lc("cf-exp,imp") + " \u00b7 " + Lc("cf-exp,unimp"),
     "counterfactual change / preserve at both levels", ["", "", "0.05", "0.05"]),
]


def loss_panel(p):
    panel(p, 1210, PY, 1160, PHGT, CRIM, "Which losses apply", tag="weight per term")
    xs = [1840, 1980, 2120, 2260]
    headers(p, xs, 172)
    y0, dy = 262, 52
    for k, (name, desc, vals) in enumerate(LOSSES):
        y = y0 + k * dy + 26
        if k % 2 == 0:
            box(p, 1226, y - 26, 1128, 52, fill=mix("#FFFFFF", CRIM, 0.03), stroke="none", sw=0, arc=8, layer=1)
        p.t(1250, y - 10, name, fs=14, color=INK, align="left")
        p.t(1250, y + 12, desc, fs=11.5, color=MUTE, align="left")
        for (c, _, _), cx, v in zip(PH, xs, vals):
            if v:
                pill(p, cx, y, v, color=c if c != GREEN else GREEN_D, fill=mix("#FFFFFF", c, 0.16), fs=13, h=28, w=76,
                     bold=True, stroke=c)
            else:
                p.t(cx, y, "\u2013", fs=16, color=GREY_M)
    p.t(1250, PY + PHGT - 26, "Phases 3\u20134 share one objective; Phase 4 only changes which modules train.", fs=11.5,
        color=MUTE, align="left", italic=True)


# ---------------------------------------------------------------- protocol
def tile(p, x, y, w, h, color, title, lines):
    card(p, x, y, w, h, color, arc=12)
    p.t(x + 20, y + 26, title, fs=14, bold=True, color=color, align="left")
    for i, ln in enumerate(lines):
        p.t(x + 20, y + 54 + i * 21, ln, fs=12.5, color=INK if i == 0 else MUTE, align="left")


def protocol(p):
    y_panel = PY + PHGT + 40
    panel(p, 30, y_panel, 2340, 170, GREY, "Protocol and safeguards", tag="train.py \u00b7 config.py")
    tw, gap = 545, 53
    xs = [48 + k * (tw + gap) for k in range(4)]
    y = y_panel + 32
    tile(p, xs[0], y, tw, 120, BLUE, "Optimisation",
         ["AdamW (\u03b2 0.9 / 0.999), weight decay 5\u00d710<sup>\u22124</sup>",
          "batch 64, 20 epochs, no gradient clipping",
          "learning rate decays per batch (cosine), see the table"])
    tile(p, xs[1], y, tw, 120, PURP, "Checkpoint hand-off",
         ["each phase loads the previous phase's weights, strictly",
          "dataset, fold, AU vocabulary and loss schema are checked",
          "mismatched or non-finite tensors are rejected"])
    tile(p, xs[2], y, tw, 120, AMBER, "Prior knowledge",
         ["expression pseudo-labels from the SymGraphAU M<sub>A\u2013E</sub> prior",
          "primary 0.9 \u00b7 secondary 0.5 \u00b7 absent 0.1; no active AU = Neutral",
          "FACS rules drive the AU-level consistency loss"])
    tile(p, xs[3], y, tw, 120, GREEN, "Data",
         ["DISFA: 8 AUs   \u00b7   BP4D: 12 AUs   \u00b7   8 expressions",
          "3 folds: train on two, validate on one",
          "resize 256, centre-crop 224, class weights from the train split"])


def build():
    p = Page("4 \u00b7 Training curriculum", W, H, pid="curriculum")
    p.t(36, 50, "Training curriculum", fs=34, bold=True, color=INK, align="left")
    p.t(36, 88, "Four phases, each warm-started from the previous checkpoint \u00b7 what is trainable and which objective is active",
        fs=15, color=MUTE, align="left")
    trainable_panel(p)
    loss_panel(p)
    protocol(p)
    return p
