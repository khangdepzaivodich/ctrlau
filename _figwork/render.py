"""Render one page of a .drawio file to PNG using the real draw.io viewer in headless Edge.

usage: python render.py file.drawio PAGE_INDEX out.png [scale] [x y w h]
Optional crop (x y w h, in diagram px) is rendered at `scale` via a CSS transform, so it stays vector-crisp.
"""
import json, os, subprocess, sys, xml.etree.ElementTree as ET

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    src, page, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    crop = [float(v) for v in sys.argv[5:9]] if len(sys.argv) >= 9 else None

    tree = ET.parse(src)
    d = tree.getroot().findall("diagram")[page]
    model = d.find("mxGraphModel")
    pw = int(float(model.get("pageWidth", "1600")))
    ph = int(float(model.get("pageHeight", "1000")))
    single = ET.Element("mxfile")
    single.append(d)
    xml = ET.tostring(single, encoding="unicode")

    cfg = {"highlight": "#0000ff", "nav": False, "resize": False, "toolbar": None, "border": 0, "xml": xml}
    if crop:
        cx, cy, cw, chh = crop
        win_w, win_h = int(cw * scale), int(chh * scale)
        tf = f"transform-origin:0 0;transform:translate({-cx * scale}px,{-cy * scale}px) scale({scale});"
    else:
        win_w, win_h = int(pw * scale), int(ph * scale)
        tf = f"transform-origin:0 0;transform:scale({scale});" if scale != 1 else ""
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{{margin:0;padding:0;background:#fff;overflow:hidden}}
.mxgraph{{max-width:none !important;}}</style></head>
<body><div id="host" style="{tf}width:{pw}px;height:{ph}px"></div>
<script src="viewer-static.min.js"></script>
<script>
var cfg = {json.dumps(cfg)};
var d = document.createElement('div'); d.className = 'mxgraph';
d.setAttribute('data-mxgraph', JSON.stringify(cfg));
document.getElementById('host').appendChild(d);
GraphViewer.processElements();
</script></body></html>"""
    html_path = os.path.join(HERE, "view.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    out_abs = os.path.abspath(out)
    cmd = [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
           f"--window-size={win_w},{win_h}", "--virtual-time-budget=12000", f"--screenshot={out_abs}",
           "file:///" + html_path.replace("\\", "/")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    msg = (r.stdout + r.stderr).strip().splitlines()
    print([m for m in msg if "written" in m] or msg[-2:])


if __name__ == "__main__":
    main()
