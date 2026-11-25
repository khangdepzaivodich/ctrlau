import importlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figlib
from figlib import write_drawio

mods = sys.argv[1].split(",") if len(sys.argv) > 1 else ["page1"]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.drawio")
pages = []
for m in mods:
    mod = importlib.import_module(m)
    pages.append(mod.build())
write_drawio(out, pages)
print("wrote", out, [len(pg.layers[0]) + sum(len(l) for l in pg.layers[1:]) for pg in pages])
