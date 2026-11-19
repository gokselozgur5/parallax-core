import os

files = [
    "parallax/dispatch/sliding_window.py",
    "parallax/dispatch/layernorm.py",
    "parallax/dispatch/rmsnorm_backward.py",
    "parallax/dispatch/gelu_fast.py",
    "parallax/dispatch/fused_layernorm.py"
]

for f in files:
    path = os.path.join("/tmp/wt/personal/parallax-core/refactor-import-paths", f)
    with open(path, "r") as fr:
        content = fr.read()
    content = content.replace("from python.parallax", "from parallax")
    with open(path, "w") as fw:
        fw.write(content)
