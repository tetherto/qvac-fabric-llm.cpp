import os, sys
R = "/Users/ishan/qwen38-accuracy-20261029"
H = "/Users/ishan/ffn-probe-20261029/heldout"
def text(p):
    return open(p, encoding="utf-8", errors="replace").read()
def window(src, frac, size):
    s = text(src)
    start = int(len(s) * frac)
    start = s.find("\n", start) + 1
    w = s[start:start + size]
    assert len(w) == size, (src, len(s), frac, size)
    return w
spec = {
    "w2048": [
        ("cpp-graph", f"{R}/src/llama-graph.cpp", 0.40), ("cpp-arg", f"{R}/common/arg.cpp", 0.40),
        ("cpp-backend", f"{R}/ggml/src/ggml-backend.cpp", 0.40), ("cpp-vocab", f"{R}/src/llama-vocab.cpp", 0.40),
        ("py-base-a", f"{R}/conversion/base.py", 0.20), ("py-writer", f"{R}/gguf-py/gguf/gguf_writer.py", 0.30),
        ("py-quants", f"{R}/gguf-py/gguf/quants.py", 0.30),
        ("doc-sycl", f"{R}/docs/backend/SYCL.md", 0.20), ("doc-autoparser", f"{R}/docs/autoparser.md", 0.20),
        ("doc-openvino", f"{R}/docs/backend/OPENVINO.md", 0.20),
        ("prose-a", f"{H}/raw/pride.txt", 0.10), ("prose-b", f"{H}/raw/pride.txt", 0.30),
        ("prose-c", f"{H}/raw/pride.txt", 0.50), ("prose-d", f"{H}/raw/pride.txt", 0.70),
        ("math-bayes", f"{H}/raw/wiki-Bayes_theorem.txt", 0.05), ("math-fourier", f"{H}/raw/wiki-Fourier_series.txt", 0.05),
        ("math-ftc", f"{H}/raw/wiki-Fundamental_theorem_of_calculus.txt", 0.05), ("math-eigen", f"{H}/raw/wiki-Eigenvalues_and_eigenvectors.txt", 0.05),
    ],
    "w8192": [
        ("prose-long", f"{H}/raw/pride.txt", 0.85, 52000), ("cpp-long", f"{R}/src/llama-model.cpp", 0.30, 40000),
        ("math-long", f"{H}/raw/wiki-Eigenvalues_and_eigenvectors.txt", 0.40, 48000),
    ],
}
for group, items in spec.items():
    os.makedirs(f"{H}/{group}", exist_ok=True)
    for item in items:
        name, src, frac = item[:3]
        size = item[3] if len(item) > 3 else 14000
        open(f"{H}/{group}/{name}.txt", "w").write(window(src, frac, size))
        print(group, name, os.path.basename(src), size)
