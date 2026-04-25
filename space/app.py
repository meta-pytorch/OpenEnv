import sys, os

# Make all local modules importable
for p in [
    os.path.dirname(__file__),
    os.path.join(os.path.dirname(__file__), 'src'),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ui import build_ui

demo = build_ui()

if __name__ == '__main__':
    demo.launch()
