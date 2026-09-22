"""测试共享的路径引导。"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (_ROOT, _TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
