"""pytest 根配置：把仓库根目录放进 sys.path。

这样无论用 ``pytest`` 还是 ``python -m pytest`` 运行，
``import memory`` / ``import tools`` 都能解析到仓库内的包，
不依赖运行目录，也不需要安装成包。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
