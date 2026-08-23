"""pytest 测试配置：把 plugins 目录加入 sys.path，保证插件包可导入。"""

import sys
from pathlib import Path

PLUGINS_DIR = Path(__file__).resolve().parent.parent.parent
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))
