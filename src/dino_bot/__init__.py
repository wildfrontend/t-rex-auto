"""猛龍計畫自動化框架。"""

from .config import AppConfig, load_config

__all__ = ["AppConfig", "load_config"]
# 版號的唯一來源。pyproject.toml 以 setuptools 的 dynamic version 讀這裡,
# 兩個打包腳本也從這一行取值,所以發版只需要改這個字串。
#
# 這裡曾經獨立於 pyproject 的套件版號各走各的(套件已到 0.0.9 時這裡還停在
# 0.2.25),診斷包的 bot_version 因此對不上使用者手上的發佈版本。
__version__ = "0.0.49"
