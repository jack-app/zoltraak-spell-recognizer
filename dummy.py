import mmap
import os

import numpy as np


def initialize_memory_mapped_files() -> None:
    """
    呪文検出用のメモリマップトファイルを初期化します。
    存在しない場合は /tmp/is_spell_detected と /tmp/spell_id を作成します。
    """
    # is_spell_detectedの初期化（1ビット）
    if not os.path.exists("/tmp/is_spell_detected"):
        with open("/tmp/is_spell_detected", "wb") as f:
            f.write(b"\x00")

    # spell_idの初期化（int16）
    if not os.path.exists("/tmp/spell_id"):
        with open("/tmp/spell_id", "wb") as f:
            f.write(np.int16(0).tobytes())


def update_memory_mapped_files(spell_id: int) -> None:
    """
    メモリマップトファイルを呪文検出結果で更新します。

    Args:
        spell_id (int): 検出された呪文ID（呪文が検出されない場合は0）
    """
    # is_spell_detectedの更新
    with open("/tmp/is_spell_detected", "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm[0] = 1 if spell_id > 0 else 0
        mm.close()

    # spell_idの更新
    with open("/tmp/spell_id", "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm.write(np.int16(spell_id).tobytes())
        mm.close()


if __name__ == "__main__":
    initialize_memory_mapped_files()
    while True:
        spell_id = int(input("enter spell id: "))
        update_memory_mapped_files(spell_id)
