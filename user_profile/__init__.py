"""ユーザープロファイルビルダー (docs/PROFILE.md §2, §5)。

実行時 (feedback.jsonl 経由) と学習時 (仮想ユーザーストリーム) の両方で、
同一の :class:`~user_profile.builder.ProfileBuilder` を更新作用素として使う。
"""

from __future__ import annotations
