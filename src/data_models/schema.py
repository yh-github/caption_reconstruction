from __future__ import annotations

import hashlib

# noinspection PyUnresolvedReferences, PyProtectedMember
HashType = hashlib._hashlib.HASH

from data_models.captions_only import CaptionedClip
from data_models.complex_struct import VideoSegment

type_from_str = {
    "list[VideoSegment]": list[VideoSegment],
    "list[CaptionedClip]": list[CaptionedClip]
}
