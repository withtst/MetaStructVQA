"""
Conversion utility between Anno ID (sub_xxx) and Image ID (PTXH...).
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MAPPING_PATH = os.path.join(BASE_DIR, '..', 'anno_mapping_table.csv')

# Cache the mapping table to avoid re-reading the file on every call
_cache = None


def _load_mapping():
    global _cache
    if _cache is None:
        _cache = pd.read_csv(_MAPPING_PATH, encoding='utf-8')
    return _cache


def anno_converter(anno_id: str) -> str:
    """
    Look up the Image ID (e.g. PTXH220103024) for a given anno_id (e.g. sub_001).
    """
    mapping_df = _load_mapping()
    idx = mapping_df.index[mapping_df['anno_id'] == anno_id]
    return mapping_df.loc[idx, 'Image ID'].values[0]


if __name__ == "__main__":
    print(anno_converter('sub_001'))
