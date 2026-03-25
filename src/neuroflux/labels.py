"""
NEURO//FLUX — FreeSurfer Label Mapping  (labels.py)
====================================================
SynthSeg 2.0 outputs FreeSurfer aseg integer labels (up to ~255 structures).
This module provides:

  fs_to_tissue(fs_arr)  →  SEG_FULL  (6-class, uint8, same integers as v3.x)
  fs_to_hemi(fs_arr)    →  SEG_HEMI  (10-class, uint8, same integers as v3.x)

SEG_FULL classes:
  0  background
  1  CSF                (ventricles, subarachnoid)
  2  cortical GM
  3  WM
  4  deep GM            (subcortical nuclei + hippocampus/amygdala)
  5  brainstem
  6  cerebellum

SEG_HEMI classes:
  0  background
  1  CSF
  2  GM left
  3  GM right
  4  WM left
  5  WM right
  6  corpus callosum
  7  deep GM left
  8  deep GM right
  9  brainstem
  10 cerebellum

FreeSurfer label reference:
  https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

NOTE: Unlike v3.x (which used a midplane geometric heuristic for hemispheres),
      this module leverages native left/right FS labels — anatomically robust
      for atrophied, paediatric, and pathological brains.
"""

import numpy as np

# ---------------------------------------------------------------------------
# FreeSurfer integer label -> NEUROFLUX SEG_FULL class
# ---------------------------------------------------------------------------
_FS_TO_TISSUE: dict = {
    # Background / extracranial
    0:   0,

    # CSF -- ventricles, subarachnoid, cisterns
    4:   1,   # Left-Lateral-Ventricle
    5:   1,   # Left-Inf-Lat-Vent
    14:  1,   # 3rd-Ventricle
    15:  1,   # 4th-Ventricle
    24:  1,   # CSF
    31:  1,   # Left-choroid-plexus
    43:  1,   # Right-Lateral-Ventricle
    44:  1,   # Right-Inf-Lat-Vent
    63:  1,   # Right-choroid-plexus
    72:  1,   # 5th-Ventricle

    # Cortical grey matter
    3:   2,   # Left-Cerebral-Cortex
    42:  2,   # Right-Cerebral-Cortex

    # White matter (includes CC and WM hyperintensities)
    2:   3,   # Left-Cerebral-White-Matter
    41:  3,   # Right-Cerebral-White-Matter
    77:  3,   # WM-hypointensities
    78:  3,   # Left-WM-hypointensities
    79:  3,   # Right-WM-hypointensities
    251: 3,   # CC-Posterior
    252: 3,   # CC-Mid-Posterior
    253: 3,   # CC-Central
    254: 3,   # CC-Mid-Anterior
    255: 3,   # CC-Anterior

    # Deep GM -- subcortical nuclei + hippocampus + amygdala
    10:  4,   # Left-Thalamus-Proper
    11:  4,   # Left-Caudate
    12:  4,   # Left-Putamen
    13:  4,   # Left-Pallidum
    17:  4,   # Left-Hippocampus
    18:  4,   # Left-Amygdala
    26:  4,   # Left-Accumbens-area
    28:  4,   # Left-VentralDC
    30:  4,   # Left-vessel
    49:  4,   # Right-Thalamus-Proper
    50:  4,   # Right-Caudate
    51:  4,   # Right-Putamen
    52:  4,   # Right-Pallidum
    53:  4,   # Right-Hippocampus
    54:  4,   # Right-Amygdala
    58:  4,   # Right-Accumbens-area
    60:  4,   # Right-VentralDC
    62:  4,   # Right-vessel

    # Brainstem
    16:  5,   # Brain-Stem
    170: 5,   # Brainstem (alt label)
    173: 5,   # Medulla
    174: 5,   # Pons
    175: 5,   # SCP
    178: 5,   # Midbrain

    # Cerebellum
    6:   6,   # Left-Cerebellum-Exterior
    7:   6,   # Left-Cerebellum-White-Matter
    8:   6,   # Left-Cerebellum-Cortex
    45:  6,   # Right-Cerebellum-Exterior
    46:  6,   # Right-Cerebellum-White-Matter
    47:  6,   # Right-Cerebellum-Cortex
}

# ---------------------------------------------------------------------------
# Sub-group masks used for lateralised hemi segmentation
# ---------------------------------------------------------------------------
_CC_LABELS     = frozenset([251, 252, 253, 254, 255])
_LEFT_CORTEX   = frozenset([3])
_RIGHT_CORTEX  = frozenset([42])
_LEFT_WM       = frozenset([2, 78])
_RIGHT_WM      = frozenset([41, 79])
_LEFT_DEEP_GM  = frozenset([10, 11, 12, 13, 17, 18, 26, 28, 30])
_RIGHT_DEEP_GM = frozenset([49, 50, 51, 52, 53, 54, 58, 60, 62])

# ---------------------------------------------------------------------------
# LUT -- pre-built once at module import for O(1) vectorised remapping
# ---------------------------------------------------------------------------
_LUT_MAX = 512   # covers all standard FreeSurfer labels


def _build_lut():
    lut = np.zeros(_LUT_MAX, dtype=np.uint8)
    for lbl, cls in _FS_TO_TISSUE.items():
        if lbl < _LUT_MAX:
            lut[lbl] = cls
    return lut


_TISSUE_LUT = _build_lut()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fs_to_tissue(fs_arr):
    """
    Remap a FreeSurfer label volume (from SynthSeg) to the NEUROFLUX
    7-class tissue segmentation (uint8).

    Parameters
    ----------
    fs_arr : np.ndarray, integer dtype
        3-D array of FreeSurfer label integers.

    Returns
    -------
    np.ndarray, uint8  --  NEUROFLUX SEG_FULL classes 0-6.
    """
    clipped = np.clip(fs_arr, 0, _LUT_MAX - 1).astype(np.int32)
    return _TISSUE_LUT[clipped]


def fs_to_hemi(fs_arr, tissue_arr):
    """
    Build the NEUROFLUX 10-class hemispheric segmentation.

    Uses native FreeSurfer left/right label sidedness instead of the
    geometric midplane heuristic used in v3.x -- more robust for
    atrophied, paediatric, and pathological brains.

    Parameters
    ----------
    fs_arr     : np.ndarray, integer dtype  -- raw SynthSeg FreeSurfer labels.
    tissue_arr : np.ndarray, uint8          -- output of fs_to_tissue().

    Returns
    -------
    np.ndarray, uint8  --  NEUROFLUX SEG_HEMI classes 0-10.
    """
    hemi = np.zeros_like(tissue_arr, dtype=np.uint8)

    # 1  CSF -- bilateral
    hemi[tissue_arr == 1] = 1

    # 2/3  Cortical GM -- lateralised by native FS labels
    hemi[np.isin(fs_arr, list(_LEFT_CORTEX))]  = 2
    hemi[np.isin(fs_arr, list(_RIGHT_CORTEX))] = 3

    # 4/5  WM -- lateralised, corpus callosum split out
    cc_mask = np.isin(fs_arr, list(_CC_LABELS))
    hemi[np.isin(fs_arr, list(_LEFT_WM))  & ~cc_mask] = 4
    hemi[np.isin(fs_arr, list(_RIGHT_WM)) & ~cc_mask] = 5

    # 6  Corpus callosum
    hemi[cc_mask] = 6

    # 7/8  Deep GM -- lateralised
    hemi[np.isin(fs_arr, list(_LEFT_DEEP_GM))]  = 7
    hemi[np.isin(fs_arr, list(_RIGHT_DEEP_GM))] = 8

    # 9  Brainstem -- bilateral
    hemi[tissue_arr == 5] = 9

    # 10  Cerebellum -- bilateral
    hemi[tissue_arr == 6] = 10

    return hemi


# Human-readable class names
# ---------------------------------------------------------------------------
TISSUE_NAMES = {
    1: "CSF",
    2: "GM",
    3: "WM",
    4: "deep-GM",
    5: "brainstem",
    6: "cerebellum",
}

HEMI_NAMES = {
    1:  "CSF",
    2:  "GM-L",
    3:  "GM-R",
    4:  "WM-L",
    5:  "WM-R",
    6:  "CC",
    7:  "dGM-L",
    8:  "dGM-R",
    9:  "BS",
    10: "CB",
}
