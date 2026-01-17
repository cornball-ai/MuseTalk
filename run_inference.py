#!/usr/bin/env python
"""
MuseTalk inference wrapper for PyTorch 2.7+

This script patches torch.load BEFORE any other imports to work with
legacy pickled checkpoints (like DWPose) that contain numpy arrays.
"""
import torch

# Patch torch.load IMMEDIATELY before any imports
_orig_load = torch.load

def patched_load(*args, **kwargs):
    """Wrapper that defaults to weights_only=False for legacy checkpoints."""
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)

torch.load = patched_load

# Now we can safely import the rest
import sys
import os

# Add app dir to path
app_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(app_dir)
sys.path.insert(0, app_dir)

# Run the original inference script as __main__
if __name__ == "__main__":
    # Remove our script name from argv so argparse works correctly
    sys.argv[0] = "scripts/inference.py"

    # Execute the inference script
    exec(open("scripts/inference.py").read())
