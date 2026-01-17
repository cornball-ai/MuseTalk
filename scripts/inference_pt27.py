#!/usr/bin/env python
"""
MuseTalk inference wrapper for PyTorch 2.7+

This script patches torch.load to work with legacy pickled checkpoints
(like DWPose) that contain numpy arrays.
"""
import torch

# Patch torch.load before any imports that might load checkpoints
_orig_load = torch.load

def patched_load(*args, **kwargs):
    """Wrapper that defaults to weights_only=False for legacy checkpoints."""
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)

torch.load = patched_load

# Now run the original inference script
if __name__ == "__main__":
    import sys
    import os
    import argparse

    # Change to the app directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(script_dir)
    os.chdir(app_dir)
    sys.path.insert(0, app_dir)
    sys.path.insert(0, script_dir)

    # Import the main inference function
    from scripts.inference import main

    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--result_dir", default='./results')
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_vid_name", type=str, default=None)
    parser.add_argument("--use_saved_coord", action="store_true")
    parser.add_argument("--saved_coord", action="store_true")
    parser.add_argument("--use_float16", action="store_true")
    parser.add_argument("--parsing_mode", default='jaw')
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])

    args = parser.parse_args()

    # Adjust paths for v15
    if args.version == "v15":
        args.unet_config = "./models/musetalkV15/musetalk.json"
        args.unet_model_path = "./models/musetalkV15/unet.pth"
    else:
        args.unet_config = "./models/musetalk/musetalk.json"
        args.unet_model_path = "./models/musetalk/pytorch_model.bin"

    main(args)
