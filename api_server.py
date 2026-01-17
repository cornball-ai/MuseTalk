"""
MuseTalk API Server

REST API for lip-sync video generation.

Endpoints:
    POST /lipsync - Generate lip-synced video from image/video + audio
    GET /health - Health check

Usage:
    python api_server.py --port 7861
"""

import os
import sys
import uuid
import torch
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from musetalk.utils.utils import load_all_model
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

app = FastAPI(
    title="MuseTalk API",
    description="Real-time high quality lip synchronization",
    version="1.5"
)

# Global model references (loaded on startup)
models = {}

def load_models(device: torch.device, use_float16: bool = True, version: str = "v15"):
    """Load all models into memory."""
    global models

    if version == "v15":
        unet_config = "./models/musetalkV15/musetalk.json"
        unet_model_path = "./models/musetalkV15/unet.pth"
    else:
        unet_config = "./models/musetalk/musetalk.json"
        unet_model_path = "./models/musetalk/pytorch_model.bin"

    whisper_dir = "./models/whisper"

    print(f"Loading MuseTalk {version} models...")

    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type="sd-vae",
        unet_config=unet_config,
        device=device
    )

    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    fp = FaceParsing()

    models = {
        "vae": vae,
        "unet": unet,
        "pe": pe,
        "audio_processor": audio_processor,
        "whisper": whisper,
        "fp": fp,
        "device": device,
        "weight_dtype": weight_dtype,
        "version": version
    }

    print("Models loaded successfully")


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    version = os.environ.get("MUSETALK_VERSION", "v15")
    use_float16 = os.environ.get("MUSETALK_FP16", "1") == "1"
    load_models(device, use_float16=use_float16, version=version)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": models.get("version", "unknown"),
        "device": str(models.get("device", "unknown")),
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/lipsync")
async def lipsync(
    video: UploadFile = File(..., description="Source video or image"),
    audio: UploadFile = File(..., description="Audio file to sync"),
    bbox_shift: int = Form(0, description="Bounding box shift (-10 to 10)"),
    batch_size: int = Form(8, description="Batch size for inference"),
    fps: int = Form(25, description="Output video FPS (for image input)")
):
    """
    Generate lip-synced video.

    - **video**: Source video (.mp4, .avi) or image (.jpg, .png)
    - **audio**: Audio file (.mp3, .wav) to sync lips to
    - **bbox_shift**: Adjust mouth openness (-10 to 10, 0 = default)
    - **batch_size**: Processing batch size (default: 8)
    - **fps**: Output FPS when input is an image (default: 25)

    Returns the generated video file.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Create temp directory for this request
    request_id = str(uuid.uuid4())[:8]
    temp_dir = Path(tempfile.gettempdir()) / f"musetalk_{request_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded files
        video_ext = Path(video.filename).suffix or ".mp4"
        audio_ext = Path(audio.filename).suffix or ".mp3"

        video_path = temp_dir / f"input{video_ext}"
        audio_path = temp_dir / f"audio{audio_ext}"
        output_path = temp_dir / "output.mp4"

        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        # Run inference
        from scripts.inference import main as run_inference
        from argparse import Namespace

        version = models["version"]

        args = Namespace(
            ffmpeg_path="",
            gpu_id=0,
            vae_type="sd-vae",
            unet_config=f"./models/musetalk{'V15' if version == 'v15' else ''}/musetalk.json",
            unet_model_path=f"./models/musetalk{'V15' if version == 'v15' else ''}/{'unet.pth' if version == 'v15' else 'pytorch_model.bin'}",
            whisper_dir="./models/whisper",
            inference_config=None,
            bbox_shift=bbox_shift,
            result_dir=str(temp_dir),
            extra_margin=10,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
            batch_size=batch_size,
            output_vid_name="output.mp4",
            use_saved_coord=False,
            saved_coord=False,
            use_float16=True,
            parsing_mode="jaw",
            left_cheek_width=90,
            right_cheek_width=90,
            version=version
        )

        # Create a simple config for the inference
        import yaml
        config_path = temp_dir / "config.yaml"
        config = {
            "task1": {
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "result_name": "output.mp4"
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        args.inference_config = str(config_path)

        # Run inference
        run_inference(args)

        # Find output file
        result_path = temp_dir / version / "output.mp4"
        if not result_path.exists():
            # Try alternative path
            for f in temp_dir.rglob("*.mp4"):
                if "output" in f.name or "concat" not in f.name:
                    result_path = f
                    break

        if not result_path.exists():
            raise HTTPException(status_code=500, detail="Output video not generated")

        # Return the video file
        return FileResponse(
            path=str(result_path),
            media_type="video/mp4",
            filename=f"lipsync_{request_id}.mp4",
            background=None  # Don't delete temp files until response is sent
        )

    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lipsync/async")
async def lipsync_async(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    bbox_shift: int = Form(0),
    batch_size: int = Form(8),
    fps: int = Form(25),
    output_dir: str = Form("/app/output", description="Directory to save output")
):
    """
    Generate lip-synced video asynchronously.

    Returns immediately with the output path. Video will be saved to output_dir.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    request_id = str(uuid.uuid4())[:8]
    output_filename = f"lipsync_{request_id}.mp4"
    output_path = Path(output_dir) / output_filename

    # Save files and queue processing
    temp_dir = Path(tempfile.gettempdir()) / f"musetalk_{request_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    video_ext = Path(video.filename).suffix or ".mp4"
    audio_ext = Path(audio.filename).suffix or ".mp3"

    video_path = temp_dir / f"input{video_ext}"
    audio_path = temp_dir / f"audio{audio_ext}"

    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    with open(audio_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    # TODO: Queue for background processing
    # For now, return the paths for manual processing

    return JSONResponse({
        "request_id": request_id,
        "status": "queued",
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "output_path": str(output_path)
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    args = parser.parse_args()

    os.environ["MUSETALK_VERSION"] = args.version

    uvicorn.run(app, host=args.host, port=args.port)
