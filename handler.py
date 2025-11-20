import os
import uuid
import base64
import torch
import torchaudio
import soundfile as sf
import runpod
import traceback
from acestep.pipeline_ace_step import ACEStepPipeline
from io import BytesIO

# --- Model Loading at Startup ---
INIT_ERROR_FILE = "/tmp/init_error.log"
model_demo = None

try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)
    print("Loading ACE-Step pipeline...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints")
    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=True,
        overlapped_decode=True,
    )
    if not model_demo.loaded:
        model_demo.load_checkpoint()
    print("✅ ACE-Step model loaded successfully.")
except Exception as e:
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize model: {tb_str}")
    model_demo = None

# --- Helper: Save WAV from tensor
def save_wav_bytes(tensor, sample_rate=48000):
    buffer = BytesIO()
    audio = tensor.float().cpu()
    if audio.dim() == 1:
        data = audio.numpy()
    else:
        data = audio.transpose(0, 1).numpy()
    sf.write(buffer, data, sample_rate, format="wav")
    return buffer.getvalue()

# --- RunPod Handler ---
def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            error_msg = f.read()
        return {"error": f"Model failed to load: {error_msg}"}

    # Parse input
    job_input = event.get("input", {})
    audio_duration = float(job_input.get("audio_duration", 120))
    prompt = job_input.get("prompt", "rock guitar solo")
    lyrics = job_input.get("lyrics", "[inst]")

    # Hardcoded parameters for ACE-Step generation
    infer_step = 60
    guidance_scale = 14.0
    scheduler_type = "euler"
    cfg_type = "apg"
    omega_scale = 10.0
    actual_seeds = [42, 99]
    guidance_interval = 0.5
    guidance_interval_decay = 0.0
    min_guidance_scale = 3.0
    use_erg_tag = True
    use_erg_lyric = True
    use_erg_diffusion = True
    oss_steps = []
    guidance_scale_text = 3.0
    guidance_scale_lyric = 0.0

    try:
        print(f"🎵 Starting generation: {prompt} for {audio_duration}s")
        output_path = f"/tmp/generate_{uuid.uuid4().hex}.wav"
        model_demo(
            format="wav",
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=actual_seeds,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=None,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            save_path=output_path,
        )
        with open(output_path, "rb") as f:
            wav_bytes = f.read()
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        print(f"✅ Audio generated ({len(wav_bytes)} bytes)")
        return {
            "audio_base64": audio_base64,
            "format": "wav",
            "status": "completed"
        }
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"❌ Generation error: {error_msg}")
        return {"error": error_msg, "status": "failed"}

runpod.serverless.start({"handler": handler})
