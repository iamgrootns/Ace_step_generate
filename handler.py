import os
import torch
import runpod
import base64
import traceback
import requests
import urllib.parse
import uuid
import soundfile as sf
import time
from huggingface_hub import snapshot_download

# --- Global Variables & Model Loading ---
INIT_ERROR_FILE = "/tmp/init_error.log"
model_demo = None

try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)
    
    print("🔥 Python version:", os.popen("python --version").read().strip())
    print("🔥 PyTorch version:", torch.__version__)
    print("🔥 CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("🔥 CUDA version:", torch.version.cuda)
    
    print("Loading ACEStep model...")
    from acestep.pipeline_ace_step import ACEStepPipeline
    
    # Use Runpod network volume or download to persistent storage
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/runpod-volume/checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Download checkpoints if not present
    checkpoint_file = os.path.join(checkpoint_path, "ace_step_v1_3.5b.safetensors")
    if not os.path.exists(checkpoint_file):
        print(f"📥 Downloading ACEStep checkpoints to {checkpoint_path}...")
        try:
            snapshot_download(
                repo_id="ACE-Step/ACE-Step-v1-3.5B",
                local_dir=checkpoint_path,
                local_dir_use_symlinks=False,
            )
            print("✅ Checkpoints downloaded successfully")
        except Exception as download_error:
            print(f"❌ Failed to download checkpoints: {download_error}")
            raise
    else:
        print(f"✅ Using cached checkpoints from {checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize ACEStep with Python 3.12 compatible settings
    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=True,
        overlapped_decode=True,
    )
    
    if not model_demo.loaded:
        model_demo.load_checkpoint()
    
    print("✅ ACEStep model loaded successfully")

except Exception as e:
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize ACEStep model: {tb_str}")
    print(f"❌ Initialization error: {tb_str}")
    model_demo = None


def upload_to_gcs(signed_url, audio_bytes, content_type="audio/wav"):
    try:
        response = requests.put(
            signed_url,
            data=audio_bytes,
            headers={"Content-Type": content_type},
            timeout=300
        )
        response.raise_for_status()
        print(f"✅ Uploaded to GCS")
        return True
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")
        return False


def notify_backend(callback_url, status, error_message=None):
    try:
        parsed = urllib.parse.urlparse(callback_url)
        params = urllib.parse.parse_qs(parsed.query)
        params['status'] = [status]
        if error_message:
            params['error_message'] = [error_message]
        
        new_query = urllib.parse.urlencode(params, doseq=True)
        webhook_url = urllib.parse.urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, new_query, parsed.fragment
        ))
        
        response = requests.post(webhook_url, timeout=30)
        response.raise_for_status()
        print(f"✅ Backend notified: {status}")
        return True
    except Exception as e:
        print(f"❌ Webhook failed: {e}")
        return False


def patch_save_method(model):
    original_save = model.save_wav_file

    def patched_save(target_wav, idx, save_path=None, sample_rate=48000, format="wav"):
        if save_path is None:
            base_path = "/tmp/outputs"
            os.makedirs(base_path, exist_ok=True)
            output_path_wav = f"{base_path}/output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
        else:
            if os.path.isdir(save_path):
                output_path_wav = os.path.join(save_path, f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}")
            else:
                output_path_wav = save_path
            output_dir = os.path.dirname(os.path.abspath(output_path_wav))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        target_wav = target_wav.float().cpu()
        if target_wav.dim() == 1:
            audio_data = target_wav.numpy()
        else:
            audio_data = target_wav.transpose(0, 1).numpy()

        sf.write(output_path_wav, audio_data, sample_rate)
        return output_path_wav

    return original_save, patched_save


def handler(event):
    """
    Input:
    {
        "input": {
            "audio_duration": 30.0,
            "prompt": "upbeat electronic music",
            "lyrics": "[inst]",
            "callback_url": "https://...",
            "upload_urls": {"wav_url": "https://..."}
        }
    }
    """
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            error_msg = f"Worker init failed: {f.read()}"
        return {"error": error_msg, "status": "failed"}

    job_input = event.get("input", {})
    audio_duration = job_input.get("audio_duration")
    prompt = job_input.get("prompt")
    
    if not audio_duration or not prompt:
        error_msg = "Missing: audio_duration and prompt"
        if job_input.get("callback_url"):
            notify_backend(job_input["callback_url"], "failed", error_msg)
        return {"error": error_msg, "status": "failed"}
    
    callback_url = job_input.get("callback_url")
    upload_urls = job_input.get("upload_urls", {})
    
    try:
        # Extract parameters
        lyrics = job_input.get("lyrics", "[inst]")
        infer_step = job_input.get("infer_step", 60)
        guidance_scale = job_input.get("guidance_scale", 14.0)
        scheduler_type = job_input.get("scheduler_type", "euler")
        cfg_type = job_input.get("cfg_type", "apg")
        omega_scale = job_input.get("omega_scale", 10.0)
        actual_seeds = job_input.get("manual_seeds", [42, 99])
        guidance_interval = job_input.get("guidance_interval", 0.5)
        guidance_interval_decay = job_input.get("guidance_interval_decay", 0.0)
        min_guidance_scale = job_input.get("min_guidance_scale", 3.0)
        use_erg_tag = job_input.get("use_erg_tag", True)
        use_erg_lyric = job_input.get("use_erg_lyric", True)
        use_erg_diffusion = job_input.get("use_erg_diffusion", True)
        guidance_scale_text = job_input.get("guidance_scale_text", 3.0)
        guidance_scale_lyric = job_input.get("guidance_scale_lyric", 0.0)
        
        print(f"🎵 Generating: prompt='{prompt}', duration={audio_duration}s")
        
        output_path = f"/tmp/output_{uuid.uuid4().hex}.wav"
        original_save, patched_save = patch_save_method(model_demo)
        model_demo.save_wav_file = patched_save
        
        start_time = time.time()
        
        try:
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
        finally:
            model_demo.save_wav_file = original_save
        
        generation_time = time.time() - start_time
        print(f"✅ Generated in {generation_time:.2f}s")
        
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        
        if upload_urls and upload_urls.get("wav_url"):
            upload_success = upload_to_gcs(upload_urls["wav_url"], audio_bytes)
            if not upload_success:
                raise Exception("GCS upload failed")
        
        if callback_url:
            notify_backend(callback_url, "completed")
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        if os.path.exists(output_path):
            os.remove(output_path)
        
        return {
            "audio_base64": audio_base64,
            "sample_rate": 48000,
            "format": "wav",
            "duration": audio_duration,
            "generation_time": generation_time,
            "status": "completed"
        }
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"❌ Error: {error_msg}")
        
        if callback_url:
            notify_backend(callback_url, "failed", str(e))
        
        return {"error": error_msg, "status": "failed"}


runpod.serverless.start({"handler": handler})
