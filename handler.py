import os
import torch
import torchaudio
import runpod
import base64
import traceback
from io import BytesIO
import requests
import urllib.parse
import soundfile as sf
from acestep.pipeline_ace_step import ACEStepPipeline
import uuid

INIT_ERROR_FILE = "/tmp/init_error.log"
model = None

try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)
    print("Loading ACE-Step model...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints")
    model = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=True,
        overlapped_decode=True,
    )
    if not model.loaded:
        model.load_checkpoint()
    print("Model loaded.")
except Exception as e:
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize model: {tb_str}")
    model = None

def upload_to_gcs(signed_url, audio_bytes, content_type="audio/wav"):
    try:
        response = requests.put(
            signed_url,
            data=audio_bytes,
            headers={"Content-Type": content_type},
            timeout=300
        )
        response.raise_for_status()
        print(f"Uploaded to GCS: {signed_url[:100]}...")
        return True
    except Exception as e:
        print(f"GCS upload failed: {e}")
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
        print(f"Calling webhook: {webhook_url}")
        response = requests.post(webhook_url, timeout=30)
        response.raise_for_status()
        print(f"Backend notified: {status}")
        return True
    except Exception as e:
        print(f"Webhook notification failed: {e}")
        return False

def save_wav_bytes(tensor, sample_rate=48000):
    buffer = BytesIO()
    audio = tensor.float().cpu()
    if audio.dim() == 1:
        data = audio.numpy()
    else:
        data = audio.transpose(0, 1).numpy()
    sf.write(buffer, data, sample_rate, format="wav")
    return buffer.getvalue()

def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            error_msg = f.read()
        return {"error": f"Model failed to load: {error_msg}"}

    job_input = event.get("input", {})
    prompt = job_input.get("prompt")
    audio_duration = float(job_input.get("audio_duration", 120))
    lyrics = job_input.get("lyrics", "[inst]")
    callback_url = job_input.get("callback_url")
    upload_urls = job_input.get("upload_urls", {})

    if not prompt:
        error_msg = "No prompt provided."
        if callback_url:
            notify_backend(callback_url, "failed", error_msg)
        return {"error": error_msg}

    # Hardcoded ACE-Step config
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
        print(f"🎵 Generating audio: prompt='{prompt}', duration={audio_duration}s")
        output_path = f"/tmp/generate_{uuid.uuid4().hex}.wav"
        model(
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
        print(f"Audio generated ({len(wav_bytes)} bytes)")

        # Optional GCS upload
        if upload_urls:
            wav_url = upload_urls.get("wav_url")
            if wav_url:
                upload_success = upload_to_gcs(wav_url, wav_bytes)
                if not upload_success:
                    raise Exception("Failed to upload WAV to GCS")

        if callback_url:
            notify_backend(callback_url, "completed")

        return {
            "audio_base64": audio_base64,
            "format": "wav",
            "status": "completed"
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Error: {error_msg}")
        if callback_url:
            notify_backend(callback_url, "failed", str(e))
        return {"error": error_msg, "status": "failed"}

runpod.serverless.start({"handler": handler})
