from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import os
from acestep.pipeline_ace_step import ACEStepPipeline
import uuid
import time
import soundfile as sf
    
model_demo = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_demo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints")
    print(f"📥 Loading models from {checkpoint_path}...")
    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=True,
        overlapped_decode=True,
    )
    if not model_demo.loaded:
        model_demo.load_checkpoint()
        print("✅ Models loaded successfully")
    yield

app = FastAPI(title="ACEStep Music Generation API", lifespan=lifespan)

class ACEStepGenerateInput(BaseModel):
    audio_duration: float
    prompt: str
    lyrics: Optional[str] = "[inst]"

class ACEStepOutput(BaseModel):
    status: str
    output_path: str
    message: str

def patch_save_method(model):
    original_save = model.save_wav_file
    def patched_save(target_wav, idx, save_path=None, sample_rate=48000, format="wav"):
        import time
        if save_path is None:
            base_path = "./outputs"
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

@app.post("/generate", response_model=ACEStepOutput)
async def generate_audio_simple(input_data: ACEStepGenerateInput):
    global model_demo
    if model_demo is None or not model_demo.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_{uuid.uuid4().hex}.wav")

    original_save, patched_save = patch_save_method(model_demo)
    model_demo.save_wav_file = patched_save

    # Hardcoded parameters
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

    start_time = time.time()
    try:
        model_demo(
            format="wav",
            audio_duration=input_data.audio_duration,
            prompt=input_data.prompt,
            lyrics=input_data.lyrics if input_data.lyrics else "[inst]",
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
    total_time = time.time() - start_time
    print(f"[INFO] Audio generation took {total_time:.2f} seconds.")

    return ACEStepOutput(
        status="success",
        output_path=output_path,
        message=f"Generated {input_data.audio_duration}s audio successfully in {total_time:.2f} seconds"
    )


@app.get("/health")
async def health_check():
    global model_demo
    return {"status": "healthy", "model_loaded": model_demo is not None and model_demo.loaded}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
