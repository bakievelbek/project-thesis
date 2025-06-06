from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torchaudio
import torch
import numpy as np
import os
from fastapi import Form
import uuid
import matplotlib.pyplot as plt
import librosa.display
from speechbrain.inference.enhancement import SpectralMaskEnhancement
from pesq import pesq
from pystoi import stoi

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, restrict in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SpeechBrain model
enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank"
)

def save_waveform_plot(waveform, sr, out_path):
    plt.figure(figsize=(10, 2.5), dpi=200)
    plt.plot(np.linspace(0, len(waveform)/sr, len(waveform)), waveform, color='#2a3949', linewidth=1.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_melspec_plot(waveform, sr, out_path):
    plt.figure(figsize=(10, 3), dpi=200)
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(
        S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000,
        cmap='inferno'
    )
    plt.colorbar(format='%+2.0f dB', label='dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def calc_metrics(clean, noisy, enhanced, sr):
    metrics = {}
    try:
        metrics["pesq_noisy"] = pesq(sr, clean, noisy, 'wb')
        metrics["pesq_enhanced"] = pesq(sr, clean, enhanced, 'wb')
        metrics["stoi_noisy"] = stoi(clean, noisy, sr, extended=False)
        metrics["stoi_enhanced"] = stoi(clean, enhanced, sr, extended=False)
        metrics["snr_noisy"] = 10 * np.log10(np.mean(clean ** 2) / np.mean((clean - noisy) ** 2))
        metrics["snr_enhanced"] = 10 * np.log10(np.mean(clean ** 2) / np.mean((clean - enhanced) ** 2))
    except Exception as e:
        metrics["error"] = str(e)
    return metrics



def to_mono(waveform):
    # waveform: either torch.Tensor or np.ndarray, shape [channels, n] or [n]
    if isinstance(waveform, torch.Tensor):
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            return waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 2:
            return waveform
        elif waveform.ndim == 1:
            return waveform.unsqueeze(0)
        else:
            raise ValueError("Unexpected tensor shape")
    else:
        # numpy
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            return waveform.mean(axis=0, keepdims=True)
        elif waveform.ndim == 2:
            return waveform
        elif waveform.ndim == 1:
            return waveform[np.newaxis, :]
        else:
            raise ValueError("Unexpected array shape")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    in_path = f"uploads/{uid}_{file.filename}"
    with open(in_path, "wb") as f:
        f.write(await file.read())
    waveform, sr = torchaudio.load(in_path)
    waveform = to_mono(waveform)
    waveform_np = waveform.squeeze().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.squeeze(waveform)
    wave_plot = f"outputs/{uid}_waveform.png"
    melspec_plot = f"outputs/{uid}_melspec.png"
    save_waveform_plot(waveform_np, sr, wave_plot)
    save_melspec_plot(waveform_np, sr, melspec_plot)
    return {
        "waveform_plot": wave_plot,
        "melspec_plot": melspec_plot,
        "audio_path": in_path
    }

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import torchaudio
import torch
import numpy as np
import os
import uuid
from speechbrain.inference.enhancement import SpectralMaskEnhancement

# Your helper functions:
# - save_waveform_plot
# - save_melspec_plot
# - calc_metrics
# - to_mono

@app.post("/enhance")
async def enhance_audio(
    file: UploadFile = File(...),
    clean_ref: UploadFile = File(None)
):
    uid = str(uuid.uuid4())
    in_path = f"uploads/{uid}_{file.filename}"
    with open(in_path, "wb") as f:
        f.write(await file.read())
    waveform, sr = torchaudio.load(in_path)
    waveform = to_mono(waveform)
    if sr != 16000:
        import torchaudio.transforms as T
        waveform = T.Resample(sr, 16000)(waveform)
        sr = 16000

    # Enhance
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank"
    )
    enhanced = enhancer.enhance_batch(waveform, lengths=torch.tensor([1.]))
    out_path = f"outputs/{uid}_enhanced.wav"
    torchaudio.save(out_path, enhanced.cpu(), sr)

    # Generate plots
    enhanced_np = enhanced.squeeze().cpu().numpy()
    wave_plot = f"outputs/{uid}_enhanced_waveform.png"
    melspec_plot = f"outputs/{uid}_enhanced_melspec.png"
    save_waveform_plot(enhanced_np, sr, wave_plot)
    save_melspec_plot(enhanced_np, sr, melspec_plot)

    # Metrics calculation
    metrics = {}
    if clean_ref is not None:
        ref_path = f"uploads/{uid}_ref_{clean_ref.filename}"
        with open(ref_path, "wb") as f:
            f.write(await clean_ref.read())
        clean_waveform, ref_sr = torchaudio.load(ref_path)
        clean_waveform = to_mono(clean_waveform)
        # Resample clean reference if needed
        if ref_sr != sr:
            import torchaudio.transforms as T
            clean_waveform = T.Resample(ref_sr, sr)(clean_waveform)
        # Prepare numpy arrays for metrics
        clean_np = clean_waveform.squeeze().cpu().numpy()
        noisy_np = waveform.squeeze().cpu().numpy()
        metrics = calc_metrics(clean_np, noisy_np, enhanced_np, sr)

    return {
        "enhanced_audio": out_path,
        "waveform_plot": wave_plot,
        "melspec_plot": melspec_plot,
        "metrics": metrics
    }


@app.get("/audio/{filename}")
def get_audio(filename: str):
    if os.path.exists(f"outputs/{filename}"):
        return FileResponse(f"outputs/{filename}")
    return FileResponse(f"uploads/{filename}")

@app.get("/plot/{filename}")
def get_plot(filename: str):
    return FileResponse(f"outputs/{filename}")


@app.post("/add_noise")
async def add_noise(
    file: UploadFile = File(None),
    num_dropouts: int = Form(3),
    min_dropout_length_ms: int = Form(50),
    max_dropout_length_ms: int = Form(300),
    entire_track_noise: bool = Form(False),
    snr: int = Form(10)
):
    print(num_dropouts)
    uid = str(uuid.uuid4())
    in_path = f"uploads/{uid}_{file.filename}"
    with open(in_path, "wb") as f:
        f.write(await file.read())
    waveform, sr = torchaudio.load(in_path)
    waveform = waveform.squeeze().numpy()
    print(entire_track_noise)
    # Add noise dropouts
    total_len = len(waveform)
    if entire_track_noise:
        rms = np.sqrt(np.mean(waveform ** 2))
        noise_std = rms / (10 ** (snr / 20))
        noise = np.random.normal(0, noise_std, waveform.shape)
        waveform += noise
    else:
        for n in range(num_dropouts):
            dropout_ms = np.random.randint(min_dropout_length_ms, max_dropout_length_ms + 1)
            dropout_samples = int(sr * dropout_ms / 1000)
            start = np.random.randint(0, total_len - dropout_samples)
            waveform[start:start + dropout_samples] += np.random.normal(0, 0.1, dropout_samples)
    waveform = np.clip(waveform, -1, 1)

    out_path = f"outputs/{uid}_noised.wav"
    torchaudio.save(out_path, torch.tensor(waveform).unsqueeze(0), sr)

    # Plots
    wave_plot = f"outputs/{uid}_noised_waveform.png"
    melspec_plot = f"outputs/{uid}_noised_melspec.png"
    save_waveform_plot(waveform, sr, wave_plot)
    save_melspec_plot(waveform, sr, melspec_plot)
    return {
        "audio_path": out_path,
        "waveform_plot": wave_plot,
        "melspec_plot": melspec_plot
    }