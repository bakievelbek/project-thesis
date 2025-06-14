import os
import uuid
import torch
import asyncio
import torchaudio

import numpy as np
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt

from pesq import pesq
from pystoi import stoi

from fastapi import Form
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks

from scipy.signal import wiener
from speechbrain.inference.enhancement import SpectralMaskEnhancement

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


async def remove_file_after_delay(paths: list, delay: float = 600.0):
    print(f'Background task created. Files {paths} will be deleted in {delay} seconds')
    await asyncio.sleep(delay)
    print(f'Removing files: {paths}.')
    for path in paths:
        try:
            if path:
                os.remove(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")


def save_waveform_plot(waveform, sr, out_path):
    plt.figure(figsize=(10, 2.5), dpi=200)
    plt.plot(np.linspace(0, len(waveform) / sr, len(waveform)), waveform, color='#2a3949', linewidth=1.2)
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


def normalize(waveform):
    max_val = np.max(np.abs(waveform))
    if max_val == 0:
        return waveform
    return waveform / max_val


def calc_metrics(clean, noisy, enhanced, wiener, sr):
    metrics = {}
    try:
        # PESQ calculation
        metrics["pesq_noisy"] = float(pesq(sr, clean, noisy, 'wb'))
        metrics["pesq_enhanced"] = float(pesq(sr, clean, enhanced, 'wb'))
        metrics["pesq_wiener"] = float(pesq(sr, clean, wiener, 'wb'))

        # STOI calculation
        metrics["stoi_noisy"] = float(stoi(clean, noisy, sr, extended=False))
        metrics["stoi_enhanced"] = float(stoi(clean, enhanced, sr, extended=False))
        metrics["stoi_wiener"] = float(stoi(clean, wiener, sr, extended=False))

        # SNR calculation
        if np.var(clean) == 0:  # Check if clean signal has no variance
            metrics["snr_noisy"] = None
            metrics["snr_enhanced"] = None
            metrics["snr_wiener"] = None
        else:
            metrics["snr_noisy"] = float(10 * np.log10(np.mean(clean ** 2) / np.mean((clean - noisy) ** 2)))
            metrics["snr_enhanced"] = float(10 * np.log10(np.mean(clean ** 2) / np.mean((clean - enhanced) ** 2)))
            metrics["snr_wiener"] = float(10 * np.log10(np.mean(clean ** 2) / np.mean((clean - wiener) ** 2)))
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
async def upload_audio(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None
):
    uid = str(uuid.uuid4())
    in_path = f"uploads/{uid}_{file.filename}"
    with open(in_path, "wb") as f:
        f.write(await file.read())
    waveform, sr = torchaudio.load(in_path)
    waveform = to_mono(waveform)
    waveform_np = normalize(waveform.squeeze().cpu().numpy())
    wave_plot = f"outputs/{uid}_waveform.png"
    melspec_plot = f"outputs/{uid}_melspec.png"
    save_waveform_plot(waveform_np, sr, wave_plot)
    save_melspec_plot(waveform_np, sr, melspec_plot)
    print([in_path, wave_plot, melspec_plot])
    background_tasks.add_task(remove_file_after_delay, [in_path, wave_plot, melspec_plot])
    return {
        "waveform_plot": wave_plot,
        "melspec_plot": melspec_plot,
        "audio_path": in_path
    }


@app.post("/enhance")
async def enhance_audio(
        file: UploadFile = File(...),
        clean_ref: UploadFile = File(None),
        background_tasks: BackgroundTasks = None
):
    # Read File
    uid = str(uuid.uuid4())
    in_path = f"uploads/{uid}_{file.filename}"
    with open(in_path, "wb") as f:
        f.write(await file.read())
    waveform, sr = torchaudio.load(in_path)
    waveform = to_mono(waveform)

    # Enhance using model
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank"
    )
    enhanced = enhancer.enhance_batch(waveform, lengths=torch.tensor([1.]))
    out_path = f"outputs/{uid}_enhanced.wav"
    torchaudio.save(out_path, enhanced.cpu(), sr)

    # Generate plots
    enhanced = to_mono(enhanced)
    enhanced_np = normalize(enhanced.squeeze().cpu().numpy())
    wave_plot = f"outputs/{uid}_enhanced_waveform.png"
    melspec_plot = f"outputs/{uid}_enhanced_melspec.png"
    save_waveform_plot(enhanced_np, sr, wave_plot)
    save_melspec_plot(enhanced_np, sr, melspec_plot)

    # Enhance using Wiener filter
    wiener_signal = wiener(waveform, mysize=29)
    wiener_signal = np.asarray(wiener_signal).astype(np.float32)
    if wiener_signal.ndim == 2 and wiener_signal.shape[0] == 1:
        wiener_signal = wiener_signal.squeeze()  # make mono
    if np.max(np.abs(wiener_signal)) > 1.0:
        wiener_signal = wiener_signal / np.max(np.abs(wiener_signal))  # normalize
    wiener_out_path = f"outputs/{uid}_wiener_signal.wav"
    wiener_signal = normalize(wiener_signal)
    sf.write(wiener_out_path, wiener_signal, sr)

    # Generate plots
    wave_wiener_plot = f"outputs/{uid}_wiener_waveform.png"
    melspec_wiener_plot = f"outputs/{uid}_wiener_melspec.png"
    save_waveform_plot(wiener_signal, sr, wave_wiener_plot)
    save_melspec_plot(wiener_signal, sr, melspec_wiener_plot)

    # Metrics calculation
    metrics = {}
    ref_path = ''
    if clean_ref is not None:
        ref_path = f"uploads/{uid}_ref_{clean_ref.filename}"
        with open(ref_path, "wb") as f:
            f.write(await clean_ref.read())
        clean_waveform, ref_sr = torchaudio.load(ref_path)
        clean_waveform = to_mono(clean_waveform)
        # Prepare numpy arrays for metrics
        clean_np = normalize(clean_waveform.squeeze().cpu().numpy())
        noisy_np = normalize(waveform.squeeze().cpu().numpy())
        metrics = calc_metrics(clean_np, noisy_np, enhanced_np, wiener_signal, sr)

    background_tasks.add_task(remove_file_after_delay,
                              [in_path, ref_path, out_path, wiener_out_path, wave_plot, melspec_plot, wave_wiener_plot,
                               melspec_wiener_plot])
    return {
        "enhanced_audio": out_path,
        "enhanced_wiener_audio": wiener_out_path,
        "waveform_plot": wave_plot,
        "melspec_plot": melspec_plot,
        "waveform_plot_wiener": wave_wiener_plot,
        "melspec_plot_wiener": melspec_wiener_plot,
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


@app.get("/test-sample/{filename}")
def get_test_sample(filename: str):
    return FileResponse(f"testing_sample/{filename}")


@app.post("/add_noise")
async def add_noise(
        file: UploadFile = File(None),
        num_dropouts: int = Form(3),
        min_dropout_length_ms: int = Form(50),
        max_dropout_length_ms: int = Form(300),
        entire_track_noise: bool = Form(False),
        snr: int = Form(10),
        background_tasks: BackgroundTasks = None
):
    uid = str(uuid.uuid4())
    in_path = f"uploads/{uid}_{file.filename}"
    with open(in_path, "wb") as f:
        f.write(await file.read())
    waveform, sr = torchaudio.load(in_path)
    waveform = waveform.squeeze().numpy()
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
    waveform = normalize(waveform)
    torchaudio.save(out_path, torch.tensor(waveform).unsqueeze(0), sr)

    # Plots
    wave_plot = f"outputs/{uid}_noised_waveform.png"
    melspec_plot = f"outputs/{uid}_noised_melspec.png"
    save_waveform_plot(waveform, sr, wave_plot)
    save_melspec_plot(waveform, sr, melspec_plot)
    background_tasks.add_task(remove_file_after_delay, [out_path, wave_plot, melspec_plot])

    return {
        "audio_path": out_path,
        "waveform_plot": wave_plot,
        "melspec_plot": melspec_plot
    }
