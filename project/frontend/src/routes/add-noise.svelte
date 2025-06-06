<script>
    import axios from "axios";

    let audioFile = null;
    let numDropouts = 3;
    let minDropoutLengthMs = 50
    let maxDropoutLengthMs = 300
    let noisedAudioUrl = "";
    let noisedWaveformPlot = "";
    let noisedMelspecPlot = "";
    let loading = false;
    let waveformPlot = "";
    let melspecPlot = "";
    let originalAudioUrl = "";
    let addNoiseForWholeAudio = false;
    let snr = 10;
    const backend = "https://project-thesis-frontend.onrender.com";


    async function uploadAudio() {
        if (!audioFile) return;
        loading = true;
        const formData = new FormData();
        formData.append("file", audioFile);

        const resp = await axios.post(`${backend}/upload`, formData, {
            headers: {"Content-Type": "multipart/form-data"},
        });
        waveformPlot = `${backend}/plot/${resp.data.waveform_plot.split('/').pop()}`;
        melspecPlot = `${backend}/plot/${resp.data.melspec_plot.split('/').pop()}`;
        originalAudioUrl = `${backend}/audio/${resp.data.audio_path.split('/').pop()}`;
        loading = false;
    }


    async function addNoise() {
        if (!audioFile) return;
        loading = true;
        const formData = new FormData();
        formData.append("file", audioFile);
        formData.append("num_dropouts", numDropouts);
        formData.append("min_dropout_length_ms", minDropoutLengthMs);
        formData.append("max_dropout_length_ms", maxDropoutLengthMs);
        formData.append("entire_track_noise", addNoiseForWholeAudio);
        formData.append("snr", snr);
        const resp = await axios.post(`${backend}/add_noise`, formData, {
            headers: {"Content-Type": "multipart/form-data"},
        });
        noisedWaveformPlot = `${backend}/plot/${resp.data.waveform_plot.split('/').pop()}`;
        noisedMelspecPlot = `${backend}/plot/${resp.data.melspec_plot.split('/').pop()}`;
        noisedAudioUrl = `${backend}/audio/${resp.data.audio_path.split('/').pop()}`;
        loading = false;
    }
</script>

<main>
    <h1>Add Noise to Audio</h1>
    <input type="file" accept="audio/*" on:change="{e => audioFile = e.target.files[0]}"/>
    <button on:click={uploadAudio} disabled={loading}>Visualize</button>
    <div class="main-container">

        <div class="audio-column">
            {#if waveformPlot}
                <h2>Original Audio</h2>
                <audio controls src={originalAudioUrl}></audio>
                <img src={waveformPlot} alt="Waveform"/>
                <img src={melspecPlot} alt="Melspectrogram"/>
            {:else}
                <div class="placeholder">Upload corrupted file</div>
            {/if}
        </div>
    </div>

    <div>
        <label for="">Number of dropouts:</label>
        <input type="number" bind:value={numDropouts} disabled={addNoiseForWholeAudio} min="1"/>
    </div>
    <div>
        <label for="">Min dropout length (ms):</label>
        <input type="number" bind:value={minDropoutLengthMs} disabled={addNoiseForWholeAudio} min="10" max="50" required/>
    </div>
    <div>
        <label for="">Max dropout length (ms):</label>
        <input type="number" bind:value={maxDropoutLengthMs} disabled={addNoiseForWholeAudio} min="10" max="300" required/>
    </div>
    <div>
        <label for="">Add noise for entire track</label>
        <input type="checkbox" bind:checked={addNoiseForWholeAudio}>
        <label for="">SNR</label>
        <input type="number" bind:value={snr} disabled={!addNoiseForWholeAudio}>
    </div>
    <button on:click={addNoise} disabled={loading}>Add Noise</button>
    {#if loading}
        <p>Loading...</p>
    {/if}
    <div class="main-container">

        <div class="audio-column">
            {#if noisedAudioUrl}
                <h2>Noised Audio</h2>
                <audio controls src={noisedAudioUrl}></audio>
                <img src={noisedWaveformPlot} alt="Noised Waveform"/>
                <img src={noisedMelspecPlot} alt="Noised Melspectrogram"/>
            {/if}
        </div>
    </div>
</main>


<style>
    img {
        max-width: 100%;
        margin: 20px;
        border-radius: 16px;
    }


    .main-container {
        display: flex;
        gap: 32px;
        justify-content: center;
        align-items: flex-start;
        margin-top: 32px;
    }

    .audio-column {
        min-width: 320px;
        min-height: 320px;
        max-width: 50%;
        background: rgba(122, 122, 130, 0.06);
        border-radius: 16px;
        padding: 0 25px;
        box-shadow: 0 2px 8px #0001;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
</style>