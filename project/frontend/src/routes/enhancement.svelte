<script>


    import axios from "axios";

    let audioFile = null;
    let cleanReferenceFile = null;
    let waveformPlot = "";
    let melspecPlot = "";
    let enhancedWaveformPlot = "";
    let waveformPlotWiener = "";
    let melspecPlotWiener = "";
    let enhancedWienerAudioUrl = "";
    let enhancedMelspecPlot = "";
    let originalAudioUrl = "";
    let enhancedAudioUrl = "";
    let noisySample = "";
    let origSample = "";
    let metrics = null;
    let loading = false;

    const backend = "http://localhost:8000";

    async function uploadAudio() {
        if (!audioFile) return;
        loading = true;
        try {
            const formData = new FormData();
            formData.append("file", audioFile);

            const resp = await axios.post(`${backend}/upload`, formData, {
                headers: {"Content-Type": "multipart/form-data"},
            });
            waveformPlot = `${backend}/plot/${resp.data.waveform_plot.split('/').pop()}`;
            melspecPlot = `${backend}/plot/${resp.data.melspec_plot.split('/').pop()}`;
            originalAudioUrl = `${backend}/audio/${resp.data.audio_path.split('/').pop()}`;
        } catch (err) {
            alert("Upload failed: " + err);
        }
        loading = false;
    }


    async function enhanceAudio() {
        if (!audioFile) return;
        loading = true;
        try {
            const formData = new FormData();
            formData.append("file", audioFile);
            if (cleanReferenceFile) {
                formData.append("clean_ref", cleanReferenceFile);
            }

            const resp = await axios.post(`${backend}/enhance`, formData, {
                headers: {"Content-Type": "multipart/form-data"},
            });
            enhancedWaveformPlot = `${backend}/plot/${resp.data.waveform_plot.split('/').pop()}`;
            enhancedMelspecPlot = `${backend}/plot/${resp.data.melspec_plot.split('/').pop()}`;
            waveformPlotWiener = `${backend}/plot/${resp.data.waveform_plot_wiener.split('/').pop()}`;
            melspecPlotWiener = `${backend}/plot/${resp.data.melspec_plot_wiener.split('/').pop()}`;
            enhancedWienerAudioUrl = `${backend}/audio/${resp.data.enhanced_wiener_audio.split('/').pop()}`;
            enhancedAudioUrl = `${backend}/audio/${resp.data.enhanced_audio.split('/').pop()}`;
            metrics = resp.data.metrics ?? null;
        } catch (err) {
            alert("Enhancement failed: " + err);
        }
        loading = false;
    }

    async function testSample() {
        noisySample = `${backend}/test-sample/noised.wav`
        origSample = `${backend}/test-sample/orig.wav`
    }

</script>

<main>
    <h1>Speech Enhancement App</h1>
    <div class="description">
        <p>
            This is a project thesis prototype app for audio speech signal enhancement.!!!
            You can upload an audio file corrupted with noise on the left side, visualize it, and enhance it.
            The visualization and enhancement processes will display the waveforms and Mel-spectrograms of the
            corresponding audios.
            If you don’t have a corrupted signal, you can either corrupt one in the sandbox by clicking <b>'Add
            noise'</b> in
            the navigation menu above, or use the <b>'Test sample'</b> button below to download samples.

            <br>Created by Elbek Bakiev.
            <br><span>All files will be automatically removed from the server 10 minutes after creation</span>
        </p>
    </div>


    <p></p>
    <div class="audio-upload">
        <div style="margin: 10px">
            <label for="">Corrupted file</label>
            <input type="file" accept="audio/*" on:change="{e => audioFile = e.target.files[0]}"/>
        </div>
        <div style="margin: 10px">
            <label for="">Clear file for reference(Optional)</label>
            <input type="file" accept="audio/*" on:change="{e => cleanReferenceFile = e.target.files[0]}">
        </div>
    </div>

    <br/>
    <button on:click={uploadAudio} disabled={loading}>Visualize</button>
    <button on:click={enhanceAudio} disabled={loading}>Enhance</button>
    <button on:click={testSample} disabled={loading}>Test Sample</button>
    {#if loading}
        <p>Loading...</p>
    {/if}
    {#if noisySample}
        <div>
            <h2>Noised Audio</h2>
            <audio controls src={noisySample}></audio>
            <h2>Original Audio</h2>
            <audio controls src={origSample}></audio>
        </div>
    {/if}
    <div class="main-container">
        <!-- Original Audio -->
        <div class="audio-column original-audio">
            {#if waveformPlot}
                <h2>Original Audio</h2>
                <audio controls src={originalAudioUrl}></audio>
                <img src={waveformPlot} alt="Waveform"/>
                <img src={melspecPlot} alt="Melspectrogram"/>
            {:else}
                <div class="placeholder">Upload corrupted file</div>
            {/if}
        </div>

        <!-- Enhanced Audio -->
        <div class="audio-column enhanced-audio">
            {#if enhancedAudioUrl}
                <h2>Enhanced Audio</h2>
                <audio controls src={enhancedAudioUrl}></audio>
                <img src={enhancedWaveformPlot} alt="Enhanced Waveform"/>
                <img src={enhancedMelspecPlot} alt="Enhanced Melspectrogram"/>
            {:else}
                <div class="placeholder">Appears after enhancement</div>
            {/if}
        </div>
        <!-- Enhanced Audio by Wiener filter -->
        <div class="audio-column wiener-audio">
            {#if enhancedWienerAudioUrl}
                <h2>Enhanced Audio by Wiener filter</h2>
                <audio controls src={enhancedWienerAudioUrl}></audio>
                <img src={waveformPlotWiener} alt="Enhanced Waveform by Wiener filter"/>
                <img src={melspecPlotWiener} alt="Enhanced Melspectrogram by Wiener filter"/>
            {:else}
                <div class="placeholder">Appears after enhancement</div>
            {/if}
        </div>
        {#if metrics}
            <div class="metrics">
                <h3>Metrics</h3>
                <table>
                    <thead>
                    <tr>
                        <th></th>
                        <th>Noisy</th>
                        <th>Enhanced</th>
                        <th>Wiener</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                        <td>PESQ</td>
                        <td>{metrics.pesq_noisy ? metrics.pesq_noisy.toFixed(2) : '-'}</td>
                        <td>{metrics.pesq_enhanced ? metrics.pesq_enhanced.toFixed(2) : '-'}</td>
                        <td>{metrics.pesq_wiener ? metrics.pesq_wiener.toFixed(2) : '-'}</td>
                    </tr>
                    <tr>
                        <td>STOI</td>
                        <td>{metrics.stoi_noisy ? metrics.stoi_noisy.toFixed(2) : '-'}</td>
                        <td>{metrics.stoi_enhanced ? metrics.stoi_enhanced.toFixed(2) : '-'}</td>
                        <td>{metrics.stoi_wiener ? metrics.stoi_wiener.toFixed(2) : '-'}</td>
                    </tr>
                    <tr>
                        <td>SNR (dB)</td>
                        <td>{metrics.snr_noisy ? metrics.snr_noisy.toFixed(2) : '-'}</td>
                        <td>{metrics.snr_enhanced ? metrics.snr_enhanced.toFixed(2) : '-'}</td>
                        <td>{metrics.snr_wiener ? metrics.snr_wiener.toFixed(2) : '-'}</td>
                    </tr>
                    </tbody>
                </table>
                {#if metrics.error}
                    <p class="error">{metrics.error}</p>
                {/if}
            </div>
        {/if}
    </div>
</main>


<style>
    img {
        max-width: 100%;
        margin: 20px;
        border-radius: 16px;
    }

    button {
        margin-right: 1em;
    }

    .main-container {
        display: flex;
        gap: 32px;
        justify-content: center;
        align-items: center;
        flex-direction: column;
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

    .metrics {
        margin-top: 24px;
        width: 100%;
    }

    .metrics table {
        width: 100%;
        border-collapse: collapse;
    }

    .metrics th, .metrics td {
        border: 1px solid #ccc;
        padding: 6px 12px;
        text-align: center;
    }

    .metrics th {
        background: #f0f0f0;
        color: black;
    }

    .metrics .error {
        color: #c00;
        margin-top: 12px;
    }

    .audio-column h2 {
        color: #aaa;
        font-size: 1.2em;
        text-align: center;
    }

    .description {
        width: 100%;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .description p {
        width: 50%;
    }

    .description span {
        color: red;
    }

    .placeholder {
        color: #aaa;
        font-size: 1.2em;
        text-align: center;
    }

    @media (max-width: 900px) {
        .main-container {
            flex-direction: column;
            gap: 12px;
        }

        .audio-column {
            width: 100%;
        }
    }

    .audio-upload {
        display: flex;
        flex-direction: row;
        justify-content: center;
    }

    .audio-upload div {
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: left;
    }
</style>

