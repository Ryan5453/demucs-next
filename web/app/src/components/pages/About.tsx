export function About() {
    return (
        <div className="content-page">
            <h1 className="content-title">About</h1>

            <div className="content-body">
                <p>
                    <strong>un/blend</strong> is a free, open-source audio stem separation tool powered by
                    Meta AI's Demucs model. Everything runs entirely in your browser, so your audio files
                    never leave your device.
                </p>

                <p>
                    Demucs separates a mixed track into individual stems such as drums, bass, vocals,
                    and other instruments. The model is converted to ONNX format and runs in-browser via
                    onnxruntime-web. When a model is loaded, the model weights (~91MB) and a matching
                    runtime binary are downloaded: ~26MB if your browser supports WebGPU, or ~13MB for
                    the WebAssembly-only fallback.
                </p>

                <p>
                    Audio files are decoded with <a href="https://mediabunny.dev/">MediaBunny</a>, which
                    uses your browser's native decoders where possible. For formats that can't be decoded
                    natively, the app falls back to <a href="https://ffmpegwasm.netlify.app/">ffmpeg.wasm</a>,
                    an additional ~33MB download.
                </p>

                <p>
                    Because the model itself runs locally on your machine's CPU/GPU rather than a server, it's a heavy tab:
                    you should expect high memory and
                    power use for the duration of a separation. Browsers with stricter tab memory limits,
                    Safari in particular, may reload or kill the tab on longer tracks. 
                    If that happens, a Chromium-based browser will likely perform better.
                </p>

                <p className="content-notice">
                    <strong>Notice:</strong> The pretrained Demucs weights carry no license grant from Meta
                    (they were trained on MUSDB18-HQ plus an internal proprietary dataset), so use of the
                    model's outputs is subject to that lack of a license.
                </p>
            </div>
        </div>
    );
}
