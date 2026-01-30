export function About() {
    return (
        <div className="w-full max-w-3xl mx-auto px-6 py-12 flex-1">
            {/* Content */}
            <div className="content-card">
                <h1 className="content-title">About</h1>

                <div className="content-body">
                    <p>
                        <strong>demucs.app</strong> is a free, open-source audio stem separation tool powered by Meta AI's Demucs model.
                        Everything runs entirely in your browser, so your audio files never leave your device.
                    </p>

                    <h2>The Technology</h2>

                    <p>
                        Demucs is a machine learning model that separates mixed audio into individual stems.
                        The standard 4-source model separates audio into drums, bass, vocals, and other instruments.
                        There's also an experimental 6-source model that adds guitar and piano stems, though piano separation is less reliable.
                    </p>

                    <p>
                        This app uses Demucs models converted to ONNX format for in-browser inference.
                        When you select a model, either the WebGPU runtime (~24MB) or WebAssembly runtime (~12MB) is downloaded based on your device's capabilities, along with the model weights.
                    </p>

                    <p>
                        Audio files are decoded using <a href="https://mediabunny.dev/">MediaBunny</a>, which leverages your browser's native capabilities.
                        For files that can't be decoded natively, the app falls back to <a href="https://ffmpegwasm.netlify.app/">ffmpeg.wasm</a> (~32MB).
                    </p>
                </div>
            </div>
        </div>
    );
}
