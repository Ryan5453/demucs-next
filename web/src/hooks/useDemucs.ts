import { useState, useCallback, useRef } from 'react';
import type { DemucsState, LogEntry, ModelType } from '../types';
import { SAMPLE_RATE, SEGMENT_SAMPLES, NFFT, HOP_LENGTH } from '../types';
import { loadModel as loadOnnxModel, unloadModel as unloadOnnxModel, runInference, getSources } from '../utils/onnx-runtime';
import { computeSTFT, computeISTFT, createSTFTBuffers, createISTFTBuffers } from '../utils/audio-processor';
import { createWavBlob } from '../utils/wav-utils';
import { decodeAudioFile } from '../utils/audio-decoder';

const initialState: DemucsState = {
    modelLoaded: false,
    modelLoading: false,
    audioLoaded: false,
    audioBuffer: null,
    audioFile: null,
    separating: false,
    progress: 0,
    status: 'Ready',
    logs: [],
};

export function useDemucs() {
    const [state, setState] = useState<DemucsState>(initialState);
    const [audioError, setAudioError] = useState<string | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);

    // Store pre-created blob URLs
    const [stemUrls, setStemUrls] = useState<Record<string, string>>({});
    // Store artwork URL (album art from audio file)
    const [artworkUrl, setArtworkUrl] = useState<string | null>(null);
    // Store track metadata from audio file
    const [trackTitle, setTrackTitle] = useState<string | null>(null);
    const [trackArtist, setTrackArtist] = useState<string | null>(null);
    // Store waveform data for visualization (array of 0-100 values)
    const [stemWaveforms, setStemWaveforms] = useState<Record<string, number[]>>({});

    const addLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
        setState(prev => ({
            ...prev,
            logs: [...prev.logs, { timestamp: new Date(), message, type }]
        }));
    }, []);

    const setStatus = useCallback((status: string) => {
        setState(prev => ({ ...prev, status }));
    }, []);

    const setProgress = useCallback((progress: number) => {
        setState(prev => ({ ...prev, progress }));
    }, []);

    const getAudioContext = useCallback(() => {
        if (!audioContextRef.current) {
            audioContextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
        }
        return audioContextRef.current;
    }, []);

    const loadModel = useCallback(async (model: ModelType, backend: 'webgpu' | 'wasm' = 'webgpu') => {
        setState(prev => ({ ...prev, modelLoading: true }));
        const result = await loadOnnxModel(model, addLog, backend);
        setState(prev => ({
            ...prev,
            modelLoading: false,
            modelLoaded: result.success,
        }));
        return result.success;
    }, [addLog]);

    const unloadModel = useCallback(async () => {
        await unloadOnnxModel();
        setState(prev => ({ ...prev, modelLoaded: false }));
        addLog('Model unloaded', 'info');
    }, [addLog]);

    const clearAudioError = useCallback(() => {
        setAudioError(null);
    }, []);

    const loadAudio = useCallback(async (file: File) => {
        try {
            setAudioError(null);
            addLog(`Loading audio: ${file.name}`, 'info');
            const ctx = getAudioContext();

            const { buffer: audioBuffer, artwork, title, artist, usedFallback } = await decodeAudioFile(file, ctx);

            if (usedFallback === 'ffmpeg') {
                addLog('Audio decoded using fallback decoder (ffmpeg.wasm)', 'info');
            } else {
                addLog('Audio decoded with Mediabunny', 'info');
            }

            // Store artwork if present
            if (artwork) {
                setArtworkUrl(artwork);
                addLog('Album artwork extracted', 'info');
            }

            // Store track metadata if present
            if (title) {
                setTrackTitle(title);
                addLog(`Track title: ${title}`, 'info');
            }
            if (artist) {
                setTrackArtist(artist);
                addLog(`Artist: ${artist}`, 'info');
            }

            addLog('Audio loaded successfully.', 'success');

            setState(prev => ({
                ...prev,
                audioLoaded: true,
                audioBuffer,
                audioFile: file,
            }));
        } catch (error) {
            const errorMessage = (error as Error).message;
            addLog(`Failed to load audio: ${errorMessage}`, 'error');
            setAudioError(errorMessage);
        }
    }, [addLog, getAudioContext]);

    const separateAudio = useCallback(async (skipModelCheck = false) => {
        // Check if model is loaded (either main thread session or worker)
        if (!skipModelCheck && !state.modelLoaded) {
            addLog('Model not loaded', 'error');
            return;
        }
        if (!state.audioBuffer) {
            addLog('Audio not loaded', 'error');
            return;
        }

        try {
            setState(prev => ({ ...prev, separating: true }));
            setStemUrls({}); // Clear old URLs
            setStemWaveforms({}); // Clear old waveforms
            setStatus('Preparing audio...');
            setProgress(0);

            // Yield to allow React to render the separating UI before heavy processing
            await new Promise(resolve => setTimeout(resolve, 0));

            const startTime = performance.now();
            addLog('Starting separation...', 'info');

            const numChannels = 2;
            const numSamples = state.audioBuffer.length;
            const audio = new Float32Array(numSamples * numChannels);

            const left = state.audioBuffer.getChannelData(0);
            const right = state.audioBuffer.numberOfChannels > 1
                ? state.audioBuffer.getChannelData(1)
                : left;

            for (let i = 0; i < numSamples; i++) {
                audio[i * 2] = left[i];
                audio[i * 2 + 1] = right[i];
            }



            const OVERLAP = Math.floor(SEGMENT_SAMPLES * 0.5);
            const STEP = SEGMENT_SAMPLES - OVERLAP;
            const numSegments = Math.ceil((numSamples - OVERLAP) / STEP);

            // Get sources from the loaded model
            const sources = getSources();

            const outputs: Record<string, Float32Array> = {};
            for (const source of sources) {
                outputs[source] = new Float32Array(numSamples * numChannels);
            }

            const fadeIn = new Float32Array(OVERLAP);
            const fadeOut = new Float32Array(OVERLAP);
            for (let i = 0; i < OVERLAP; i++) {
                fadeIn[i] = i / OVERLAP;
                fadeOut[i] = 1 - i / OVERLAP;
            }

            console.log('[Demucs] Created fade buffers');

            // Pre-allocate reusable buffers to reduce GC pressure
            const segmentPlanar = new Float32Array(SEGMENT_SAMPLES * numChannels);
            const segmentInterleaved = new Float32Array(SEGMENT_SAMPLES * numChannels);
            const specBufferSize = numChannels * (NFFT / 2) * Math.ceil(SEGMENT_SAMPLES / HOP_LENGTH);
            const sourceReal = new Float32Array(specBufferSize);
            const sourceImag = new Float32Array(specBufferSize);
            console.log('[Demucs] Creating STFT buffers...');
            const stftBuffers = createSTFTBuffers();
            console.log('[Demucs] Creating ISTFT buffers...');
            const istftBuffers = createISTFTBuffers();
            console.log('[Demucs] Buffers created, starting segment loop...');

            for (let seg = 0; seg < numSegments; seg++) {
                const segStart = seg * STEP;

                const segEnd = Math.min(segStart + SEGMENT_SAMPLES, numSamples);
                const segLength = segEnd - segStart;

                console.log(`[Demucs] Processing segment ${seg + 1}/${numSegments}`);
                setStatus(`Separating segment ${seg + 1} of ${numSegments}...`);
                setProgress(((seg + 1) / numSegments) * 95);

                // Yield occasionally to allow React to render progress updates
                if (seg % 5 === 0) {
                    await new Promise(resolve => requestAnimationFrame(resolve));
                }


                segmentPlanar.fill(0);
                for (let i = 0; i < segLength; i++) {
                    const srcIdx = (segStart + i) * numChannels;
                    segmentPlanar[i] = audio[srcIdx];
                    segmentPlanar[SEGMENT_SAMPLES + i] = audio[srcIdx + 1];
                }

                segmentInterleaved.fill(0);
                for (let i = 0; i < SEGMENT_SAMPLES; i++) {
                    segmentInterleaved[i * 2] = segmentPlanar[i];
                    segmentInterleaved[i * 2 + 1] = segmentPlanar[SEGMENT_SAMPLES + i];
                }

                console.log('[Demucs] Computing STFT...');
                const stft = computeSTFT(segmentInterleaved, stftBuffers);
                console.log('[Demucs] STFT complete');

                const specShape = [1, numChannels, stft.numBins, stft.numFrames];
                const audioShape = [1, numChannels, SEGMENT_SAMPLES];

                console.log('[Demucs] Running model inference...');
                const results = await runInference(
                    stft.real,
                    stft.imag,
                    segmentPlanar,
                    specShape,
                    audioShape
                );
                console.log('[Demucs] Inference complete');

                // Results are always Float32Arrays (runInference handles tensor extraction)
                const specRealData = results.outSpecReal;
                const specImagData = results.outSpecImag;
                const waveData = results.outWave;

                for (let s = 0; s < sources.length; s++) {
                    const specOffset = s * numChannels * stft.numBins * stft.numFrames;

                    sourceReal.fill(0);
                    sourceImag.fill(0);

                    for (let c = 0; c < numChannels; c++) {
                        const cOffset = c * stft.numBins * stft.numFrames;
                        for (let b = 0; b < stft.numBins; b++) {
                            for (let t = 0; t < stft.numFrames; t++) {
                                const idx = b * stft.numFrames + t;
                                const specIdx = specOffset + cOffset + idx;
                                sourceReal[cOffset + idx] = specRealData[specIdx];
                                sourceImag[cOffset + idx] = specImagData[specIdx];
                            }
                        }
                    }

                    // iSTFT computation
                    const freqAudio = computeISTFT(sourceReal, sourceImag, numChannels, stft.numBins, stft.numFrames, SEGMENT_SAMPLES, istftBuffers);

                    const sourceWaveOffset = s * numChannels * SEGMENT_SAMPLES;

                    for (let i = 0; i < segLength; i++) {
                        const globalIdx = segStart + i;
                        if (globalIdx >= numSamples) continue;

                        const outIdx = globalIdx * numChannels;

                        const leftFreq = freqAudio[i];
                        const rightFreq = freqAudio[SEGMENT_SAMPLES + i];
                        const leftTime = waveData[sourceWaveOffset + i];
                        const rightTime = waveData[sourceWaveOffset + SEGMENT_SAMPLES + i];

                        const leftVal = leftFreq + leftTime;
                        const rightVal = rightFreq + rightTime;

                        let weight = 1.0;
                        if (seg > 0 && i < OVERLAP) {
                            weight = fadeIn[i];
                        }
                        if (seg < numSegments - 1 && i >= SEGMENT_SAMPLES - OVERLAP) {
                            const fadeIdx = i - (SEGMENT_SAMPLES - OVERLAP);
                            weight = fadeOut[fadeIdx];
                        }

                        outputs[sources[s]][outIdx] += leftVal * weight;
                        outputs[sources[s]][outIdx + 1] += rightVal * weight;
                    }
                }
                // Note: tensor disposal is now handled inside runInference()
            }


            // Create blob URLs IMMEDIATELY after separation (like original code)
            setStatus('Finalizing...');
            setProgress(98);
            // addLog('Creating audio files...', 'info');

            const urls: Record<string, string> = {};
            const waveforms: Record<string, number[]> = {};
            const numBars = 60; // Number of waveform bars to display

            for (const source of sources) {
                const blob = createWavBlob(outputs[source], numChannels, SAMPLE_RATE);
                urls[source] = URL.createObjectURL(blob);

                // Compute waveform for visualization
                const audioData = outputs[source];
                const samplesPerBar = Math.floor(audioData.length / numBars);
                const bars: number[] = [];

                for (let i = 0; i < numBars; i++) {
                    const start = i * samplesPerBar;
                    const end = Math.min(start + samplesPerBar, audioData.length);

                    // Calculate RMS (root mean square) for this segment
                    let sumSquares = 0;
                    for (let j = start; j < end; j++) {
                        sumSquares += audioData[j] * audioData[j];
                    }
                    const rms = Math.sqrt(sumSquares / (end - start));

                    // Convert to percentage (0-100), with some scaling for visual appeal
                    // Audio RMS is typically 0-0.3 for normal audio, scale to 0-100
                    const barHeight = Math.min(100, Math.max(15, rms * 300));
                    bars.push(barHeight);
                }

                waveforms[source] = bars;
            }

            setStemUrls(urls);
            setStemWaveforms(waveforms);

            const duration = ((performance.now() - startTime) / 1000).toFixed(2);
            setStatus('Complete!');
            setProgress(100);
            addLog(`Finished separation in ${duration}s.`, 'success');

            setState(prev => ({
                ...prev,
                separating: false,
            }));

        } catch (error) {
            addLog(`Separation failed: ${(error as Error).message}`, 'error');
            setStatus('Error during separation');
            setState(prev => ({ ...prev, separating: false }));
        }
    }, [state.modelLoaded, state.audioBuffer, addLog, setStatus, setProgress]);

    const resetForNewTrack = useCallback(() => {
        // Revoke old blob URLs to prevent memory leaks
        Object.values(stemUrls).forEach(url => URL.revokeObjectURL(url));
        if (artworkUrl) {
            URL.revokeObjectURL(artworkUrl);
        }

        setState(prev => ({
            ...prev,
            audioLoaded: false,
            audioBuffer: null,
            audioFile: null,
            separating: false,
            progress: 0,
            status: 'Ready',
        }));
        setStemUrls({});
        setStemWaveforms({});
        setArtworkUrl(null);
        setTrackTitle(null);
        setTrackArtist(null);
        setAudioError(null);
    }, [stemUrls, artworkUrl]);

    return {
        ...state,
        stemUrls,
        stemWaveforms,
        artworkUrl,
        trackTitle,
        trackArtist,
        audioError,
        loadModel,
        unloadModel,
        loadAudio,
        clearAudioError,
        separateAudio,
        resetForNewTrack,
    };
}
