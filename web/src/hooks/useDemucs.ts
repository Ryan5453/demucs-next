import { useState, useCallback, useRef } from 'react';
import type { DemucsState, LogEntry, ModelType } from '../types';
import { SAMPLE_RATE, SEGMENT_SAMPLES } from '../types';
import { loadModel as loadOnnxModel, unloadModel as unloadOnnxModel, runInference, getSources } from '../utils/onnx-runtime';
import { processISTFT, initISTFTWorker, terminateISTFTWorker, type ISTFTResult } from '../utils/istft-worker-client';
import { processSTFT, initSTFTWorker, terminateSTFTWorker } from '../utils/stft-worker-client';
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

    const resetProcessingWorkers = useCallback(() => {
        terminateSTFTWorker();
        terminateISTFTWorker();
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
        resetProcessingWorkers();
        await unloadOnnxModel();
        setState(prev => ({ ...prev, modelLoaded: false }));
        addLog('Model unloaded', 'info');
    }, [addLog, resetProcessingWorkers]);

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
            // Start each separation with fresh workers so a previous failed run
            // cannot deliver stale responses into the new pipeline.
            resetProcessingWorkers();
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

            // Double-buffer planar audio so the next segment can be prepared
            // without mutating the buffer currently being read by inference.
            const planarBuffers = [
                new Float32Array(SEGMENT_SAMPLES * numChannels),
                new Float32Array(SEGMENT_SAMPLES * numChannels),
            ];
            console.log('[Demucs] Initializing workers...');
            initSTFTWorker();
            initISTFTWorker();
            console.log('[Demucs] Workers ready, starting segment loop...');

            let totalInferenceMs = 0;

            // Pipeline: STFT and ISTFT run in Web Workers in parallel with GPU inference
            // Main thread orchestrates: prepare segment → STFT worker → inference → ISTFT worker
            let prevIstftPromise: Promise<ISTFTResult> | null = null;

            function accumulateISTFTResult(result: ISTFTResult) {
                const { chunks, segStart, segLength } = result;
                for (let s = 0; s < sources.length; s++) {
                    const chunk = chunks[s];
                    for (let i = 0; i < segLength; i++) {
                        const globalIdx = segStart + i;
                        if (globalIdx >= numSamples) continue;
                        const outIdx = globalIdx * numChannels;
                        outputs[sources[s]][outIdx] += chunk[i * numChannels];
                        outputs[sources[s]][outIdx + 1] += chunk[i * numChannels + 1];
                    }
                }
            }

            function prepareSegmentInterleaved(segStart: number, segLength: number): Float32Array {
                const interleaved = new Float32Array(SEGMENT_SAMPLES * numChannels);
                for (let i = 0; i < segLength; i++) {
                    const srcIdx = (segStart + i) * numChannels;
                    interleaved[i * 2] = audio[srcIdx];
                    interleaved[i * 2 + 1] = audio[srcIdx + 1];
                }
                return interleaved;
            }

            function preparePlanar(
                buffer: Float32Array,
                segStart: number,
                segLength: number
            ): Float32Array {
                buffer.fill(0);
                for (let i = 0; i < segLength; i++) {
                    const srcIdx = (segStart + i) * numChannels;
                    buffer[i] = audio[srcIdx];
                    buffer[SEGMENT_SAMPLES + i] = audio[srcIdx + 1];
                }
                return buffer;
            }

            // Kick off STFT for first segment
            const seg0Start = 0;
            const seg0End = Math.min(SEGMENT_SAMPLES, numSamples);
            const seg0Length = seg0End - seg0Start;
            let pendingStft = processSTFT(prepareSegmentInterleaved(seg0Start, seg0Length));
            let pendingPlanarIndex = 0;
            let pendingPlanar = preparePlanar(planarBuffers[pendingPlanarIndex], seg0Start, seg0Length);

            for (let seg = 0; seg < numSegments; seg++) {
                const segStart = seg * STEP;
                const segEnd = Math.min(segStart + SEGMENT_SAMPLES, numSamples);
                const segLength = segEnd - segStart;

                setStatus(`Separating segment ${seg + 1} of ${numSegments}...`);
                setProgress(((seg + 1) / numSegments) * 95);

                if (seg % 5 === 0) {
                    await new Promise(resolve => requestAnimationFrame(resolve));
                }

                // Await STFT result for this segment (already computing in worker)
                const stft = await pendingStft;
                const currentPlanar = pendingPlanar;

                const specShape = [1, numChannels, stft.numBins, stft.numFrames];
                const audioShape = [1, numChannels, SEGMENT_SAMPLES];

                // Kick off inference (GPU)
                const inferenceStart = performance.now();
                const inferencePromise = runInference(
                    stft.real,
                    stft.imag,
                    currentPlanar,
                    specShape,
                    audioShape
                );

                // While inference runs on GPU, kick off STFT for next segment in worker
                if (seg + 1 < numSegments) {
                    const nextSegStart = (seg + 1) * STEP;
                    const nextSegEnd = Math.min(nextSegStart + SEGMENT_SAMPLES, numSamples);
                    const nextSegLength = nextSegEnd - nextSegStart;
                    pendingStft = processSTFT(prepareSegmentInterleaved(nextSegStart, nextSegLength));
                    pendingPlanarIndex = 1 - pendingPlanarIndex;
                    pendingPlanar = preparePlanar(
                        planarBuffers[pendingPlanarIndex],
                        nextSegStart,
                        nextSegLength
                    );
                }

                // Await the GPU result
                const results = await inferencePromise;
                const inferenceEnd = performance.now();
                totalInferenceMs += inferenceEnd - inferenceStart;

                // Collect previous ISTFT result (should already be done by now)
                if (prevIstftPromise) {
                    const istftResult = await prevIstftPromise;
                    accumulateISTFTResult(istftResult);
                }

                // Post current results to ISTFT worker (runs in parallel with next segment's inference)
                prevIstftPromise = processISTFT({
                    specReal: new Float32Array(results.outSpecReal),
                    specImag: new Float32Array(results.outSpecImag),
                    wave: new Float32Array(results.outWave),
                    numSources: sources.length,
                    numChannels,
                    numBins: stft.numBins,
                    numFrames: stft.numFrames,
                    segStart, segLength, seg,
                    numSegments, numSamples,
                    fadeIn, fadeOut,
                    overlap: OVERLAP,
                });
            }

            // Collect the final segment's ISTFT result
            if (prevIstftPromise) {
                const istftResult = await prevIstftPromise;
                accumulateISTFTResult(istftResult);
            }

            console.log(`[Demucs] === Timing Summary ===`);
            console.log(`[Demucs]   Inference: ${(totalInferenceMs / 1000).toFixed(2)}s`);
            console.log(`[Demucs]   Wall time: ${((performance.now() - startTime) / 1000).toFixed(2)}s`);

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
            resetProcessingWorkers();

            setState(prev => ({
                ...prev,
                separating: false,
            }));

        } catch (error) {
            resetProcessingWorkers();
            addLog(`Separation failed: ${(error as Error).message}`, 'error');
            setStatus('Error during separation');
            setState(prev => ({ ...prev, separating: false }));
        }
    }, [state.modelLoaded, state.audioBuffer, addLog, setStatus, setProgress, resetProcessingWorkers]);

    const resetForNewTrack = useCallback(() => {
        resetProcessingWorkers();
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
    }, [stemUrls, artworkUrl, resetProcessingWorkers]);

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
