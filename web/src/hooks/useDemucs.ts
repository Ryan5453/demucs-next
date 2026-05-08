import { useState, useCallback, useRef } from 'react';
import type { DemucsState, LogEntry, ModelType } from '../types';
import { SAMPLE_RATE } from '../types';
import { loadModel as loadOnnxModel, unloadModel as unloadOnnxModel } from '../utils/onnx-runtime';
import { terminateISTFTWorker } from '../utils/istft-worker-client';
import { terminateSTFTWorker } from '../utils/stft-worker-client';
import { createWavBlob } from '../utils/wav-utils';
import { decodeAudioFile } from '../utils/audio-decoder';
import { separateAudioBuffer } from '../utils/separator';

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
        if (!skipModelCheck && !state.modelLoaded) {
            addLog('Model not loaded', 'error');
            return;
        }
        if (!state.audioBuffer) {
            addLog('Audio not loaded', 'error');
            return;
        }

        try {
            // Reset workers so a previous failed run can't leak stale state.
            resetProcessingWorkers();
            setState(prev => ({ ...prev, separating: true }));
            setStemUrls({});
            setStemWaveforms({});
            setStatus('Preparing audio...');
            setProgress(0);

            // Yield once so React paints the "separating" UI before the
            // pipeline starts hammering the main thread.
            await new Promise(resolve => setTimeout(resolve, 0));
            addLog('Starting separation...', 'info');

            const audioBuffer = state.audioBuffer;
            const result = await separateAudioBuffer(audioBuffer, {
                onProgress: ({ segIdx, totalSegs, fraction }) => {
                    setStatus(`Separating segment ${segIdx} of ${totalSegs}...`);
                    setProgress(fraction * 95);
                },
            });

            // Build blob URLs and waveform RMS bars for the player UI.
            setStatus('Finalizing...');
            setProgress(98);

            const urls: Record<string, string> = {};
            const waveforms: Record<string, number[]> = {};
            const numBars = 60;

            for (const [source, samples] of Object.entries(result.stems)) {
                const blob = createWavBlob(samples, 2, SAMPLE_RATE);
                urls[source] = URL.createObjectURL(blob);

                const samplesPerBar = Math.floor(samples.length / numBars);
                const bars: number[] = [];
                for (let i = 0; i < numBars; i++) {
                    const start = i * samplesPerBar;
                    const end = Math.min(start + samplesPerBar, samples.length);
                    let sumSquares = 0;
                    for (let j = start; j < end; j++) {
                        sumSquares += samples[j] * samples[j];
                    }
                    const rms = Math.sqrt(sumSquares / (end - start));
                    bars.push(Math.min(100, Math.max(15, rms * 300)));
                }
                waveforms[source] = bars;
            }

            setStemUrls(urls);
            setStemWaveforms(waveforms);

            setStatus('Complete!');
            setProgress(100);
            addLog(`Finished separation in ${(result.wallMs / 1000).toFixed(2)}s.`, 'success');
            resetProcessingWorkers();
            setState(prev => ({ ...prev, separating: false }));
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
