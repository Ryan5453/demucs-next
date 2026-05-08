export interface DemucsState {
    modelLoaded: boolean;
    modelLoading: boolean;
    audioLoaded: boolean;
    audioBuffer: AudioBuffer | null;
    audioFile: File | null;
    separating: boolean;
    progress: number;
    status: string;

    logs: LogEntry[];
}



export interface LogEntry {
    timestamp: Date;
    message: string;
    type: 'info' | 'success' | 'error';
}

export interface STFTResult {
    real: Float32Array;
    imag: Float32Array;
    numBins: number;
    numFrames: number;
}

export const SAMPLE_RATE = 44100;
export const NFFT = 4096;
export const HOP_LENGTH = NFFT / 4;
// HTDemucs training length (7.8s); ONNX is traced at this exact size, so
// callers must feed it.
export const SEGMENT_SAMPLES = 343980;
export const SEGMENT_SECONDS = SEGMENT_SAMPLES / SAMPLE_RATE;
export const SEGMENT_OVERLAP = 0.25;

export type ModelType = 'htdemucs' | 'htdemucs_6s';
