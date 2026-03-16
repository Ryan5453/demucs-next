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

export const SOURCES = ['drums', 'bass', 'other', 'vocals'] as const;
export type SourceName = typeof SOURCES[number];

export const SAMPLE_RATE = 44100;
export const NFFT = 4096;
export const HOP_LENGTH = NFFT / 4;
export const SEGMENT_SECONDS = 10;
export const SEGMENT_SAMPLES = SEGMENT_SECONDS * SAMPLE_RATE;

export type ModelType = 'htdemucs' | 'htdemucs_6s';
