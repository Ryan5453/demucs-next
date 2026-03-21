/**
 * Client for communicating with the ISTFT Worker.
 * Sends inference results to the worker for parallel ISTFT computation.
 */

interface ISTFTRequest {
    specReal: Float32Array;
    specImag: Float32Array;
    wave: Float32Array;
    numSources: number;
    numChannels: number;
    numBins: number;
    numFrames: number;
    segStart: number;
    segLength: number;
    seg: number;
    numSegments: number;
    numSamples: number;
    fadeIn: Float32Array;
    fadeOut: Float32Array;
    overlap: number;
}

export interface ISTFTResult {
    chunks: Float32Array[];
    segStart: number;
    segLength: number;
}

let worker: Worker | null = null;
let pendingResolve: ((result: ISTFTResult) => void) | null = null;
let pendingReject: ((error: Error) => void) | null = null;

function clearPending(): void {
    pendingResolve = null;
    pendingReject = null;
}

function failPending(message: string): void {
    const reject = pendingReject;
    clearPending();
    if (reject) {
        reject(new Error(message));
    }
}

export function initISTFTWorker(): void {
    if (worker) return;

    worker = new Worker(
        new URL('../workers/istft-worker.ts', import.meta.url),
        { type: 'module' }
    );

    worker.onmessage = (event: MessageEvent<ISTFTResult & { type: string }>) => {
        if (pendingResolve) {
            pendingResolve(event.data);
            clearPending();
        }
    };

    worker.onerror = (error) => {
        console.error('[ISTFTWorker] Error:', error);
        failPending(error.message || 'ISTFT worker failed');
        worker?.terminate();
        worker = null;
    };
}

export function terminateISTFTWorker(): void {
    failPending('ISTFT worker terminated');
    if (worker) {
        worker.terminate();
        worker = null;
    }
}

export function processISTFT(request: ISTFTRequest): Promise<ISTFTResult> {
    if (!worker) {
        initISTFTWorker();
    }

    return new Promise((resolve, reject) => {
        pendingResolve = resolve;
        pendingReject = reject;
        worker!.postMessage({
            type: 'process',
            ...request,
        });
    });
}
