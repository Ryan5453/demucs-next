/**
 * Client for communicating with the STFT Worker.
 */

import type { STFTResult } from '../types';

let worker: Worker | null = null;
let pendingResolve: ((result: STFTResult) => void) | null = null;
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

export function initSTFTWorker(): void {
    if (worker) return;

    worker = new Worker(
        new URL('../workers/stft-worker.ts', import.meta.url),
        { type: 'module' }
    );

    worker.onmessage = (event: MessageEvent<STFTResult & { type: string }>) => {
        if (pendingResolve) {
            pendingResolve(event.data);
            clearPending();
        }
    };

    worker.onerror = (error) => {
        console.error('[STFTWorker] Error:', error);
        failPending(error.message || 'STFT worker failed');
        worker?.terminate();
        worker = null;
    };
}

export function terminateSTFTWorker(): void {
    failPending('STFT worker terminated');
    if (worker) {
        worker.terminate();
        worker = null;
    }
}

export function processSTFT(segmentInterleaved: Float32Array): Promise<STFTResult> {
    if (!worker) {
        initSTFTWorker();
    }

    failPending('Superseded by a new STFT request');

    return new Promise((resolve, reject) => {
        pendingResolve = resolve;
        pendingReject = reject;
        worker!.postMessage(
            { type: 'process', segmentInterleaved },
            [segmentInterleaved.buffer] as unknown as Transferable[]
        );
    });
}
