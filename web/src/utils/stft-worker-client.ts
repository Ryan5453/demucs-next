/**
 * Client for communicating with the STFT Worker.
 */

import type { STFTResult } from '../types';

let worker: Worker | null = null;
let pendingResolve: ((result: STFTResult) => void) | null = null;

export function initSTFTWorker(): void {
    if (worker) return;

    worker = new Worker(
        new URL('../workers/stft-worker.ts', import.meta.url),
        { type: 'module' }
    );

    worker.onmessage = (event: MessageEvent<STFTResult & { type: string }>) => {
        if (pendingResolve) {
            pendingResolve(event.data);
            pendingResolve = null;
        }
    };

    worker.onerror = (error) => {
        console.error('[STFTWorker] Error:', error);
        pendingResolve = null;
    };
}

export function terminateSTFTWorker(): void {
    if (worker) {
        worker.terminate();
        worker = null;
    }
}

export function processSTFT(segmentInterleaved: Float32Array): Promise<STFTResult> {
    if (!worker) {
        initSTFTWorker();
    }

    return new Promise((resolve) => {
        pendingResolve = resolve;
        worker!.postMessage(
            { type: 'process', segmentInterleaved },
            [segmentInterleaved.buffer] as unknown as Transferable[]
        );
    });
}
