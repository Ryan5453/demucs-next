/**
 * Client for communicating with the ONNX Worker
 * 
 * This module provides a Promise-based API for the main thread to
 * interact with the ONNX worker for WASM inference.
 */

// Message types (must match worker)
interface LoadModelMessage {
    type: 'load';
    modelUrl: string;
}

interface RunInferenceMessage {
    type: 'run';
    specReal: Float32Array;
    specImag: Float32Array;
    audio: Float32Array;
    specShape: number[];
    audioShape: number[];
}

interface UnloadMessage {
    type: 'unload';
}

// Response types
interface LoadResponse {
    type: 'load';
    success: boolean;
    error?: string;
}

interface RunResponse {
    type: 'run';
    success: boolean;
    outSpecReal?: Float32Array;
    outSpecImag?: Float32Array;
    outWave?: Float32Array;
    outSpecShape?: number[];
    outWaveShape?: number[];
    error?: string;
}

interface UnloadResponse {
    type: 'unload';
    success: boolean;
}

interface ProgressResponse {
    type: 'progress';
    message: string;
}

type WorkerResponse = LoadResponse | RunResponse | UnloadResponse | ProgressResponse;

export interface InferenceResult {
    outSpecReal: Float32Array;
    outSpecImag: Float32Array;
    outWave: Float32Array;
    outSpecShape: number[];
    outWaveShape: number[];
}

let worker: Worker | null = null;
let pendingResolve: ((value: WorkerResponse) => void) | null = null;
let pendingReject: ((reason?: unknown) => void) | null = null;
let onProgress: ((message: string) => void) | null = null;

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

/**
 * Initialize the ONNX worker
 */
export function initWorker(): void {
    if (worker) return;

    // Use classic worker (not module) because importScripts doesn't work in module workers
    worker = new Worker(
        new URL('../workers/onnx-worker.ts', import.meta.url)
    );


    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const response = event.data;

        if (response.type === 'progress') {
            if (onProgress) {
                onProgress(response.message);
            }
            return;
        }

        if (pendingResolve) {
            pendingResolve(response);
            clearPending();
        }
    };

    worker.onerror = (error) => {
        console.error('[WorkerClient] Worker error:', error);
        failPending(error.message || 'ONNX worker failed');
    };
}

/**
 * Terminate the worker
 */
export function terminateWorker(): void {
    failPending('ONNX worker terminated');
    if (worker) {
        worker.terminate();
        worker = null;
    }
}

/**
 * Set a callback for progress messages
 */
export function setProgressCallback(callback: ((message: string) => void) | null): void {
    onProgress = callback;
}

/**
 * Load a model in the worker
 */
export async function workerLoadModel(modelUrl: string): Promise<boolean> {
    initWorker();

    const message: LoadModelMessage = { type: 'load', modelUrl };

    const response = await sendMessage(message) as LoadResponse;

    if (!response.success && response.error) {
        throw new Error(response.error);
    }

    return response.success;
}

/**
 * Run inference in the worker
 */
export async function workerRunInference(
    specReal: Float32Array,
    specImag: Float32Array,
    audio: Float32Array,
    specShape: number[],
    audioShape: number[]
): Promise<InferenceResult> {
    if (!worker) {
        throw new Error('Worker not initialized');
    }

    const message: RunInferenceMessage = {
        type: 'run',
        specReal,
        specImag,
        audio,
        specShape,
        audioShape,
    };

    const response = await sendMessage(message) as RunResponse;

    if (!response.success) {
        throw new Error(response.error || 'Inference failed');
    }

    return {
        outSpecReal: response.outSpecReal!,
        outSpecImag: response.outSpecImag!,
        outWave: response.outWave!,
        outSpecShape: response.outSpecShape!,
        outWaveShape: response.outWaveShape!,
    };
}

/**
 * Unload the model in the worker
 */
export async function workerUnloadModel(): Promise<void> {
    if (!worker) return;

    const message: UnloadMessage = { type: 'unload' };
    await sendMessage(message);
}

/**
 * Check if worker is active
 */
export function isWorkerActive(): boolean {
    return worker !== null;
}

/**
 * Send a message to the worker and wait for response
 */
function sendMessage(message: LoadModelMessage | RunInferenceMessage | UnloadMessage): Promise<WorkerResponse> {
    return new Promise((resolve, reject) => {
        pendingResolve = resolve;
        pendingReject = reject;
        worker!.postMessage(message);
    });
}
