/**
 * Web Worker for ONNX Runtime WASM inference
 * 
 * This worker runs ONNX inference off the main thread to prevent UI blocking
 * when using the WASM backend (CPU inference).
 */

// Import ONNX Runtime in worker context
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0-dev.20251116-b39e144322/dist/ort.min.js');

// Declare the types we need from ONNX Runtime
declare function importScripts(...urls: string[]): void;
declare const ort: {
    env: {
        wasm: {
            wasmPaths: string;
            numThreads: number;
        };
        logLevel: string;
    };
    InferenceSession: {
        create(path: string, options?: { executionProviders: string[]; graphOptimizationLevel?: string }): Promise<OnnxSession>;
    };
    Tensor: new (type: string, data: Float32Array, dims: number[]) => OnnxTensor;
};

interface OnnxSession {
    run(feeds: Record<string, OnnxTensor>): Promise<Record<string, OnnxTensor>>;
    release(): Promise<void>;
}

interface OnnxTensor {
    data: Float32Array | Int32Array | BigInt64Array;
    dims: readonly number[];
    dispose(): void;
}

// Configure ONNX Runtime
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0-dev.20251116-b39e144322/dist/';
ort.env.wasm.numThreads = 4;
ort.env.logLevel = 'warning';

let session: OnnxSession | null = null;

// Message types
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

type WorkerMessage = LoadModelMessage | RunInferenceMessage | UnloadMessage;

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



// Handle messages from main thread
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
    const message = event.data;

    switch (message.type) {
        case 'load':
            await handleLoad(message);
            break;
        case 'run':
            await handleRun(message);
            break;
        case 'unload':
            await handleUnload();
            break;
    }
};

async function handleLoad(message: LoadModelMessage): Promise<void> {
    try {
        sendProgress('Creating ONNX session in worker...');

        session = await ort.InferenceSession.create(message.modelUrl, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        });

        const response: LoadResponse = { type: 'load', success: true };
        self.postMessage(response);
    } catch (error) {
        const response: LoadResponse = {
            type: 'load',
            success: false,
            error: (error as Error).message,
        };
        self.postMessage(response);
    }
}

async function handleRun(message: RunInferenceMessage): Promise<void> {
    if (!session) {
        const response: RunResponse = {
            type: 'run',
            success: false,
            error: 'No session loaded',
        };
        self.postMessage(response);
        return;
    }

    try {
        const specRealTensor = new ort.Tensor('float32', message.specReal, message.specShape);
        const specImagTensor = new ort.Tensor('float32', message.specImag, message.specShape);
        const audioTensor = new ort.Tensor('float32', message.audio, message.audioShape);

        const results = await session.run({
            'spec_real': specRealTensor,
            'spec_imag': specImagTensor,
            'audio': audioTensor,
        });

        const outSpecReal = results['out_spec_real'];
        const outSpecImag = results['out_spec_imag'];
        const outWave = results['out_wave'];

        const response: RunResponse = {
            type: 'run',
            success: true,
            outSpecReal: outSpecReal.data as Float32Array,
            outSpecImag: outSpecImag.data as Float32Array,
            outWave: outWave.data as Float32Array,
            outSpecShape: outSpecReal.dims as number[],
            outWaveShape: outWave.dims as number[],
        };

        // Dispose tensors
        specRealTensor.dispose();
        specImagTensor.dispose();
        audioTensor.dispose();

        self.postMessage(response);
    } catch (error) {
        const response: RunResponse = {
            type: 'run',
            success: false,
            error: (error as Error).message,
        };
        self.postMessage(response);
    }
}

async function handleUnload(): Promise<void> {
    if (session) {
        await session.release();
        session = null;
    }
    const response: UnloadResponse = { type: 'unload', success: true };
    self.postMessage(response);
}

function sendProgress(message: string): void {
    const response: ProgressResponse = { type: 'progress', message };
    self.postMessage(response);
}
