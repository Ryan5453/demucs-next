// Use global 'ort' from CDN script instead of bundled package
// Import types only - actual runtime comes from CDN
import type { InferenceSession } from 'onnxruntime-web';
import type { LogEntry, ModelType } from '../types';
import {
    workerLoadModel,
    workerRunInference,
    workerUnloadModel,
    setProgressCallback,
    type InferenceResult,
} from './onnx-worker-client';


// Access the global ort object loaded from CDN
const ort = (window as unknown as { ort: typeof import('onnxruntime-web') }).ort;

// Load WASM files from CDN to avoid bundling large files
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0-dev.20251116-b39e144322/dist/';
ort.env.wasm.numThreads = 4;
ort.env.logLevel = 'warning';

// Session for main thread (WebGPU)
let session: InferenceSession | null = null;
let loadedSources: string[] = [];
let currentBackend: 'webgpu' | 'wasm' | null = null;
let usingWorker = false;

// Model URLs on HuggingFace
const MODEL_URLS: Record<ModelType, string> = {
    'htdemucs': 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs.onnx',
    'htdemucs_6s': 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs_6s.onnx',
    'hdemucs_mmi': 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/hdemucs_mmi.onnx',
};



// Sources for each model type - must match the order in the trained model
const MODEL_SOURCES: Record<ModelType, string[]> = {
    'htdemucs': ['drums', 'bass', 'other', 'vocals'],
    'htdemucs_6s': ['drums', 'bass', 'guitar', 'piano', 'other', 'vocals'],
    'hdemucs_mmi': ['drums', 'bass', 'other', 'vocals'],
};

/**
 * Check if WebGPU is available in the current browser
 */
export async function isWebGPUAvailable(): Promise<boolean> {
    if (!navigator.gpu) {
        return false;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter !== null;
    } catch {
        return false;
    }
}


/**
 * Get the current backend being used
 */
export function getBackend(): 'webgpu' | 'wasm' | null {
    return currentBackend;
}

/**
 * Check if using worker for inference
 */
export function isUsingWorker(): boolean {
    return usingWorker;
}

export interface ModelLoadResult {
    success: boolean;
    sources: string[];
    backend?: 'webgpu' | 'wasm';
}

export async function loadModel(
    model: ModelType,
    addLog: (message: string, type: LogEntry['type']) => void,
    preferredBackend: 'webgpu' | 'wasm' = 'webgpu'
): Promise<ModelLoadResult> {
    try {
        addLog(`Starting model load: ${model}...`, 'info');

        const startTime = performance.now();

        // Check WebGPU availability
        const hasWebGPU = await isWebGPUAvailable();

        // HDemucs (v3) uses bidirectional LSTMs which aren't supported on WebGPU
        // So we must use WASM-only for that model
        const requiresWasmOnly = model === 'hdemucs_mmi';
        
        // Use WebGPU only if: user prefers it, it's available, and model supports it
        const useWebGPU = preferredBackend === 'webgpu' && hasWebGPU && !requiresWasmOnly;

        const modelUrl = MODEL_URLS[model];

        console.log('[ONNX] Preferred backend:', preferredBackend);
        console.log('[ONNX] WebGPU available:', hasWebGPU);
        console.log('[ONNX] Model requires WASM only:', requiresWasmOnly);
        console.log('[ONNX] Using WebGPU:', useWebGPU);
        console.log('[ONNX] Model URL:', modelUrl);


        if (useWebGPU) {
            // Try WebGPU (non-blocking, runs on GPU)
            try {
                addLog('Using WebGPU backend', 'info');
                usingWorker = false;

                session = await ort.InferenceSession.create(modelUrl, {
                    executionProviders: ['webgpu'],
                    graphOptimizationLevel: 'all',
                });

                console.log('[ONNX] WebGPU session created successfully');
                currentBackend = 'webgpu';
            } catch (webgpuError) {
                // WebGPU failed - fall back to WASM worker
                console.warn('[ONNX] WebGPU failed, falling back to WASM:', webgpuError);
                addLog('WebGPU failed, falling back to WASM backend', 'info');

                addLog('Using WASM backend (via Web Worker)', 'info');
                usingWorker = true;

                setProgressCallback((message) => {
                    console.log('[ONNX Worker]', message);
                });

                await workerLoadModel(modelUrl);

                console.log('[ONNX] WASM session created in worker (fallback)');
                currentBackend = 'wasm';
            }
        } else {
            // Use worker thread with WASM (prevents main thread blocking)
            addLog('Using WASM backend (via Web Worker)', 'info');
            usingWorker = true;

            setProgressCallback((message) => {
                console.log('[ONNX Worker]', message);
            });

            await workerLoadModel(modelUrl);

            console.log('[ONNX] WASM session created in worker');
            currentBackend = 'wasm';
        }

        // Set sources based on model type
        loadedSources = MODEL_SOURCES[model];
        addLog(`Model sources: ${loadedSources.join(', ')}`, 'info');

        const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
        addLog(`Model loaded in ${loadTime}s`, 'success');

        return { success: true, sources: loadedSources, backend: currentBackend };
    } catch (error) {
        console.error('[ONNX] Failed to load model:', error);
        addLog(`Failed to load model: ${(error as Error).message}`, 'error');
        return { success: false, sources: [] };
    }
}


/**
 * Run inference - routes to main thread or worker based on backend
 * Always returns the same InferenceResult type for both paths
 */
export async function runInference(
    specReal: Float32Array,
    specImag: Float32Array,
    audio: Float32Array,
    specShape: number[],
    audioShape: number[]
): Promise<InferenceResult> {
    if (usingWorker) {
        // Run in worker (WASM)
        return workerRunInference(specReal, specImag, audio, specShape, audioShape);
    } else {
        // Run on main thread (WebGPU)
        if (!session) {
            throw new Error('No session loaded');
        }

        const specRealTensor = new ort.Tensor('float32', specReal, specShape);
        const specImagTensor = new ort.Tensor('float32', specImag, specShape);
        const audioTensor = new ort.Tensor('float32', audio, audioShape);

        const results = await session.run({
            'spec_real': specRealTensor,
            'spec_imag': specImagTensor,
            'audio': audioTensor,
        });

        // Get output tensors
        const outSpecReal = results['out_spec_real'];
        const outSpecImag = results['out_spec_imag'];
        const outWave = results['out_wave'];

        // Extract data as Float32Arrays to match worker output format
        const result: InferenceResult = {
            outSpecReal: outSpecReal.data as Float32Array,
            outSpecImag: outSpecImag.data as Float32Array,
            outWave: outWave.data as Float32Array,
            outSpecShape: outSpecReal.dims as number[],
            outWaveShape: outWave.dims as number[],
        };

        // Dispose all tensors
        specRealTensor.dispose();
        specImagTensor.dispose();
        audioTensor.dispose();
        outSpecReal.dispose();
        outSpecImag.dispose();
        outWave.dispose();

        return result;
    }
}


export function getSession(): InferenceSession | null {
    return session;
}

export function getSources(): string[] {
    return loadedSources;
}

export async function unloadModel(): Promise<void> {
    if (usingWorker) {
        await workerUnloadModel();
    } else if (session) {
        await session.release();
        session = null;
    }
    loadedSources = [];
    currentBackend = null;
    usingWorker = false;
}

export { ort };
