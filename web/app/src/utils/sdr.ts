/**
 * SDR = 10 * log10(reference_energy / noise_energy), where noise = estimate - reference.
 * Both inputs are interleaved Float32Arrays; lengths get truncated to the shorter.
 */

export interface StemSDR {
    [stem: string]: number;
}

/**
 * Yielding SDR. The energy reduction over a 5-minute stereo track is ~26M
 * iterations and would block the main thread for hundreds of milliseconds;
 * splitting on ``await setTimeout`` lets the browser repaint.
 */
export async function computeSDRAsync(
    estimate: Float32Array,
    reference: Float32Array,
    chunkSamples: number = 1_000_000
): Promise<number> {
    const length = Math.min(estimate.length, reference.length);
    if (length === 0) return Number.NaN;

    let referenceEnergy = 0;
    let noiseEnergy = 0;

    for (let start = 0; start < length; start += chunkSamples) {
        const end = Math.min(start + chunkSamples, length);
        for (let i = start; i < end; i++) {
            const r = reference[i];
            const e = estimate[i];
            const n = e - r;
            referenceEnergy += r * r;
            noiseEnergy += n * n;
        }
        if (end < length) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }

    if (referenceEnergy <= 0) return Number.NaN;
    if (noiseEnergy <= 1e-12) return Number.POSITIVE_INFINITY;
    return 10 * Math.log10(referenceEnergy / noiseEnergy);
}

export function meanFiniteSDR(sdrs: StemSDR): number {
    const values = Object.values(sdrs).filter(v => Number.isFinite(v));
    if (values.length === 0) return Number.NaN;
    return values.reduce((a, b) => a + b, 0) / values.length;
}
