import { StemPlayer } from './StemPlayer';
import { Download } from 'lucide-react';

interface StemControlsProps {
    stemUrls: Record<string, string>;
}

// Labels for all possible stems (4-stem and 6-stem models)
const STEM_LABELS: Record<string, string> = {
    drums: 'Drums',
    bass: 'Bass',
    guitar: 'Guitar',
    piano: 'Piano',
    other: 'Other',
    vocals: 'Vocals',
};

export function StemControls({ stemUrls }: StemControlsProps) {
    // Get stems from the actual separation results
    const stems = Object.keys(stemUrls);
    const hasStemsReady = stems.length > 0;

    const handleDownloadAll = () => {
        stems.forEach((source, index) => {
            const url = stemUrls[source];
            if (url) {
                // Stagger downloads slightly to avoid browser blocking
                setTimeout(() => {
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${source}.wav`;
                    a.click();
                }, index * 200);
            }
        });
    };

    if (!hasStemsReady) {
        return (
            <div className="aqua-inset-panel text-center py-4">
                <span className="text-[12px] text-gray-500">
                    Separate audio to view stems
                </span>
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-2">
            <table className="tiger-table">
                <thead>
                    <tr>
                        <th style={{ width: '30px' }}>#</th>
                        <th>Stem</th>
                        <th style={{ width: '50px' }}>Time</th>
                        <th style={{ width: '40px' }}></th>
                        <th style={{ width: '40px' }}></th>
                    </tr>
                </thead>
                <tbody>
                    {stems.map((source, index) => (
                        <StemPlayer
                            key={source}
                            index={index + 1}
                            name={source}
                            label={STEM_LABELS[source] || source}
                            audioUrl={stemUrls[source]}
                        />
                    ))}
                </tbody>
            </table>

            <div className="flex justify-end">
                <button
                    onClick={handleDownloadAll}
                    className="tiger-transport-btn"
                    style={{ width: 'auto', height: 'auto', padding: '4px 10px', gap: '4px' }}
                >
                    <Download size={12} />
                    <span className="text-[11px]">Download All</span>
                </button>
            </div>
        </div>
    );
}
