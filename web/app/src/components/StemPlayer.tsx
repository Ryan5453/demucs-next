import { useState, useRef } from 'react';
import { Play, Pause, Download } from 'lucide-react';

interface StemPlayerProps {
    index: number;
    name: string;
    label: string;
    audioUrl?: string;
}

export function StemPlayer({ index, name, label, audioUrl }: StemPlayerProps) {
    const audioRef = useRef<HTMLAudioElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [duration, setDuration] = useState(0);

    const togglePlay = async () => {
        if (!audioRef.current || !audioUrl) return;

        if (isPlaying) {
            audioRef.current.pause();
            setIsPlaying(false);
        } else {
            await audioRef.current.play();
            setIsPlaying(true);
        }
    };

    const handleTimeUpdate = () => {
        // Keep audio element tracking time for future use
    };

    const handleLoadedMetadata = () => {
        if (audioRef.current) {
            setDuration(audioRef.current.duration);
        }
    };

    const handleDownload = () => {
        if (!audioUrl) return;
        const a = document.createElement('a');
        a.href = audioUrl;
        a.download = `${name}.wav`;
        a.click();
    };

    const formatTime = (time: number) => {
        const mins = Math.floor(time / 60);
        const secs = Math.floor(time % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <tr>
            <audio
                ref={audioRef}
                src={audioUrl}
                onEnded={() => setIsPlaying(false)}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
            />

            {/* Index */}
            <td>{index}</td>

            {/* Stem Name */}
            <td>{label}</td>

            {/* Duration */}
            <td className="font-mono">{formatTime(duration)}</td>

            {/* Play Button */}
            <td>
                <button
                    onClick={togglePlay}
                    disabled={!audioUrl}
                    className="tiger-icon-btn"
                    title={isPlaying ? 'Pause' : 'Play'}
                >
                    {isPlaying ? <Pause size={10} /> : <Play size={10} />}
                </button>
            </td>

            {/* Download */}
            <td>
                <button
                    onClick={handleDownload}
                    disabled={!audioUrl}
                    className="tiger-icon-btn"
                    title="Download"
                >
                    <Download size={10} />
                </button>
            </td>
        </tr>
    );
}
