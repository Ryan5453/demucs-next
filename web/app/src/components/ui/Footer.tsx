export default function Footer() {
    return (
        <footer className="site-footer">
            <div className="flex justify-center gap-6">
                <a href="/" className="footer-link">Home</a>
                <a href="/about" className="footer-link">About</a>
                <a href="/privacy" className="footer-link">Privacy</a>
                <a href="https://github.com/Ryan5453/demucs-next" target="_blank" rel="noopener noreferrer" className="footer-link">
                    GitHub
                </a>
            </div>
        </footer>
    );
}
