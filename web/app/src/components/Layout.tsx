import { Outlet } from 'react-router-dom';
import Footer from './ui/Footer';

export function Layout() {
    return (
        <div className="stage">
            <div className="flex-1 flex flex-col">
                <Outlet />
            </div>
            <Footer />
        </div>
    );
}
