import { Brain, User, Settings, Moon } from 'lucide-react';

export default function Navbar() {
    return (
        <nav className="fixed top-0 left-0 right-0 h-16 bg-[#0B1120] border-b border-slate-800 flex items-center justify-between px-6 z-50">
            <div className="flex items-center gap-3">
                <div className="text-blue-500">
                    <Brain size={28} strokeWidth={1.5} />
                </div>
                <div className="flex flex-col">
                    <h1 className="text-base font-semibold text-slate-100 tracking-tight leading-none">Solomind.ai</h1>
                    <span className="text-[10px] text-blue-300/80 font-medium tracking-wide mt-0.5 uppercase">Clinical Support System</span>
                </div>
            </div>

            <div className="flex items-center gap-5 text-slate-400">
                <button className="hover:text-blue-400 transition-colors p-1"><Moon size={20} strokeWidth={1.5} /></button>
                <button className="hover:text-blue-400 transition-colors p-1"><User size={20} strokeWidth={1.5} /></button>
                <button className="hover:text-blue-400 transition-colors p-1"><Settings size={20} strokeWidth={1.5} /></button>
            </div>
        </nav>
    );
}
