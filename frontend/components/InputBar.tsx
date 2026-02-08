import { Send } from 'lucide-react';
import { useState, KeyboardEvent } from 'react';

interface InputBarProps {
    onSend: (message: string) => void;
    isLoading?: boolean;
}

export default function InputBar({ onSend, isLoading }: InputBarProps) {
    const [input, setInput] = useState('');

    const handleSend = () => {
        if (input.trim() && !isLoading) {
            onSend(input);
            setInput('');
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="fixed bottom-0 left-0 right-0 bg-[#0B1120] border-t border-slate-800 p-4 pb-2 z-50">
            <div className="max-w-4xl mx-auto flex flex-col gap-2">
                <div className="relative">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask about drug interactions, clinical trials, dosing guidelines..."
                        className="w-full bg-[#1E293B]/50 border border-slate-700 text-slate-200 text-sm rounded-xl pl-5 pr-12 py-4 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all placeholder:text-slate-500"
                        disabled={isLoading}
                    />
                    <button
                        onClick={handleSend}
                        disabled={!input.trim() || isLoading}
                        className={`absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-lg transition-colors ${input.trim() && !isLoading
                                ? 'bg-blue-600/20 text-blue-400 hover:bg-blue-600/30'
                                : 'text-slate-600 cursor-not-allowed'
                            }`}
                    >
                        <Send size={18} strokeWidth={2} />
                    </button>
                </div>

                <div className="text-center py-2 flex items-center justify-center gap-1">
                    <span className="text-[11px] text-slate-500">This AI provides educational information only and should not replace professional medical judgment.</span>
                    <a href="https://www.canada.ca/en/health-canada/services/drugs-health-products/drug-products/drug-product-database.html" target="_blank" rel="noopener noreferrer" className="text-[11px] text-slate-500 underline hover:text-slate-400 transition-colors">Health Canada DPD</a>
                </div>
            </div>
        </div>
    );
}
