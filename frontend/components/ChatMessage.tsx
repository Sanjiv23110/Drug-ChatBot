interface ChatMessageProps {
    role: 'assistant' | 'user';
    content: string;
    timestamp: string;
}

export default function ChatMessage({ role, content, timestamp }: ChatMessageProps) {
    const isAssistant = role === 'assistant';

    return (
        <div className={`flex w-full mb-6 ${isAssistant ? 'justify-start' : 'justify-end'}`}>
            <div className={`max-w-[70%] flex flex-col ${isAssistant ? 'items-start' : 'items-end'}`}>
                <div
                    className={`px-6 py-4 rounded-2xl text-sm leading-relaxed whitespace-pre-line shadow-sm ${isAssistant
                            ? 'bg-[#1E293B] text-slate-200 border border-slate-700/50 rounded-tl-sm'
                            : 'bg-blue-600 text-white rounded-tr-sm'
                        }`}
                >
                    {content}
                </div>
                <span className="text-[10px] text-slate-500 mt-2 px-1 font-medium">
                    {timestamp}
                </span>
            </div>
        </div>
    );
}
