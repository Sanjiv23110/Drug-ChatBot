import React, { useState, useRef, useEffect } from 'react';
import ReactGA from 'react-ga4';
import DisclaimerModal from './DisclaimerModal';
import { useTheme } from '../context/ThemeContext';
import { Moon, Sun, User, Settings, Send, HelpCircle, Activity, ExternalLink } from 'lucide-react';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: string[];
}

const Chat: React.FC = () => {
    const { theme, toggleTheme } = useTheme();
    const [messages, setMessages] = useState<Message[]>([{
        id: '0',
        role: 'assistant',
        content: "Hello! I'm your Solomind.ai Healthcare Assistant. I'm here to help pharmacists with drug information. How can I assist you today?"
    }]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showDisclaimer, setShowDisclaimer] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Check if user has accepted disclaimer before
    useEffect(() => {
        const hasAccepted = localStorage.getItem('disclaimerAccepted');
        if (!hasAccepted) {
            setShowDisclaimer(true);
        }
    }, []);

    const handleAcceptDisclaimer = () => {
        localStorage.setItem('disclaimerAccepted', 'true');
        localStorage.setItem('disclaimerAcceptedDate', new Date().toISOString());
        setShowDisclaimer(false);

        // Track disclaimer acceptance in Google Analytics (if initialized)
        if (import.meta.env.VITE_GA_MEASUREMENT_ID) {
            ReactGA.event({
                category: 'Legal',
                action: 'Disclaimer Accepted',
                label: 'First Time',
            });
        }
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input
        };

        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: userMsg.content }),
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const data = await response.json();

            const botMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.answer,
                sources: data.sources
            };

            setMessages(prev => [...prev, botMsg]);

            // Track successful query in Google Analytics (if initialized)
            if (import.meta.env.VITE_GA_MEASUREMENT_ID) {
                ReactGA.event({
                    category: 'Chat',
                    action: 'Query',
                    label: 'Success',
                });
            }
        } catch (error) {
            console.error('Error:', error);
            const errorMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: "Sorry, I encountered an error processing your request."
            };
            setMessages(prev => [...prev, errorMsg]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <>
            <DisclaimerModal
                isOpen={showDisclaimer}
                onAccept={handleAcceptDisclaimer}
            />

            <div className="flex flex-col h-screen">
                {/* Header */}
                <header className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 px-6 py-4 flex items-center justify-between transition-colors duration-200">
                    <div className="flex items-center gap-3">
                        {/* Medical Logo Icon */}
                        <div className="w-10 h-10 bg-gradient-to-br from-teal-400 to-cyan-500 rounded-lg flex items-center justify-center shadow-md">
                            <Activity className="w-6 h-6 text-white" strokeWidth={2.5} />
                        </div>
                        <div>
                            <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
                                Solomind.ai
                            </h1>
                            <span className="inline-block px-2 py-0.5 text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded border border-gray-300 dark:border-gray-700">
                                Clinical Support System
                            </span>
                        </div>
                    </div>

                    {/* Header Icons */}
                    <div className="flex items-center gap-2">
                        {/* Theme Toggle */}
                        <button
                            onClick={toggleTheme}
                            className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                            aria-label="Toggle theme"
                        >
                            {theme === 'light' ? (
                                <Moon className="w-5 h-5 text-gray-700 dark:text-gray-300" />
                            ) : (
                                <Sun className="w-5 h-5 text-gray-700 dark:text-gray-300" />
                            )}
                        </button>

                        {/* User Icon */}
                        <button
                            className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                            aria-label="User profile"
                        >
                            <User className="w-5 h-5 text-gray-700 dark:text-gray-300" />
                        </button>

                        {/* Settings Icon */}
                        <button
                            className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                            aria-label="Settings"
                        >
                            <Settings className="w-5 h-5 text-gray-700 dark:text-gray-300" />
                        </button>
                    </div>
                </header>

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
                    {messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div className={`flex gap-3 max-w-3xl ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                {/* Avatar */}
                                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${msg.role === 'assistant'
                                    ? 'bg-gradient-to-br from-emerald-400 to-teal-500'
                                    : 'bg-gradient-to-br from-blue-400 to-indigo-500'
                                    }`}>
                                    {msg.role === 'assistant' ? (
                                        <Activity className="w-4 h-4 text-white" strokeWidth={2.5} />
                                    ) : (
                                        <User className="w-4 h-4 text-white" />
                                    )}
                                </div>

                                {/* Message Bubble */}
                                <div className="flex-1">
                                    <div className={`px-4 py-3 rounded-2xl ${msg.role === 'assistant'
                                        ? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
                                        : 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white'
                                        }`}>
                                        <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                                    </div>
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 px-1">
                                        {new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                                    </p>
                                </div>
                            </div>
                        </div>
                    ))}

                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="flex gap-3 max-w-3xl">
                                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center">
                                    <Activity className="w-4 h-4 text-white animate-pulse" strokeWidth={2.5} />
                                </div>
                                <div className="px-4 py-3 rounded-2xl bg-gray-100 dark:bg-gray-800">
                                    <div className="flex gap-1">
                                        <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                        <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                        <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 px-6 py-4 transition-colors duration-200">
                    <form onSubmit={handleSubmit} className="space-y-3">
                        <div className="flex gap-3">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask about drug interactions, clinical trials, dosing guidelines..."
                                className="flex-1 px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent transition-all"
                                disabled={isLoading}
                            />
                            <button
                                type="submit"
                                disabled={isLoading || !input.trim()}
                                className="p-3 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg"
                            >
                                <Send className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                            <div className="flex items-center gap-2 flex-wrap">
                                <p>Press Enter to send, Shift + Enter for new line.</p>
                                <span className="text-gray-400 dark:text-gray-500">•</span>
                                <p>This AI provides educational information only and should not replace professional medical judgment.</p>
                                <span className="text-gray-400 dark:text-gray-500">•</span>
                                <a
                                    href="https://health-products.canada.ca/dpd-bdpp/"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium underline"
                                >
                                    Health Canada DPD
                                    <ExternalLink className="w-3 h-3" />
                                </a>
                            </div>
                            <button
                                type="button"
                                onClick={() => setShowDisclaimer(true)}
                                className="p-1.5 rounded-full bg-gray-900 dark:bg-gray-800 hover:bg-gray-800 dark:hover:bg-gray-700 transition-colors flex-shrink-0"
                                aria-label="View full disclaimer"
                            >
                                <HelpCircle className="w-4 h-4 text-white" />
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </>
    );
};

export default Chat;
