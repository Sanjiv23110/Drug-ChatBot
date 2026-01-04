import React, { useState, useRef, useEffect } from 'react';
import ReactGA from 'react-ga4';
import DisclaimerModal from './DisclaimerModal';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: string[];
}

const Chat: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
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
            const response = await fetch('http://localhost:8000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMsg.content }),
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

            // Track errors in Google Analytics (if initialized)
            if (import.meta.env.VITE_GA_MEASUREMENT_ID) {
                ReactGA.event({
                    category: 'Chat',
                    action: 'Query',
                    label: 'Error',
                });
            }
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <>
            {showDisclaimer && <DisclaimerModal onAccept={handleAcceptDisclaimer} />}

            <div className="flex flex-col h-screen max-w-4xl mx-auto p-4">
                <header className="mb-4">
                    <h1 className="text-2xl font-bold text-gray-800">Solomind Drug Chatbot</h1>

                    {/* Warning Banner - Always Visible */}
                    <div className="mt-2 bg-yellow-50 border border-yellow-200 rounded-md p-3">
                        <p className="text-xs text-yellow-800">
                            ⚠️ <strong>Reference Tool Only:</strong> This chatbot provides drug information for informational purposes only.
                            Always verify critical information with official sources and consult healthcare professionals for medical decisions.
                        </p>
                        <p className="text-xs text-yellow-700 mt-1">
                            <strong>Source:</strong> All drug monographs sourced from{' '}
                            <a
                                href="https://health-products.canada.ca/dpd-bdpp/?lang=eng"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="underline hover:text-yellow-900"
                            >
                                Health Canada Drug Product Database
                            </a>{' '}
                            (Official Government of Canada)
                        </p>
                    </div>
                </header>

                <div className="flex-1 overflow-y-auto bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-4">
                    {messages.length === 0 && (
                        <div className="text-center text-gray-500 mt-20">
                            <p>Welcome! Ask me anything about drug monographs.</p>
                        </div>
                    )}

                    {messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`max-w-[80%] rounded-lg p-3 ${msg.role === 'user'
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-100 text-gray-800'
                                    }`}
                            >
                                <p className="whitespace-pre-wrap">{msg.content}</p>
                                {msg.sources && msg.sources.length > 0 && (
                                    <div className="mt-2 text-xs opacity-75 border-t border-gray-300/20 pt-1">
                                        Sources: {msg.sources.join(', ')}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex justify-start mb-4">
                            <div className="bg-gray-100 rounded-lg p-3 text-gray-500">
                                Thinking...
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <form onSubmit={handleSubmit} className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask a question..."
                        className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading}
                        className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
                    >
                        Send
                    </button>
                </form>

                {/* Footer Disclaimer */}
                <div className="mt-3 text-center">
                    <p className="text-xs text-gray-500">
                        Solomind Drug Chatbot v1.0 | For healthcare professionals only |
                        <button
                            onClick={() => setShowDisclaimer(true)}
                            className="ml-1 text-blue-600 hover:underline"
                        >
                            View Full Disclaimer
                        </button>
                    </p>
                </div>
            </div>
        </>
    );
};

export default Chat;
