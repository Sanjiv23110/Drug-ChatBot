"use client";

import { useState, useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import InputBar from './InputBar';

interface Message {
    id: string;
    role: 'assistant' | 'user';
    content: string;
    timestamp: string;
}

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 'welcome',
            role: 'assistant',
            content: "Hello! I'm your Solomind.ai Healthcare Assistant. \nI'm here to help pharmacists with drug information.\nHow can I assist you today?",
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
    ]);
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = async (text: string) => {
        // Add user message
        const userMsg: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: text,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };

        setMessages(prev => [...prev, userMsg]);
        setIsLoading(true);

        try {
            // Using auto-generated type-safe client
            const { chatQuery } = await import('../app/lib/api-client');
            const data = await chatQuery({ query: text });

            const aiResponse: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.answer || "I apologize, but I received an empty response.",
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            };

            setMessages(prev => [...prev, aiResponse]);
        } catch (error) {
            console.error('Error:', error);
            // Fallback message if backend is unreachable
            const errorMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: "Unable to connect to the Solomind Clinical Engine. Please ensure the backend server is running on port 8000.",
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            };
            setMessages(prev => [...prev, errorMsg]);
        } finally {
            setIsLoading(false);
        }
    };


    return (
        <>
            <main className="flex-1 overflow-y-auto w-full max-w-4xl mx-auto pt-24 pb-32 px-4 scrollbar-thin">
                <div className="flex flex-col min-h-0">
                    {messages.map((msg) => (
                        <ChatMessage
                            key={msg.id}
                            role={msg.role}
                            content={msg.content}
                            timestamp={msg.timestamp}
                        />
                    ))}
                    <div ref={messagesEndRef} />
                </div>
            </main>
            <InputBar onSend={handleSendMessage} isLoading={isLoading} />
        </>
    );
}
