import Navbar from "@/components/Navbar";
import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <div className="flex flex-col h-screen bg-[#0B1120]">
      <Navbar />
      <ChatInterface />
    </div>
  );
}
