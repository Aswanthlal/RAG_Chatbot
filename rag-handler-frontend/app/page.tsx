"use client";

import { useState } from "react";
import { Send, Loader2, Video, Smartphone, Activity } from "lucide-react";

export default function Home() {
  const [ytUrl, setYtUrl] = useState("");
  const [igUrl, setIgUrl] = useState("");
  const [metadata, setMetadata] = useState<any>(null);
  const [isIngesting, setIsIngesting] = useState(false);

  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [isTyping, setIsTyping] = useState(false);

  // 1. Ingestion Function
  const handleIngest = async () => {
    setIsIngesting(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ youtube_url: ytUrl, instagram_url: igUrl }),
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Ingestion server error");
      }

      const data = await res.json();
      if (data.status === "success") {
        setMetadata(data.data);
        // Optional: Clear chat history when new videos are loaded
        setMessages([]); 
      }
    } catch (error: any) {
      console.error("Ingestion failed:", error);
      alert(`Pipeline Error: ${error.message}`); // Prevents silent failing during live demos
    }
    setIsIngesting(false);
  };

  // 2. Streaming Chat Function
  const handleChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    // 1. Capture the history *before* adding the new user message 
    // or append the user message so the backend sees the current turn.
    const updatedMessages = [...messages, { role: "user", content: chatInput }];
    setMessages(updatedMessages);
    setChatInput("");
    setIsTyping(true);

    // 2. Add an empty assistant message template for the upcoming stream
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: chatInput, 
          history: updatedMessages // Passing the updated chat log for backend RAG memory
        }),
      });

      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let done = false;
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ") && line !== "data: [DONE]") {
              const text = line.replace("data: ", "");
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1].content += text;
                return updated;
              });
            }
          }
        }
      }
    } catch (error) {
      console.error("Chat failed:", error);
    }
    setIsTyping(false);
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-8 font-sans">
      <header className="mb-8 border-b border-neutral-800 pb-6">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
          Creator RAG Engine
        </h1>
        <p className="text-neutral-400 mt-2">Analyze and compare video engagement dynamically.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* LEFT COLUMN: Data Ingestion & Metadata Cards */}
        <div className="space-y-6">
          <div className="bg-neutral-900 p-6 rounded-xl border border-neutral-800">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" /> Data Pipeline
            </h2>
            <div className="space-y-4">
              <input 
                type="text" 
                value={ytUrl} 
                onChange={(e) => setYtUrl(e.target.value)} 
                className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors"
                placeholder="YouTube URL"
              />
              <input 
                type="text" 
                value={igUrl} 
                onChange={(e) => setIgUrl(e.target.value)} 
                className="w-full bg-neutral-950 border border-neutral-800 rounded-lg px-4 py-3 focus:outline-none focus:border-pink-500 transition-colors"
                placeholder="Instagram Reel URL"
              />
              <button 
                onClick={handleIngest} 
                disabled={isIngesting}
                className="w-full bg-white text-black font-semibold rounded-lg px-4 py-3 hover:bg-neutral-200 transition-colors flex items-center justify-center gap-2"
              >
                {isIngesting ? <Loader2 className="w-5 h-5 animate-spin" /> : "Extract & Vectorize"}
              </button>
            </div>
          </div>

          {metadata && (
            <div className="grid grid-cols-2 gap-4">
              {/* YouTube Card */}
              <div className="bg-neutral-900 p-5 rounded-xl border border-neutral-800">
                <div className="flex items-center justify-between mb-4">
                  <span className="font-semibold text-neutral-300">Video A</span>
                  <Video className="text-red-500 w-6 h-6" />
                </div>
                <div className="space-y-2 text-sm text-neutral-400">
                  <p><span className="text-neutral-500">Views:</span> {metadata.video_A.views.toLocaleString()}</p>
                  <p><span className="text-neutral-500">Likes:</span> {metadata.video_A.likes.toLocaleString()}</p>
                  <div className="mt-4 pt-4 border-t border-neutral-800">
                    <p className="text-2xl font-bold text-white">{metadata.video_A.engagement_rate}%</p>
                    <p className="text-xs text-neutral-500">Engagement Rate</p>
                  </div>
                </div>
              </div>

              {/* Instagram Card */}
              <div className="bg-neutral-900 p-5 rounded-xl border border-neutral-800">
                <div className="flex items-center justify-between mb-4">
                  <span className="font-semibold text-neutral-300">Video B</span>
                  <Smartphone className="text-pink-500 w-6 h-6" />
                </div>
                <div className="space-y-2 text-sm text-neutral-400">
                  <p><span className="text-neutral-500">Views:</span> {metadata.video_B.views.toLocaleString()}</p>
                  <p><span className="text-neutral-500">Likes:</span> {metadata.video_B.likes.toLocaleString()}</p>
                  <div className="mt-4 pt-4 border-t border-neutral-800">
                    <p className="text-2xl font-bold text-white">{metadata.video_B.engagement_rate}%</p>
                    <p className="text-xs text-neutral-500">Engagement Rate</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: RAG Chat Interface */}
        <div className="bg-neutral-900 rounded-xl border border-neutral-800 flex flex-col h-[600px]">
          <div className="p-4 border-b border-neutral-800 bg-neutral-900/50 rounded-t-xl">
            <h2 className="font-semibold">Strategy Agent</h2>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="h-full flex items-center justify-center text-neutral-600 text-sm">
                Ingest videos on the left, then ask me to compare their hooks or engagement!
              </div>
            )}
            {messages.map((m, idx) => (
              <div key={idx} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[80%] p-3 rounded-xl text-sm ${m.role === "user" ? "bg-blue-600 text-white" : "bg-neutral-800 text-neutral-200"}`}>
                  {m.content}
                </div>
              </div>
            ))}
          </div>

          <form onSubmit={handleChat} className="p-4 border-t border-neutral-800 flex gap-2">
            <input 
              type="text" 
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask why Video A performed better..."
              className="flex-1 bg-neutral-950 border border-neutral-800 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500"
              disabled={isTyping}
            />
            <button 
              type="submit" 
              disabled={isTyping || !chatInput.trim()}
              className="bg-white text-black p-2 rounded-lg hover:bg-neutral-200 disabled:opacity-50 transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>

      </div>
    </div>
  );
}