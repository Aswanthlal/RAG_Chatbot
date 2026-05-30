
"use client";
import React, { useState, useRef } from 'react';

export default function Dashboard() {
  const [urls, setUrls] = useState({ youtube: '', instagram: '' });
  const [metadata, setMetadata] = useState<any>({ a: null, b: null });
  const [chat, setChat] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Fix: Persist session ID across re-renders
  const [sessionId, setSessionId] = useState("session_" + crypto.randomUUID());

  const handleIngest = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ youtube_url: urls.youtube, instagram_url: urls.instagram })
      });
      const resData = await res.json();
      setMetadata({ a: resData.data.video_A, b: resData.data.video_B });
    } catch (e) {
      console.error("Ingestion failed", e);
    }
    setIsLoading(false);
  };

  const handleClear = async () => {
    await fetch('http://127.0.0.1:8000/api/clear', { method: 'POST' });
    setMetadata({ a: null, b: null });
    setChat([]);
    
    // Force LangGraph to start a brand new memory thread!
    setSessionId("session_" + crypto.randomUUID()); 
  };

  const handleSend = async () => {
    if (!input.trim() || !metadata.a?.video_id) return;
    
    const userMsg = input;
    setInput('');
    setChat(prev => [...prev, { role: 'user', content: userMsg }, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg,
          session_id: sessionId,
          video_a_id: metadata.a.video_id,
          video_b_id: metadata.b.video_id
        })
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        let assistantReply = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ') && line !== 'data: [DONE]') {
              assistantReply += line.replace('data: ', '');
              setChat(prev => {
                const newChat = [...prev];
                newChat[newChat.length - 1].content = assistantReply;
                return newChat;
              });
            }
          }
        }
      }
    } catch (e) {
      console.error("Chat failed", e);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-50 p-4 font-mono">
      {/* Control Bar */}
      <div className="flex gap-4 mb-4 bg-zinc-900 p-3 rounded border border-zinc-800">
        <input 
          placeholder="YouTube URL" 
          className="bg-zinc-950 border border-zinc-700 p-2 rounded flex-1 text-xs"
          onChange={(e) => setUrls({ ...urls, youtube: e.target.value })}
        />
        <input 
          placeholder="Instagram Reel URL" 
          className="bg-zinc-950 border border-zinc-700 p-2 rounded flex-1 text-xs"
          onChange={(e) => setUrls({ ...urls, instagram: e.target.value })}
        />
        <button onClick={handleIngest} disabled={isLoading} className="bg-emerald-600 px-4 py-2 rounded text-xs hover:bg-emerald-500 font-bold disabled:opacity-50">
          {isLoading ? 'INGESTING...' : 'INGEST & EMBED'}
        </button>
        <button onClick={handleClear} className="bg-red-900/50 text-red-400 border border-red-900 px-4 py-2 rounded text-xs hover:bg-red-900/80 font-bold">
          RESET DB
        </button>
      </div>

      <div className="flex flex-1 gap-4 overflow-hidden">
        {/* Videos & Metadata Panel */}
        <div className="flex flex-col gap-4 w-1/2 overflow-y-auto pr-2">
          {/* Video A */}
          {metadata.a && (
            <div className="bg-zinc-900 rounded border border-zinc-800 p-3 flex flex-col gap-2">
              <div className="text-xs flex justify-between text-zinc-400">
                <span className="font-bold text-emerald-400">YOUTUBE: {metadata.a.creator}</span>
                <span>ER: {metadata.a.engagement_rate}% | Views: {metadata.a.views.toLocaleString()} | Likes: {metadata.a.likes.toLocaleString()}</span>
              </div>
              <iframe src={`https://www.youtube.com/embed/${metadata.a.video_id}`} className="w-full h-64 rounded border border-zinc-800" allowFullScreen />
            </div>
          )}
          {/* Video B */}
          {metadata.b && (
            <div className="bg-zinc-900 rounded border border-zinc-800 p-3 flex flex-col gap-2">
              <div className="text-xs flex justify-between text-zinc-400">
                <span className="font-bold text-emerald-400">INSTAGRAM: {metadata.b.creator}</span>
                {metadata.b.metadata_unavailable ? (
                  <span className="text-yellow-500">API Telemetry Unavailable</span>
                ) : (
                  <span>ER: {metadata.b.engagement_rate}% | Views: {metadata.b.views.toLocaleString()} | Likes: {metadata.b.likes.toLocaleString()}</span>
                )}
              </div>
              <iframe src={`https://www.instagram.com/p/${metadata.b.video_id}/embed`} className="w-full h-64 rounded border border-zinc-800" />
            </div>
          )}
        </div>

        {/* Chat Interface */}
        <div className="w-1/2 bg-zinc-900 rounded border border-zinc-800 p-4 flex flex-col justify-between">
          <div className="overflow-y-auto flex-1 space-y-4 mb-4 text-xs pr-2">
            {chat.map((msg, i) => (
              <div key={i} className={`p-3 rounded ${msg.role === 'user' ? 'bg-zinc-800 border-l-2 border-zinc-400' : 'bg-zinc-950 border-l-2 border-emerald-500'}`}>
                <div className="font-bold mb-1 uppercase tracking-wider text-[10px] text-zinc-400">
                  {msg.role === 'user' ? 'USER' : 'AGENT'}
                </div>
                <div className="whitespace-pre-wrap">{msg.content}</div>
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <input 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="e.g. Compare the hooks in the first 5 seconds..." 
              className="bg-zinc-950 border border-zinc-700 p-3 rounded flex-1 text-xs focus:outline-none focus:border-emerald-500"
            />
            <button onClick={handleSend} className="bg-zinc-800 border border-zinc-700 px-4 py-2 rounded text-xs hover:bg-zinc-700 font-bold">
              SEND
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}