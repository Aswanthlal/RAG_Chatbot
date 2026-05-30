
"use client";
import React, { useState, useRef } from 'react';

const formatDuration = (seconds: number) => {
  if (!seconds) return "Unknown";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
};


export default function Dashboard() {
  const [urls, setUrls] = useState({ youtube: '', instagram: '' });
  const [metadata, setMetadata] = useState<any>({ a: null, b: null });
  const [chat, setChat] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  

  // Fix: Persist session ID across re-renders
  const [sessionId, setSessionId] = useState("session_" + crypto.randomUUID());

  const handleIngest = async () => {
    if (!urls.youtube || !urls.instagram) return;
    setIsLoading(true);
    setChat([]);

    try {
      const res = await fetch('http://127.0.0.1:9000/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          // 🚨 Matches Python perfectly AND scrubs tracking tags!
          youtube_url: urls.youtube.split('?')[0], 
          instagram_url: urls.instagram.split('?')[0]
        }),
      });
      
      const data = await res.json();
      console.log("BACKEND RESPONSE:", data);
      
      // Update state with the returned data
      // (Adjust 'data.a' and 'data.b' if your backend uses different keys like data.metadata_a)
      setMetadata({ 
        a: data.video_a || data.metadata_a || data.a || null, 
        b: data.video_b || data.metadata_b || data.b || null 
      });

    } catch (error) {
      console.error('Ingestion failed', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = async () => {
    await fetch('http://127.0.0.1:9000/api/clear', { method: 'POST' });
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
      const response = await fetch('http://127.0.0.1:9000/api/chat', {
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
          {/* --- VIDEO A (YOUTUBE) CARD --- */}
{metadata.a && (
  <div className="bg-zinc-900 p-4 rounded border border-zinc-800">
    <h2 className="font-bold text-red-500 mb-2">YOUTUBE: {metadata.a.creator}</h2>
    <div className="text-xs text-zinc-400 space-y-1">
      <p><span className="text-zinc-50">ER:</span> {metadata?.a?.engagement_rate ?? 0}%</p>
      <p><span className="text-zinc-50">Views:</span> {metadata?.a?.views?.toLocaleString() ?? 0}</p>
      <p><span className="text-zinc-50">Likes:</span> {metadata?.a?.likes?.toLocaleString() ?? 0}</p>
      <p><span className="text-zinc-50">Comments:</span> {metadata?.a?.comments?.toLocaleString() ?? 0}</p>
      <p><span className="text-zinc-50">Subscribers:</span> {metadata?.a?.follower_count?.toLocaleString() ?? 'Hidden'}</p>
    </div>
  </div>
)}

{/* --- VIDEO B (INSTAGRAM) CARD --- */}
{metadata.b && (
  <div className="bg-zinc-900 p-4 rounded border border-zinc-800">
    <h2 className="font-bold text-pink-500 mb-2">INSTAGRAM: {metadata.b.creator}</h2>
    <div className="text-xs text-zinc-400 space-y-1">
      <p><span className="text-zinc-50">ER:</span> {metadata?.b?.engagement_rate ?? 0}%</p>
      <p><span className="text-zinc-50">Views:</span> {metadata?.b?.views?.toLocaleString() ?? 0}</p>
      <p><span className="text-zinc-50">Likes:</span> {metadata?.b?.likes?.toLocaleString() ?? 0}</p>
      <p><span className="text-zinc-50">Comments:</span> {metadata?.b?.comments?.toLocaleString() ?? 0}</p>
      <p><span className="text-zinc-50">Followers:</span> {
        metadata?.b?.follower_count === null 
          ? <span className="text-yellow-500 font-bold">Hidden by IG API</span>
          : metadata?.b?.follower_count?.toLocaleString() ?? 0
      }</p>
    </div>
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