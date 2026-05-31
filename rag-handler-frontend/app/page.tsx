
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
          youtube_url: urls.youtube.split('?')[0], 
          instagram_url: urls.instagram.split('?')[0]
        }),
      });
      
      const data = await res.json();
      console.log("BACKEND RESPONSE:", data);
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
    setSessionId("session_" + crypto.randomUUID()); 
  };

  const handleSend = async () => {
    if (!input.trim() || !metadata.a?.video_id || isLoading) return;
    
    setIsLoading(true);
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
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-50 p-4 font-mono">
      
      {/* --- NEW HEADER SECTION --- */}
      <div className="flex justify-between items-center mb-4 px-1">
        <h1 className="text-xl font-bold tracking-widest text-zinc-300 uppercase">
          Creator Analytics Engine
        </h1>
        <div className="flex items-center gap-2 text-[10px] text-zinc-400 uppercase tracking-widest font-bold">
          {/* Pulsing Status Dot */}
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          Qdrant Online
        </div>
      </div>
      {/* --------------------------- */}

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
        <button onClick={handleIngest} disabled={isLoading} className="bg-emerald-600 px-4 py-2 rounded text-xs hover:bg-emerald-500 font-bold disabled:opacity-50 tracking-wider">
          {isLoading ? 'INGESTING...' : 'INGEST & EMBED'}
        </button>
        <button onClick={handleClear} className="bg-red-900/50 text-red-400 border border-red-900 px-4 py-2 rounded text-xs hover:bg-red-900/80 font-bold tracking-wider">
          RESET DB
        </button>
      </div>

      <div className="flex flex-1 gap-4 overflow-hidden">
        {/* Videos & Metadata Panel */}
        <div className="flex flex-col gap-4 w-1/2 overflow-y-auto pr-2">
          {/* VIDEO A (YOUTUBE) CARD */}
        {metadata.a && (
          <div className="bg-zinc-900 p-4 rounded border border-zinc-800">
            {/* NEW: YouTube Thumbnail */}
            <div className="w-full h-48 mb-4 rounded overflow-hidden border border-zinc-700 relative group">
              <img 
                src={`https://img.youtube.com/vi/${metadata.a.video_id}/hqdefault.jpg`} 
                alt="YouTube Thumbnail" 
                className="w-full h-full object-cover"
              />
              <div className="absolute top-2 right-2 bg-black/70 px-2 py-1 rounded text-[10px] font-bold">
                {formatDuration(metadata.a.duration)}
              </div>
            </div>

            <h2 className="font-bold text-red-500 mb-2 truncate">YOUTUBE: {metadata.a.creator}</h2>
            <div className="text-xs text-zinc-400 space-y-1">
              <p><span className="text-zinc-50">ER:</span> {metadata?.a?.engagement_rate ?? 0}%</p>
              <p><span className="text-zinc-50">Views:</span> {metadata?.a?.views?.toLocaleString() ?? 0}</p>
              <p><span className="text-zinc-50">Likes:</span> {metadata?.a?.likes?.toLocaleString() ?? 0}</p>
              <p><span className="text-zinc-50">Comments:</span> {metadata?.a?.comments?.toLocaleString() ?? 0}</p>
              <p><span className="text-zinc-50">Subscribers:</span> {metadata?.a?.follower_count?.toLocaleString() ?? 'Hidden'}</p>
            </div>
          </div>
        )}

        {/* VIDEO B (INSTAGRAM) CARD */}
        {metadata.b && (
          <div className="bg-zinc-900 p-4 rounded border border-zinc-800">
            {/* NEW: Instagram Thumbnail */}
            <div className="w-full h-48 mb-4 rounded overflow-hidden border border-zinc-700 bg-zinc-950 flex items-center justify-center">
            {metadata.b.thumbnail_url ? (
                <img 
                  src={metadata.b.thumbnail_url} 
                  alt="Instagram Thumbnail" 
                  className="w-full h-full object-cover"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <span className="text-zinc-600 text-[10px] uppercase font-bold tracking-widest">Reel Thumbnail</span>
              )}
            </div>

            <h2 className="font-bold text-pink-500 mb-2 truncate">INSTAGRAM: {metadata.b.creator}</h2>
            <div className="text-xs text-zinc-400 space-y-1">
              <p><span className="text-zinc-50">ER:</span> {metadata?.b?.engagement_rate ?? 0}%</p>
              <p><span className="text-zinc-50">Views:</span> {metadata?.b?.views?.toLocaleString() ?? 0}</p>
              <p><span className="text-zinc-50">Likes:</span> {
                metadata?.b?.likes === -1 
                  ? <span className="text-yellow-500 font-bold">Hidden</span>
                  : metadata?.b?.likes?.toLocaleString() ?? 0
              }</p>
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
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !isLoading) handleSend();
              }}
              placeholder="e.g. Compare the hooks in the first 5 seconds..." 
              className="bg-zinc-950 border border-zinc-700 p-3 rounded flex-1 text-xs focus:outline-none focus:border-emerald-500 disabled:opacity-50"
              disabled={isLoading} 
            />
            <button 
              onClick={handleSend} 
              disabled={isLoading} 
              className="bg-zinc-800 border border-zinc-700 px-4 py-2 rounded text-xs hover:bg-zinc-700 font-bold disabled:opacity-50"
            >
              SEND
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}