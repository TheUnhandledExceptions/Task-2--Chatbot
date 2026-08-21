"use client";

import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [analytics, setAnalytics] = useState<any>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const fetchAnalytics = async () => {
    try {
      const res = await fetch("http://localhost:8000/analytics");
      if (res.ok) {
        setAnalytics(await res.json());
      }
    } catch (e) {
      console.error("Failed to fetch analytics", e);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current);
        await sendAudio(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Microphone access denied or not available.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const sendAudio = async (audioBlob: Blob) => {
    setIsLoading(true);
    setResult(null);
    const formData = new FormData();
    formData.append("file", audioBlob, "query.webm");

    try {
      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
      fetchAnalytics();
    } catch (error) {
      console.error("Error submitting query:", error);
      setResult({ error: "Failed to connect to the backend server." });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="w-full max-w-4xl space-y-8">
        <header className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            Voice-Enabled RAG
          </h1>
          <p className="text-slate-400 text-lg">
            Speak your query. Retrieve context. Get ultra-fast answers.
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Main Interaction Area */}
          <div className="glass-card rounded-2xl p-8 flex flex-col items-center justify-center space-y-8 min-h-[400px]">
            <div className="relative">
              <button
                onMouseDown={startRecording}
                onMouseUp={stopRecording}
                onTouchStart={startRecording}
                onTouchEnd={stopRecording}
                className={`w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${
                  isRecording
                    ? "bg-red-500 recording-pulse"
                    : "bg-blue-600 hover:bg-blue-500 hover:scale-105 shadow-lg shadow-blue-500/30"
                }`}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-12 w-12 text-white"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                  />
                </svg>
              </button>
            </div>
            <p className="text-slate-300 font-medium">
              {isRecording ? "Listening... Release to send" : "Hold to speak"}
            </p>
            {isLoading && (
              <div className="flex items-center space-x-2 text-blue-400">
                <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Processing...</span>
              </div>
            )}
          </div>

          {/* Results Area */}
          <div className="glass-card rounded-2xl p-6 flex flex-col space-y-4 max-h-[600px] overflow-y-auto">
            <h2 className="text-xl font-semibold text-white border-b border-slate-700 pb-2">Result</h2>
            
            {!result && !isLoading && (
              <div className="flex-1 flex items-center justify-center text-slate-500">
                Awaiting your query...
              </div>
            )}

            {result && (
              <div className="space-y-4">
                {result.error && (
                  <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200">
                    <p className="font-semibold text-red-400">Error</p>
                    <p>{result.error}</p>
                  </div>
                )}
                
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Transcript</p>
                  <div className="p-3 bg-slate-800/50 rounded-lg text-slate-200">
                    {result.query || "No transcription"}
                  </div>
                </div>

                {result.answer && (
                  <div>
                    <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Answer</p>
                    <div className="p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg text-white font-medium leading-relaxed">
                      {result.answer}
                    </div>
                  </div>
                )}
                
                {result.context && (
                  <div>
                    <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Retrieved Context</p>
                    <div className="p-3 bg-slate-800/50 rounded-lg text-slate-300 text-sm max-h-32 overflow-y-auto">
                      {result.context}
                    </div>
                  </div>
                )}

                {result.timings && (
                  <div>
                    <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Latency Breakdown</p>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-slate-800/50 p-2 rounded flex justify-between">
                        <span className="text-slate-400">STT:</span>
                        <span className="font-mono text-emerald-400">{(result.timings.stt * 1000).toFixed(0)} ms</span>
                      </div>
                      {result.timings.retrieval && (
                        <div className="bg-slate-800/50 p-2 rounded flex justify-between">
                          <span className="text-slate-400">Retrieval:</span>
                          <span className="font-mono text-emerald-400">{(result.timings.retrieval * 1000).toFixed(0)} ms</span>
                        </div>
                      )}
                      {result.timings.generation && (
                        <div className="bg-slate-800/50 p-2 rounded flex justify-between">
                          <span className="text-slate-400">Generation:</span>
                          <span className="font-mono text-emerald-400">{(result.timings.generation * 1000).toFixed(0)} ms</span>
                        </div>
                      )}
                      {result.timings.total && (
                        <div className="bg-slate-800/80 p-2 rounded flex justify-between col-span-2 border border-slate-700">
                          <span className="text-slate-300 font-bold">Total Latency:</span>
                          <span className="font-mono text-emerald-400 font-bold">{(result.timings.total * 1000).toFixed(0)} ms</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Analytics Dashboard */}
        <div className="glass-card rounded-2xl p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-white">Latency Analytics</h2>
            {analytics && analytics.count !== undefined && (
              <span className="bg-blue-500/20 text-blue-300 text-xs px-2 py-1 rounded-full border border-blue-500/30">
                {analytics.count} Queries
              </span>
            )}
          </div>
          
          {analytics && !analytics.error ? (
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 flex flex-col items-center">
                <span className="text-slate-400 text-sm mb-1">P50 Latency</span>
                <span className="text-2xl font-bold font-mono text-white">{(analytics.total_pipeline.p50 * 1000).toFixed(0)} <span className="text-sm text-slate-500">ms</span></span>
              </div>
              <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 flex flex-col items-center">
                <span className="text-slate-400 text-sm mb-1">P70 Latency</span>
                <span className="text-2xl font-bold font-mono text-amber-400">{(analytics.total_pipeline.p70 * 1000).toFixed(0)} <span className="text-sm text-slate-500">ms</span></span>
              </div>
              <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 flex flex-col items-center">
                <span className="text-slate-400 text-sm mb-1">P100 Latency</span>
                <span className="text-2xl font-bold font-mono text-red-400">{(analytics.total_pipeline.p100 * 1000).toFixed(0)} <span className="text-sm text-slate-500">ms</span></span>
              </div>
            </div>
          ) : (
            <div className="text-center text-slate-500 py-4">No analytics data yet. Make a query to populate!</div>
          )}
        </div>
      </div>
    </main>
  );
}
