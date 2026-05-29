import React, { useState } from 'react';
import { Play, Pause, Download, Activity, Layers } from 'lucide-react';

export default function App(): React.JSX.Element {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  const handleTogglePlay = (): void => {
    setIsPlaying((prev) => !prev);
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* 1. Header & Controls Panel */}
      <header className="h-16 border-b border-slate-800 bg-slate-900 px-6 flex items-center justify-between z-10">
        <div className="flex items-center gap-3">
          <Activity className="h-6 w-6 text-emerald-500" />
          <h1 className="text-lg font-semibold tracking-wide">Treadmill Energy Dashboard</h1>
        </div>
        
        <div className="flex items-center gap-4">
          <button 
            onClick={handleTogglePlay}
            className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded font-medium text-sm transition"
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isPlaying ? 'Pause Session' : 'Start Session'}
          </button>
        </div>
      </header>

      {/* 2. Middle Dashboard Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Hand: 3D Visualization Canvas Frame */}
        <div className="w-1/2 border-r border-slate-800 bg-slate-950 relative flex items-center justify-center">
          <div className="text-center">
            <Layers className="h-12 w-12 text-slate-700 mx-auto mb-2" />
            <p className="text-sm text-slate-400">3D WebGL Canvas Placeholder</p>
            <p className="text-xs text-slate-600 mt-1">WebGL view will initialize here (TypeScript)</p>
          </div>
        </div>

        {/* Right Hand: High Frequency Rolling Charts Container */}
        <div className="w-1/2 bg-slate-900/40 p-6 flex flex-col gap-6 overflow-y-auto">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 h-48 flex items-center justify-center">
            <span className="text-xs text-slate-500">Live Ground Reaction Force (GRF) Plot</span>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 h-48 flex items-center justify-center">
            <span className="text-xs text-slate-500">Exoskeleton Telemetry Plot</span>
          </div>
        </div>
      </div>

      {/* 3. Footer: Aggregate Analysis Panel */}
      <footer className="h-80 border-t border-slate-800 bg-slate-900 p-6 overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
            Stride Normalized Analysis
          </h2>
          <button className="flex items-center gap-1 text-xs border border-slate-700 hover:border-slate-600 bg-slate-800 px-3 py-1.5 rounded transition">
            <Download className="h-3 w-3" />
            Export CSV
          </button>
        </div>
        <div className="grid grid-cols-3 gap-6">
          <div className="bg-slate-950 border border-slate-850 rounded p-4 h-52 flex items-center justify-center">
            <span className="text-xs text-slate-650">Stride-Average Power Curves</span>
          </div>
          <div className="bg-slate-950 border border-slate-850 rounded p-4 h-52 flex items-center justify-center">
            <span className="text-xs text-slate-650">Metabolic Sensitivity Model</span>
          </div>
          <div className="bg-slate-950 border border-slate-850 rounded p-4 h-52 flex items-center justify-center">
            <span className="text-xs text-slate-650">Waterbed Optimization Maps</span>
          </div>
        </div>
      </footer>
    </div>
  );
}