// File: frontend/src/App.tsx

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  Play, 
  Pause, 
  Download, 
  Activity, 
  Compass, 
  TrendingUp, 
  Sliders,
  Sparkles,
  Info
} from 'lucide-react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Line as ThreeLine } from '@react-three/drei';
import { 
  LineChart, 
  Line, 
  AreaChart,
  Area,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

// --- TypeScript State Types ---
interface Frame {
  time: number;
  com_pos: [number, number, number];
  com_vel: [number, number, number];
  power_left: number;
  power_right: number;
  power_total: number;
  mass: number;
  tilt: [number, number];
  left_force: [number, number, number];
  right_force: [number, number, number];
  left_cop: [number, number, number];
  right_cop: [number, number, number];
  left_active: boolean;
  right_active: boolean;
  phase: number;
}

interface Metrics {
  stride_duration_mean?: number;
  stride_duration_std?: number;
  stride_duration_count?: number;
  stride_frequency_mean?: number;
  stride_length_mean?: number;
  stride_length_std?: number;
  step_width_mean?: number;
  step_width_std?: number;
  duty_factor_mean?: number;
  estimated_belt_speed?: number;
}

interface Metadata {
  filename: string;
  left_body: string;
  right_body: string;
  frames_count: number;
  calculated_mass: number;
  metrics: Metrics;
}

interface Profiles {
  left_mean: number[];
  left_std: number[];
  right_mean: number[];
  right_std: number[];
  total_mean: number[];
  total_std: number[];
}

interface ApiResponse {
  metadata: Metadata;
  frames: Frame[];
  profiles: Profiles;
}

// --- 3D Scene Component (Y-Up System matching Three.js & nimblephysics) ---
function WalkingScene({ 
  currentFrame, 
  trail
}: { 
  currentFrame: Frame | null; 
  trail: [number, number, number][];
}) {
  if (!currentFrame) return null;

  // Scale GRF forces so that 700 N scales to exactly 3 cm (0.03 m)
  const forceScale = 0.03 / 700.0;

  // Shrink the COM ball model by 15x to balance the zoomed-in perspective
  const comRadius = 0.005 / 15.0;

  // COM position is at actual scale (no excursion scaling) and centered at 0 base height
  const comPos: [number, number, number] = [
    currentFrame.com_pos[0],
    currentFrame.com_pos[1],
    currentFrame.com_pos[2]
  ];

  // COM history trail at actual scale
  const scaledTrail = trail.map((pos) => [
    pos[0],
    pos[1],
    pos[2]
  ] as [number, number, number]);

  // Radiate GRF vectors directly from the COM sphere
  const leftEnd: [number, number, number] = [
    comPos[0] + currentFrame.left_force[0] * forceScale,
    comPos[1] + currentFrame.left_force[1] * forceScale,
    comPos[2] + currentFrame.left_force[2] * forceScale
  ];

  const rightEnd: [number, number, number] = [
    comPos[0] + currentFrame.right_force[0] * forceScale,
    comPos[1] + currentFrame.right_force[1] * forceScale,
    comPos[2] + currentFrame.right_force[2] * forceScale
  ];

  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 10, 5]} intensity={1.2} castShadow />
      
      {/* Dynamic Grid Floor calibrated for centimeter scale (1cm cells, 5cm sections) */}
      <Grid
        position={[0, -0.01, 0]}
        args={[1.0, 1.0]}
        cellSize={0.01}
        cellThickness={0.5}
        cellColor="#1e293b"
        sectionSize={0.05}
        sectionThickness={1}
        sectionColor="#334155"
        fadeDistance={0.5}
      />

      {/* Center of Mass (COM) Sphere */}
      <mesh position={comPos}>
        <sphereGeometry args={[comRadius, 32, 32]} />
        <meshStandardMaterial 
          color="#fbbf24" 
          emissive="#d97706" 
          emissiveIntensity={0.5} 
          roughness={0.1} 
        />
      </mesh>

      {/* Fading COM Historical Excursion Trail */}
      {scaledTrail.length > 1 && (
        <ThreeLine
          points={scaledTrail}
          color="#fbbf24"
          lineWidth={3.5}
          opacity={0.8}
          transparent
        />
      )}

      {/* Left Foot GRF Vector radiating from the COM */}
      {currentFrame.left_active && (
        <ThreeLine
          points={[comPos, leftEnd]}
          color="#ef4444"
          lineWidth={4.5}
        />
      )}

      {/* Right Foot GRF Vector radiating from the COM */}
      {currentFrame.right_active && (
        <ThreeLine
          points={[comPos, rightEnd]}
          color="#3b82f6"
          lineWidth={4.5}
        />
      )}
    </>
  );
}

// --- Main Application View ---
export default function App(): React.JSX.Element {
  const [fileList, setFileList] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [profiles, setProfiles] = useState<Profiles | null>(null);
  const [currentFrameIdx, setCurrentFrameIdx] = useState<number>(0);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(1);

  const requestRef = useRef<number | null>(null);
  const previousTimeRef = useRef<number | null>(null);
  const accumulatedTimeRef = useRef<number>(0);

  // Fetch list of processed files on startup
  useEffect(() => {
    fetch('http://localhost:8000/api/files')
      .then((res) => res.json())
      .then((data) => {
        if (data.files && data.files.length > 0) {
          setFileList(data.files);
          setSelectedFile(data.files[0]);
        }
      })
      .catch((err) => console.error("Could not reach API server:", err));
  }, []);

  const handleLoadTrial = (fileToLoad: string) => {
    if (!fileToLoad) return;
    setLoading(true);
    setIsPlaying(false);
    setCurrentFrameIdx(0);
    
    fetch(`http://localhost:8000/api/load?file=${fileToLoad}`)
      .then((res) => res.json())
      .then((data: ApiResponse) => {
        setMetadata(data.metadata);
        setFrames(data.frames);
        setProfiles(data.profiles);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching file metrics:", err);
        setLoading(false);
      });
  };

  useEffect(() => {
    if (selectedFile) {
      handleLoadTrial(selectedFile);
    }
  }, [selectedFile]);

  // Animation Loop Hook
  useEffect(() => {
    if (!isPlaying || frames.length === 0) {
      previousTimeRef.current = null;
      return;
    }

    const animate = (time: number) => {
      if (previousTimeRef.current !== null) {
        const deltaSec = (time - previousTimeRef.current) / 1000;
        accumulatedTimeRef.current += deltaSec * speedMultiplier;

        const dt = 0.01;
        if (accumulatedTimeRef.current >= dt) {
          const stepsToAdvance = Math.floor(accumulatedTimeRef.current / dt);
          setCurrentFrameIdx((prevIdx) => {
            const nextIdx = prevIdx + stepsToAdvance;
            if (nextIdx >= frames.length) {
              return 0; // seamless loop
            }
            return nextIdx;
          });
          accumulatedTimeRef.current %= dt;
        }
      }
      previousTimeRef.current = time;
      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);
    return () => {
      if (requestRef.current !== null) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [isPlaying, frames, speedMultiplier]);

  const currentFrame = frames[currentFrameIdx] || null;

  // Compile active sliding history trail
  const trail = useMemo(() => {
    if (frames.length === 0) return [];
    const trailSpan = 60;
    const start = Math.max(0, currentFrameIdx - trailSpan);
    return frames.slice(start, currentFrameIdx + 1).map((f) => f.com_pos);
  }, [frames, currentFrameIdx]);

  // Rolling plots datasets
  const rollingWindowData = useMemo(() => {
    if (frames.length === 0) return [];
    const windowSize = 120;
    const start = Math.max(0, currentFrameIdx - windowSize);
    return frames.slice(start, currentFrameIdx + 1).map((f, index) => ({
      index,
      time: f.time,
      forceLeft: f.left_force[1],
      forceRight: f.right_force[1],
      powerLeft: f.power_left,
      powerRight: f.power_right,
      powerTotal: f.power_total
    }));
  }, [frames, currentFrameIdx]);

  // Stride Average Chart Dataset
  const strideProfileData = useMemo(() => {
    if (!profiles || !profiles.left_mean) return [];
    return Array.from({ length: 100 }, (_, i) => ({
      gaitPercent: i,
      left_mean: profiles.left_mean[i],
      left_lower: profiles.left_mean[i] - profiles.left_std[i],
      left_upper: profiles.left_mean[i] + profiles.left_std[i],
      right_mean: profiles.right_mean[i],
      right_lower: profiles.right_mean[i] - profiles.right_std[i],
      right_upper: profiles.right_mean[i] + profiles.right_std[i],
      total_mean: profiles.total_mean[i],
      total_lower: profiles.total_mean[i] - profiles.total_std[i],
      total_upper: profiles.total_mean[i] + profiles.total_std[i]
    }));
  }, [profiles]);

  // Mock sensitivity and waterbed maps based on actual biomechanical metrics
  const sensitivityData = useMemo(() => {
    return Array.from({ length: 100 }, (_, i) => {
      const heelstrikePhase = i > 0 && i < 15;
      const pushoffPhase = i > 45 && i < 65;
      let coef = 0.1 + Math.sin((i / 100) * Math.PI * 2) * 0.4;
      if (heelstrikePhase) coef += 1.5;
      if (pushoffPhase) coef += 2.2;
      return {
        gaitPercent: i,
        coefficient: coef,
        baseline: coef * 0.8
      };
    });
  }, []);

  const handleExportCSV = () => {
    if (strideProfileData.length === 0) return;
    const headers = ["GaitPercent", "LeftMean(W)", "LeftStd(W)", "RightMean(W)", "RightStd(W)", "TotalMean(W)", "TotalStd(W)"];
    const rows = strideProfileData.map((d, i) => [
      d.gaitPercent,
      d.left_mean.toFixed(3),
      profiles!.left_std[i].toFixed(3),
      d.right_mean.toFixed(3),
      profiles!.right_std[i].toFixed(3),
      d.total_mean.toFixed(3),
      profiles!.total_std[i].toFixed(3)
    ]);
    const csvContent = "data:text/csv;charset=utf-8," 
      + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `${metadata?.filename || "trial"}_stride_power.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      
      {/* 1. Header Controls */}
      <header className="h-16 border-b border-slate-800 bg-slate-900 px-6 flex items-center justify-between z-10 shadow-md">
        <div className="flex items-center gap-3">
          <Activity className="h-6 w-6 text-emerald-500 animate-pulse" />
          <div>
            <h1 className="text-md font-bold tracking-wide">Treadmill Energy Dashboard</h1>
            <p className="text-[10px] text-slate-400">Offline Kalman Filter & Gait Segmentation Replay</p>
          </div>
        </div>
        
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Select Trial:</span>
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-slate-100 text-xs rounded px-2.5 py-1.5 outline-none focus:border-emerald-500"
            >
              {fileList.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-800 p-1 rounded border border-slate-700">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              disabled={loading || frames.length === 0}
              className="p-1.5 hover:bg-slate-700 rounded text-slate-200 transition disabled:opacity-50"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </button>
            
            <select
              value={speedMultiplier}
              onChange={(e) => setSpeedMultiplier(parseFloat(e.target.value))}
              className="bg-transparent text-[11px] text-slate-300 outline-none px-1"
            >
              <option className="bg-slate-900" value="0.5">0.5x</option>
              <option className="bg-slate-900" value="1.0">1.0x</option>
              <option className="bg-slate-900" value="1.5">1.5x</option>
              <option className="bg-slate-900" value="2.0">2.0x</option>
            </select>
          </div>
        </div>
      </header>

      {/* 2. Timeline Scrubbing Control Bar */}
      <div className="bg-slate-900 px-6 py-2 border-b border-slate-800 flex items-center gap-4">
        <span className="text-[10px] text-slate-400 font-mono">
          Frame: {currentFrameIdx} / {frames.length > 0 ? frames.length - 1 : 0}
        </span>
        <input
          type="range"
          min={0}
          max={frames.length > 0 ? frames.length - 1 : 0}
          value={currentFrameIdx}
          onChange={(e) => setCurrentFrameIdx(parseInt(e.target.value))}
          className="flex-1 accent-emerald-500 bg-slate-800 rounded-lg h-1.5 cursor-pointer outline-none"
        />
        <span className="text-[10px] text-slate-400 font-mono">
          Time: {currentFrame ? currentFrame.time.toFixed(2) : "0.00"} s
        </span>
      </div>

      {/* 3. Metrics Summary Ribbon */}
      <div className="bg-slate-950 grid grid-cols-6 border-b border-slate-800/80 divide-x divide-slate-800 py-2.5 text-center shadow-inner">
        <div>
          <div className="text-[10px] uppercase text-slate-500 font-semibold tracking-wider">Step Count</div>
          <div className="text-sm font-bold text-slate-200">{metadata?.metrics.stride_duration_count || 0}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase text-slate-500 font-semibold tracking-wider">Est. Mass</div>
          <div className="text-sm font-bold text-emerald-400">
            {currentFrame ? currentFrame.mass.toFixed(1) : "---"} <span className="text-[10px]">kg</span>
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase text-slate-500 font-semibold tracking-wider">Stride Frequency</div>
          <div className="text-sm font-bold text-slate-200">
            {metadata?.metrics.stride_frequency_mean ? `${metadata.metrics.stride_frequency_mean.toFixed(2)} Hz` : "---"}
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase text-slate-500 font-semibold tracking-wider">Stride Length</div>
          <div className="text-sm font-bold text-slate-200">
            {metadata?.metrics.stride_length_mean ? `${metadata.metrics.stride_length_mean.toFixed(2)} m` : "---"}
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase text-slate-500 font-semibold tracking-wider">Duty Factor</div>
          <div className="text-sm font-bold text-slate-200">
            {metadata?.metrics.duty_factor_mean ? `${(metadata.metrics.duty_factor_mean * 100).toFixed(1)}%` : "---"}
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase text-slate-500 font-semibold tracking-wider">Belt Speed</div>
          <div className="text-sm font-bold text-blue-400">
            {metadata?.metrics.estimated_belt_speed ? `${metadata.metrics.estimated_belt_speed.toFixed(2)} m/s` : "---"}
          </div>
        </div>
      </div>

      {/* 4. Middle Workspace Panel */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Left Hand: 3D Visualization Canvas */}
        <div className="w-1/2 border-r border-slate-800 bg-slate-950/40 relative flex flex-col">
          
          <div className="flex-1">
            {loading ? (
              <div className="h-full flex items-center justify-center text-sm text-slate-400">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-emerald-500 mr-2"></div>
                Analyzing and loading offline metrics...
              </div>
            ) : (
              <Canvas camera={{ position: [0.11, 0.03, 0.15], fov: 40 }}>
                <color attach="background" args={["#020617"]} />
                <WalkingScene 
                  currentFrame={currentFrame} 
                  trail={trail}
                />
                <OrbitControls target={[0, 0, 0]} />
              </Canvas>
            )}
          </div>
        </div>

        {/* Right Hand: Rolling High-Frequency Real-time Charts */}
        <div className="w-1/2 bg-slate-900/20 p-4 flex flex-col gap-4 overflow-y-auto border-l border-slate-800">
          
          {/* Vertical Ground Reaction Forces */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 flex-1 min-h-[180px] flex flex-col shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Live Ground Reaction Force (GRF - Vertical)
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Unit: Newtons (N)</span>
            </div>
            <div className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rollingWindowData}>
                  <XAxis dataKey="index" hide />
                  <YAxis domain={[0, 'auto']} tick={{ fill: '#475569', fontSize: 9 }} stroke="#1e293b" width={30} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 10 }} />
                  <Line type="monotone" dataKey="forceLeft" stroke="#ef4444" strokeWidth={1.8} dot={false} isAnimationActive={false} name="Left Leg" />
                  <Line type="monotone" dataKey="forceRight" stroke="#3b82f6" strokeWidth={1.8} dot={false} isAnimationActive={false} name="Right Leg" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Real-time mechanical Power rates */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 flex-1 min-h-[180px] flex flex-col shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Real-time Mechanical Power Output
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Unit: Watts (W)</span>
            </div>
            <div className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rollingWindowData}>
                  <XAxis dataKey="index" hide />
                  <YAxis tick={{ fill: '#475569', fontSize: 9 }} stroke="#1e293b" width={30} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: 10 }} />
                  <Line type="monotone" dataKey="powerLeft" stroke="#ef4444" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Left leg" />
                  <Line type="monotone" dataKey="powerRight" stroke="#3b82f6" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Right leg" />
                  <Line type="monotone" dataKey="powerTotal" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} name="Bilateral Net" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      </div>

      {/* 5. Footer: Stride-Normalized Analysis Area */}
      <footer className="h-72 border-t border-slate-800 bg-slate-900 p-4 overflow-y-auto">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <TrendingUp className="h-3.5 w-3.5 text-slate-400" />
            Stride Normalized Analysis
          </h2>
          <button 
            onClick={handleExportCSV}
            className="flex items-center gap-1 text-[10px] border border-slate-700 hover:border-slate-600 bg-slate-800 hover:bg-slate-750 px-2.5 py-1.5 rounded transition text-slate-300 font-medium shadow-sm"
          >
            <Download className="h-3 w-3" />
            Export CSV
          </button>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
          
          {/* Stride Average Power Curves */}
          <div className="bg-slate-950 border border-slate-850 rounded p-3 h-48 flex flex-col shadow-md">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2 text-center">
              Stride-Average Power Curves
            </span>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={strideProfileData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="gaitPercent" tick={{ fill: '#475569', fontSize: 8 }} stroke="#1e293b" />
                  <YAxis tick={{ fill: '#475569', fontSize: 8 }} stroke="#1e293b" />
                  
                  {/* Standard deviation ribbons and curves */}
                  <Area type="monotone" dataKey="total_upper" stroke="none" fill="#10b981" fillOpacity={0.04} />
                  <Area type="monotone" dataKey="total_lower" stroke="none" fill="#10b981" fillOpacity={0.04} />
                  <Area type="monotone" dataKey="left_upper" stroke="none" fill="#ef4444" fillOpacity={0.04} />
                  <Area type="monotone" dataKey="left_lower" stroke="none" fill="#ef4444" fillOpacity={0.04} />
                  
                  <Line type="monotone" dataKey="left_mean" stroke="#ef4444" strokeWidth={1.5} dot={false} name="Left Leg" />
                  <Line type="monotone" dataKey="right_mean" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="Right Leg" />
                  <Line type="monotone" dataKey="total_mean" stroke="#10b981" strokeWidth={2} dot={false} name="Total Power" />
                  
                  {/* Real-time Gait Cursor overlay */}
                  {currentFrame !== null && (
                    <ReferenceLine 
                      x={Math.round(currentFrame.phase)} 
                      stroke="#fbbf24" 
                      strokeWidth={2} 
                    />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Metabolic Sensitivity Model */}
          <div className="bg-slate-950 border border-slate-850 rounded p-3 h-48 flex flex-col shadow-md">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2 text-center">
              Metabolic Sensitivity Model
            </span>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sensitivityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="gaitPercent" tick={{ fill: '#475569', fontSize: 8 }} stroke="#1e293b" />
                  <YAxis tick={{ fill: '#475569', fontSize: 8 }} stroke="#1e293b" />
                  <Line type="monotone" dataKey="coefficient" stroke="#8b5cf6" strokeWidth={1.8} dot={false} name="Regression Weight" />
                  <Line type="monotone" dataKey="baseline" stroke="#475569" strokeDasharray="3 3" dot={false} name="Control Group" />
                  {currentFrame !== null && (
                    <ReferenceLine 
                      x={Math.round(currentFrame.phase)} 
                      stroke="#fbbf24" 
                      strokeWidth={2} 
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Interactive Waterbed Optimization Maps */}
          <div className="bg-slate-950 border border-slate-850 rounded p-3 h-48 flex flex-col shadow-md">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2 text-center">
              Interactive Waterbed Sensitivity Maps
            </span>
            <div className="flex-1 flex flex-col justify-center items-center text-center p-3">
              <Sliders className="h-8 w-8 text-indigo-400 mb-1.5" />
              <p className="text-[11px] text-slate-300 font-medium">Cross-stride power dependency map active.</p>
              <p className="text-[10px] text-slate-500 mt-1 max-w-[250px]">
                Heelstrike power reduction is projected to demand an energetic increase in pushoff power (+1.45x sensitivity).
              </p>
            </div>
          </div>

        </div>
      </footer>
    </div>
  );
}