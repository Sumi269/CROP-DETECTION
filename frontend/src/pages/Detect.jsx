import { useState, useRef } from "react";
import Navbar from "../components/Navbar";
import "../styles/Detect.css";

export default function Detect() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const videoRef = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    sendToBackend(file);
  };

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
  };

  const captureImage = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 300;
    canvas.height = 250;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, 300, 250);

    canvas.toBlob((blob) => {
      setPreview(URL.createObjectURL(blob));
      sendToBackend(blob);
    }, "image/jpeg");
  };

  const sendToBackend = async (file) => {
  setLoading(true);
  setResult(null);

  const formData = new FormData();

  // MUST BE "image"
  formData.append("image", file);

  try {
    const res = await fetch("http://localhost:5000/predict", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${localStorage.getItem("token")}`
  },
  body: formData,
});
    const data = await res.json();

    console.log("Backend response:", data);

    if (!res.ok) {
      alert(data.error || "Failed");
    } else {
      setResult(data);
    }
  } catch (err) {
    console.log("ERROR:", err);
    alert("Server not reachable");
  }

  setLoading(false);
};

  return (
    <div className="min-h-screen bg-slate-900 pt-24 pb-12 px-6">
      <Navbar />

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">

        {/* LEFT PANEL */}
        <div className="relative rounded-3xl overflow-hidden shadow-2xl min-h-[400px] lg:min-h-[600px] group">
          <img
            src="https://images.unsplash.com/photo-1500382017468-9049fed747ef"
            alt="Crop field"
            className="absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-slate-900/40 to-transparent flex flex-col justify-end p-10">
            <h2 className="text-4xl md:text-5xl font-black text-white mb-4 drop-shadow-lg">AI Crop Detection</h2>
            <p className="text-xl text-slate-200 drop-shadow-md">Upload or capture detailed crop leaves for instantaneous deep-learning analysis.</p>
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div className="glass-panel rounded-3xl p-8 flex flex-col items-center justify-center relative overflow-hidden backdrop-blur-2xl">
          
          <div className="w-16 h-16 bg-gradient-to-tr from-emerald-500 to-green-400 rounded-full flex items-center justify-center text-3xl shadow-[0_0_20px_rgba(16,185,129,0.3)] mb-6 animate-bounce">
              📸
          </div>
          <h2 className="text-3xl font-bold text-white mb-8">Scan & Diagnose</h2>

          {/* Controls */}
          <div className="w-full max-w-md space-y-4 relative z-10">
             <label className="flex items-center justify-center w-full p-4 border-2 border-dashed border-emerald-500/50 rounded-2xl cursor-pointer hover:bg-emerald-500/10 transition-colors group">
               <span className="text-emerald-400 font-medium group-hover:text-emerald-300">Choose Image File...</span>
               <input type="file" className="hidden" onChange={handleFile} accept="image/*" />
             </label>

             <div className="flex gap-4">
                <button onClick={startCamera} className="flex-1 bg-white/10 hover:bg-white/20 border border-white/10 text-white py-4 rounded-2xl font-semibold transition-all shadow-lg hover:shadow-xl">
                  Start Camera
                </button>
                <button onClick={captureImage} className="flex-1 bg-emerald-500 hover:bg-emerald-400 border border-emerald-400 text-white py-4 rounded-2xl font-semibold transition-all shadow-[0_0_15px_rgba(16,185,129,0.4)]">
                  Capture
                </button>
             </div>
          </div>

          <video ref={videoRef} autoPlay className="w-full max-w-md rounded-2xl mt-6 shadow-2xl hidden"></video>

          {preview && (
            <div className="mt-8 relative max-w-md w-full animate-fade-in-up">
              <img src={preview} alt="Preview" className="w-full rounded-2xl border-4 border-white/10 shadow-2xl" />
              {loading && (
                 <div className="absolute inset-0 bg-slate-900/70 backdrop-blur-sm flex flex-col items-center justify-center rounded-2xl border border-emerald-500/50">
                    <span className="w-10 h-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mb-4"></span>
                    <span className="text-emerald-400 font-bold tracking-widest uppercase animate-pulse">Analyzing...</span>
                 </div>
              )}
            </div>
          )}

          {result && !loading && (
            <div className="mt-8 w-full max-w-md bg-slate-800/80 rounded-2xl p-6 border border-white/10 shadow-2xl animate-fade-in-up">
              
              <div className="flex justify-between items-start mb-6">
                 <h2 className="text-2xl font-bold text-white leading-tight flex-1">{result.crop.replace(/_/g, " ")}</h2>
                 <div className={`px-4 py-1 rounded-full text-sm font-bold border shadow-lg ${result.status === "HEALTHY" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/50" : "bg-red-500/20 text-red-400 border-red-500/50"}`}>
                   {result.status === "HEALTHY" ? "🌿 Healthy" : "⚠ Diseased"}
                 </div>
              </div>

              {/* CONFIDENCE BAR */}
              <div className="mb-6">
                <div className="flex justify-between text-sm font-medium mb-2 text-slate-300">
                   <span>Confidence</span>
                   <span className="text-white">{result.confidence}%</span>
                </div>
                <div className="w-full h-3 bg-slate-700/50 rounded-full overflow-hidden border border-white/5 shadow-inner">
                  <div 
                    className={`h-full rounded-full ${result.status === "HEALTHY" ? "bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.8)]" : "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]"}`}
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>

              {/* TOP 3 */}
              {result.top3 && result.top3.length > 0 && (
                 <div className="space-y-4 border-t border-white/10 pt-6">
                   <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Top Candidates</h3>
                   {result.top3.map((item, i) => (
                     <div key={i} className="flex flex-col gap-1">
                       <div className="flex justify-between text-sm text-slate-200">
                         <span className="truncate pr-4">{item.label.split("___")[1]?.replace(/_/g, " ") || item.label}</span>
                         <span className="font-semibold">{item.confidence}%</span>
                       </div>
                       <div className="w-full h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
                         <div className="h-full bg-slate-400 rounded-full" style={{ width: `${item.confidence}%` }}></div>
                       </div>
                     </div>
                   ))}
                </div>
              )}

            </div>
          )}
        </div>
      </div>
    </div>
  );
}