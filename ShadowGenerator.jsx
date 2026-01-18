import React, { useState, useRef, useCallback, useEffect } from 'react';

// Types
interface LightSettings {
  angle: number;
  elevation: number;
  shadowLength: number;
  contactDarkness: number;
  softness: number;
}

interface ImageData {
  foreground: string | null;
  background: string | null;
  depthMap: string | null;
}

export default function ShadowGenerator() {
  const [images, setImages] = useState<ImageData>({
    foreground: null,
    background: null,
    depthMap: null,
  });
  
  const [light, setLight] = useState<LightSettings>({
    angle: 135,
    elevation: 45,
    shadowLength: 150,
    contactDarkness: 0.85,
    softness: 0.7,
  });
  
  const [foregroundMask, setForegroundMask] = useState<ImageData | null>(null);
  const [processing, setProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState<'composite' | 'shadow' | 'mask'>('composite');
  const [scale, setScale] = useState(0.6);
  const [position, setPosition] = useState({ x: 50, y: 70 });
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shadowCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);

  // Handle file upload
  const handleUpload = (type: keyof ImageData) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImages(prev => ({ ...prev, [type]: event.target?.result as string }));
      };
      reader.readAsDataURL(file);
    }
  };

  // Extract foreground using canvas-based color keying
  const extractForeground = useCallback(async (imgSrc: string): Promise<ImageData> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d')!;
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        const sampleSize = 20;
        let bgR = 0, bgG = 0, bgB = 0, samples = 0;
        
        for (let y = 0; y < sampleSize; y++) {
          for (let x = 0; x < sampleSize; x++) {
            const corners = [
              (y * canvas.width + x) * 4,
              (y * canvas.width + (canvas.width - 1 - x)) * 4,
              ((canvas.height - 1 - y) * canvas.width + x) * 4,
              ((canvas.height - 1 - y) * canvas.width + (canvas.width - 1 - x)) * 4,
            ];
            corners.forEach(i => {
              bgR += data[i];
              bgG += data[i + 1];
              bgB += data[i + 2];
              samples++;
            });
          }
        }
        
        bgR /= samples;
        bgG /= samples;
        bgB /= samples;
        
        const threshold = 80;
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          
          const dist = Math.sqrt(
            Math.pow(r - bgR, 2) +
            Math.pow(g - bgG, 2) +
            Math.pow(b - bgB, 2)
          );
          
          if (dist < threshold) {
            data[i + 3] = 0; // Make transparent
          }
        }
        
        ctx.putImageData(imageData, 0, 0);
        resolve({
          foreground: canvas.toDataURL(),
          background: null,
          depthMap: null,
        });
      };
      img.src = imgSrc;
    });
  }, []);

  const generateShadow = useCallback((
    maskCanvas: HTMLCanvasElement,
    settings: LightSettings
  ): HTMLCanvasElement => {
    const { angle, elevation, shadowLength, contactDarkness, softness } = settings;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    canvas.width = maskCanvas.width;
    canvas.height = maskCanvas.height;
    
    const shadowAngleRad = (angle + 180) * Math.PI / 180;
    const elevationFactor = Math.cos(elevation * Math.PI / 180);
    const actualLength = shadowLength * elevationFactor;
    
    const dx = Math.cos(shadowAngleRad);
    const dy = -Math.sin(shadowAngleRad);
    
    const layers = Math.max(1, Math.floor(actualLength / 3));
    
    for (let i = 0; i < layers; i++) {
      const t = i / Math.max(1, layers - 1);
      const offsetX = dx * actualLength * t;
      const offsetY = dy * actualLength * t;
      
      const blur = Math.floor(2 + softness * 30 * t);
      const opacity = (1 - t) * contactDarkness;
      
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.filter = `blur(${blur}px)`;
      ctx.translate(offsetX, offsetY);
      ctx.drawImage(maskCanvas, 0, 0);
      ctx.restore();
    }
    
    ctx.save();
    ctx.globalAlpha = contactDarkness;
    ctx.filter = `blur(${3}px)`;
    ctx.translate(dx * 5, dy * 5);
    ctx.drawImage(maskCanvas, 0, 0);
    ctx.restore();
    
    return canvas;
  }, []);

  const composite = useCallback(async () => {
    if (!images.foreground || !images.background || !canvasRef.current) return;
    
    setProcessing(true);
    
    try {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d')!;
      const shadowCanvas = shadowCanvasRef.current!;
      const shadowCtx = shadowCanvas.getContext('2d')!;
      const maskCanvas = maskCanvasRef.current!;
      const maskCtx = maskCanvas.getContext('2d')!;
      
      const bgImg = await loadImage(images.background);
      const fgResult = await extractForeground(images.foreground);
      const fgImg = await loadImage(fgResult.foreground!);
      
      canvas.width = bgImg.width;
      canvas.height = bgImg.height;
      shadowCanvas.width = bgImg.width;
      shadowCanvas.height = bgImg.height;

      const fgWidth = fgImg.width * scale;
      const fgHeight = fgImg.height * scale;
      const fgX = (canvas.width * position.x / 100) - fgWidth / 2;
      const fgY = (canvas.height * position.y / 100) - fgHeight / 2;
      
      maskCanvas.width = fgWidth;
      maskCanvas.height = fgHeight;
      maskCtx.clearRect(0, 0, fgWidth, fgHeight);
      maskCtx.drawImage(fgImg, 0, 0, fgWidth, fgHeight);

      const maskData = maskCtx.getImageData(0, 0, fgWidth, fgHeight);
      for (let i = 0; i < maskData.data.length; i += 4) {
        const alpha = maskData.data[i + 3];
        maskData.data[i] = 0;
        maskData.data[i + 1] = 0;
        maskData.data[i + 2] = 0;
        maskData.data[i + 3] = alpha;
      }
      maskCtx.putImageData(maskData, 0, 0);
      
      const shadowResult = generateShadow(maskCanvas, light);
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(bgImg, 0, 0);
      
      ctx.save();
      ctx.globalCompositeOperation = 'multiply';
      ctx.drawImage(shadowResult, fgX, fgY);
      ctx.restore();
      
      ctx.drawImage(fgImg, fgX, fgY, fgWidth, fgHeight);
      
      shadowCtx.fillStyle = 'white';
      shadowCtx.fillRect(0, 0, shadowCanvas.width, shadowCanvas.height);
      shadowCtx.drawImage(shadowResult, fgX, fgY);
      
    } catch (error) {
      console.error('Compositing error:', error);
    }
    
    setProcessing(false);
  }, [images, light, scale, position, extractForeground, generateShadow]);

  const loadImage = (src: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  };

  useEffect(() => {
    if (images.foreground && images.background) {
      composite();
    }
  }, [images.foreground, images.background, light, scale, position, composite]);

  const handleDownload = (canvasRef: React.RefObject<HTMLCanvasElement>, filename: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-mono">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center">
            <svg className="w-6 h-6 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">SHADOW<span className="text-amber-500">GEN</span></h1>
            <p className="text-xs text-zinc-500 tracking-widest uppercase">Realistic Shadow Compositor</p>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-73px)]">
        {/* Left Panel - Controls */}
        <aside className="w-80 border-r border-zinc-800 overflow-y-auto">
          {/* Image Uploads */}
          <section className="p-4 border-b border-zinc-800">
            <h2 className="text-xs font-bold tracking-widest text-zinc-500 uppercase mb-4">Input Images</h2>
            
            <div className="space-y-3">
              <UploadBox
                label="Foreground"
                icon="🧍"
                value={images.foreground}
                onChange={handleUpload('foreground')}
              />
              <UploadBox
                label="Background"
                icon="🏫"
                value={images.background}
                onChange={handleUpload('background')}
              />
              <UploadBox
                label="Depth Map (Bonus)"
                icon="🌫️"
                value={images.depthMap}
                onChange={handleUpload('depthMap')}
                optional
              />
            </div>
          </section>

          {/* Light Controls */}
          <section className="p-4 border-b border-zinc-800">
            <h2 className="text-xs font-bold tracking-widest text-zinc-500 uppercase mb-4">💡 Light Direction</h2>
            
            <div className="space-y-4">
              <SliderControl
                label="Angle"
                value={light.angle}
                min={0}
                max={360}
                unit="°"
                onChange={(v) => setLight(l => ({ ...l, angle: v }))}
              />
              <SliderControl
                label="Elevation"
                value={light.elevation}
                min={0}
                max={90}
                unit="°"
                onChange={(v) => setLight(l => ({ ...l, elevation: v }))}
              />
              
              {/* Visual angle indicator */}
              <div className="flex justify-center py-2">
                <div className="relative w-24 h-24 rounded-full border-2 border-zinc-700 bg-zinc-900">
                  <div 
                    className="absolute w-3 h-3 bg-amber-500 rounded-full shadow-lg shadow-amber-500/50"
                    style={{
                      left: '50%',
                      top: '50%',
                      transform: `translate(-50%, -50%) rotate(${-light.angle}deg) translateX(36px)`
                    }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-4 h-4 rounded-full bg-zinc-600" />
                  </div>
                  <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] text-zinc-500">0°</span>
                  <span className="absolute top-1/2 -right-6 -translate-y-1/2 text-[10px] text-zinc-500">90°</span>
                  <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-[10px] text-zinc-500">180°</span>
                  <span className="absolute top-1/2 -left-6 -translate-y-1/2 text-[10px] text-zinc-500">270°</span>
                </div>
              </div>
            </div>
          </section>

          {/* Shadow Controls */}
          <section className="p-4 border-b border-zinc-800">
            <h2 className="text-xs font-bold tracking-widest text-zinc-500 uppercase mb-4">🖤 Shadow Settings</h2>
            
            <div className="space-y-4">
              <SliderControl
                label="Length"
                value={light.shadowLength}
                min={10}
                max={300}
                unit="px"
                onChange={(v) => setLight(l => ({ ...l, shadowLength: v }))}
              />
              <SliderControl
                label="Contact Darkness"
                value={Math.round(light.contactDarkness * 100)}
                min={0}
                max={100}
                unit="%"
                onChange={(v) => setLight(l => ({ ...l, contactDarkness: v / 100 }))}
              />
              <SliderControl
                label="Softness"
                value={Math.round(light.softness * 100)}
                min={0}
                max={100}
                unit="%"
                onChange={(v) => setLight(l => ({ ...l, softness: v / 100 }))}
              />
            </div>
          </section>

          {/* Position Controls */}
          <section className="p-4">
            <h2 className="text-xs font-bold tracking-widest text-zinc-500 uppercase mb-4">📍 Position</h2>
            
            <div className="space-y-4">
              <SliderControl
                label="Scale"
                value={Math.round(scale * 100)}
                min={10}
                max={150}
                unit="%"
                onChange={(v) => setScale(v / 100)}
              />
              <SliderControl
                label="X Position"
                value={position.x}
                min={0}
                max={100}
                unit="%"
                onChange={(v) => setPosition(p => ({ ...p, x: v }))}
              />
              <SliderControl
                label="Y Position"
                value={position.y}
                min={0}
                max={100}
                unit="%"
                onChange={(v) => setPosition(p => ({ ...p, y: v }))}
              />
            </div>
          </section>
        </aside>

        {/* Main Canvas Area */}
        <main className="flex-1 flex flex-col">
          {/* Tabs */}
          <div className="flex items-center gap-1 p-2 border-b border-zinc-800 bg-zinc-900/50">
            {(['composite', 'shadow', 'mask'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === tab
                    ? 'bg-amber-500/20 text-amber-500'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
                }`}
              >
                {tab === 'composite' && '🖼️ '}
                {tab === 'shadow' && '🖤 '}
                {tab === 'mask' && '✂️ '}
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
            
            <div className="flex-1" />
            
            <button
              onClick={() => handleDownload(
                activeTab === 'shadow' ? shadowCanvasRef : 
                activeTab === 'mask' ? maskCanvasRef : canvasRef,
                `${activeTab}.png`
              )}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-amber-500 text-black hover:bg-amber-400 transition-colors"
            >
              ⬇️ Download
            </button>
          </div>

          {/* Canvas Display */}
          <div className="flex-1 p-6 overflow-auto bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-zinc-900 via-zinc-950 to-black">
            {processing && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                <div className="flex items-center gap-3 px-4 py-2 bg-zinc-800 rounded-lg">
                  <div className="w-4 h-4 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm">Processing...</span>
                </div>
              </div>
            )}
            
            <div className="flex items-center justify-center min-h-full">
              {!images.foreground || !images.background ? (
                <div className="text-center text-zinc-500">
                  <div className="text-6xl mb-4">🌅</div>
                  <p className="text-lg">Upload foreground and background images to begin</p>
                </div>
              ) : (
                <div className="relative">
                  <canvas
                    ref={canvasRef}
                    className={`max-w-full max-h-[70vh] rounded-lg shadow-2xl ${activeTab !== 'composite' ? 'hidden' : ''}`}
                  />
                  <canvas
                    ref={shadowCanvasRef}
                    className={`max-w-full max-h-[70vh] rounded-lg shadow-2xl ${activeTab !== 'shadow' ? 'hidden' : ''}`}
                  />
                  <canvas
                    ref={maskCanvasRef}
                    className={`max-w-full max-h-[70vh] rounded-lg shadow-2xl bg-white ${activeTab !== 'mask' ? 'hidden' : ''}`}
                  />
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

function UploadBox({ 
  label, 
  icon, 
  value, 
  onChange, 
  optional = false 
}: { 
  label: string;
  icon: string;
  value: string | null;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  optional?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  
  return (
    <div
      onClick={() => inputRef.current?.click()}
      className={`relative group cursor-pointer rounded-lg border-2 border-dashed transition-all overflow-hidden ${
        value 
          ? 'border-amber-500/50 bg-amber-500/5' 
          : 'border-zinc-700 hover:border-zinc-500 bg-zinc-900'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={onChange}
        className="hidden"
      />
      
      {value ? (
        <div className="relative h-20">
          <img src={value} alt={label} className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            <span className="text-xs text-white">Click to replace</span>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-3 p-3">
          <span className="text-2xl">{icon}</span>
          <div>
            <p className="text-sm font-medium">{label}</p>
            {optional && <p className="text-xs text-zinc-500">Optional</p>}
          </div>
        </div>
      )}
    </div>
  );
}

function SliderControl({
  label,
  value,
  min,
  max,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  unit: string;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-xs text-zinc-400">{label}</span>
        <span className="text-xs font-mono text-amber-500">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1 bg-zinc-700 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-500 [&::-webkit-slider-thumb]:hover:bg-amber-400 [&::-webkit-slider-thumb]:transition-colors"
      />
    </div>
  );
}
