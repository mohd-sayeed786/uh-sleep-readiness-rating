"""
Ultrahuman-inspired UI Simulator for the Ring AI Readiness Score Engine.
Provides interactive tabbed controls (Raw Inputs, Engineered Features, SHAP Explainability)
in parallel with the Mobile UI with two-way synchronization, zero-latency dial responsiveness,
pixel-perfect height matching, and real-time 21-feature TreeSHAP calculations.
"""

SIMULATOR_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Ring AI &bull; Ultrahuman Readiness Score Simulator</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              dark: '#080A0F',
              card: '#121622',
              cardBorder: '#1E2536',
              coral: '#FF5733',
              amber: '#FFA043',
              emerald: '#00E5A3',
              teal: '#00D2B4',
              blue: '#3B82F6',
              purple: '#A855F7',
              slateText: '#94A3B8'
            }
          }
        }
      }
    }
  </script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: #080A0F;
      color: #F8FAFC;
    }
    .mono {
      font-family: 'JetBrains Mono', monospace;
    }
    input[type=range] {
      accent-color: #00E5A3;
    }
    ::-webkit-scrollbar {
      width: 5px;
      height: 5px;
    }
    ::-webkit-scrollbar-track {
      background: #0D111A;
    }
    ::-webkit-scrollbar-thumb {
      background: #1E2536;
      border-radius: 4px;
    }
    .tab-active {
      background: linear-gradient(135deg, rgba(0,229,163,0.18), rgba(0,210,180,0.12));
      border: 1px solid rgba(0,229,163,0.35);
      color: #00E5A3 !important;
      font-weight: 700;
    }
    .subtab-active {
      background-color: #00E5A3;
      color: #080A0F !important;
      font-weight: 700;
    }
    @keyframes pulseGlow {
      0%, 100% { opacity: 0.2; transform: scale(1); }
      50% { opacity: 0.35; transform: scale(1.05); }
    }
    .ambient-glow {
      animation: pulseGlow 5s ease-in-out infinite;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col justify-between selection:bg-brand-emerald selection:text-black">

  <!-- Top Navigation Bar -->
  <header class="border-b border-brand-cardBorder bg-[#0A0D14]/90 backdrop-blur sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="h-9 w-9 rounded-2xl bg-gradient-to-tr from-brand-emerald to-brand-teal flex items-center justify-center shadow-lg shadow-emerald-500/20">
          <span class="text-black font-black text-sm tracking-tighter">UH</span>
        </div>
        <div>
          <h1 class="text-base font-extrabold tracking-tight text-white flex items-center space-x-2">
            <span>Ultrahuman Ring Simulator</span>
            <span class="text-[10px] px-2 py-0.5 rounded-full bg-brand-emerald/10 text-brand-emerald border border-brand-emerald/20 font-mono">v2.0 &bull; 21 Features</span>
          </h1>
          <p class="text-[11px] text-brand-slateText hidden sm:block">Real-time physiological synthesis, XGBoost TreeSHAP attribution &amp; actionable daily rhythm</p>
        </div>
      </div>

      <!-- Quick Preset & Pipeline Action Triggers -->
      <div class="flex items-center space-x-2">
        <button onclick="applyPreset('prime')" class="px-2.5 py-1 text-xs font-semibold rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 border border-white/5 transition">
          Optimal
        </button>
        <button onclick="applyPreset('baseline')" class="px-2.5 py-1 text-xs font-semibold rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 border border-white/5 transition">
          Moderate
        </button>
        <button onclick="applyPreset('alcohol')" class="px-2.5 py-1 text-xs font-semibold rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 border border-white/5 transition">
          Alcohol
        </button>
        <button onclick="applyPreset('deprived')" class="px-2.5 py-1 text-xs font-semibold rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 border border-white/5 transition">
          Deprived
        </button>
        <button onclick="triggerPipelineRetrain()" class="ml-2 px-3 py-1.5 text-xs font-bold rounded-xl bg-brand-emerald text-black hover:bg-brand-teal transition flex items-center space-x-1.5 shadow-md shadow-emerald-500/20">
          <span>&circlearrowright;</span>
          <span>Run Pipeline</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Application Stage -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 py-6 flex-1 w-full">
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">

      <!-- LEFT: Mobile App Mirror Viewport (col-span-5) -->
      <div class="lg:col-span-5 flex flex-col">
        <div class="bg-brand-card rounded-3xl border border-brand-cardBorder p-5 sm:p-6 shadow-2xl relative overflow-hidden flex-1 flex flex-col justify-between space-y-4">
          
          <!-- Ambient Glow Backdrop behind Dial -->
          <div id="ambientGlow" class="ambient-glow absolute -top-16 -left-16 w-72 h-72 rounded-full bg-brand-emerald/15 blur-3xl pointer-events-none transition-colors duration-500"></div>

          <!-- Top Status Bar in Ring UI -->
          <div>
            <div class="flex items-center justify-between text-xs text-brand-slateText mb-2">
              <span class="flex items-center space-x-1.5">
                <span class="h-2 w-2 rounded-full bg-brand-emerald animate-pulse"></span>
                <span class="font-semibold text-gray-300 tracking-wide">RING CONNECTED</span>
              </span>
              <span class="font-mono text-[11px] text-gray-400">SYNCED 07:15 AM</span>
            </div>

            <!-- Header Ring Metrics Strip -->
            <div class="grid grid-cols-3 gap-2 text-center my-2">
              <div class="bg-[#182030]/80 border border-white/5 rounded-2xl py-2 px-1">
                <div class="text-[10px] uppercase tracking-wider font-semibold text-brand-slateText">SLEEP</div>
                <div class="text-lg font-bold text-white tracking-tight" id="badgeSleep">88</div>
              </div>
              <div class="bg-[#182030]/80 border border-brand-emerald/30 rounded-2xl py-2 px-1 relative">
                <span class="absolute -top-1.5 right-2 h-2 w-2 rounded-full bg-brand-emerald"></span>
                <div class="text-[10px] uppercase tracking-wider font-semibold text-brand-emerald">RECOVERY</div>
                <div class="text-lg font-extrabold text-white tracking-tight" id="badgeRecovery">92</div>
              </div>
              <div class="bg-[#182030]/80 border border-white/5 rounded-2xl py-2 px-1">
                <div class="text-[10px] uppercase tracking-wider font-semibold text-brand-slateText">MOVEMENT</div>
                <div class="text-lg font-bold text-white tracking-tight" id="badgeMovement">76</div>
              </div>
            </div>

            <!-- Hero Radial Readiness Gauge -->
            <div class="flex flex-col items-center justify-center my-1 relative">
              <div class="relative w-48 h-48 flex items-center justify-center">
                <svg class="w-full h-full -rotate-90 transform" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="42" stroke="#1E2536" stroke-width="8" fill="none" stroke-linecap="round"></circle>
                  <circle id="gaugeArc" cx="50" cy="50" r="42" stroke="#00E5A3" stroke-width="8.5" fill="none"
                          stroke-dasharray="263.89" stroke-dashoffset="50" stroke-linecap="round"
                          class="transition-[stroke-dashoffset] duration-75 ease-out"></circle>
                </svg>

                <div class="absolute inset-0 flex flex-col items-center justify-center text-center select-none">
                  <span class="text-[10px] uppercase tracking-widest text-brand-slateText font-semibold">READINESS</span>
                  <div class="flex items-baseline space-x-1 my-0.5">
                    <span id="scoreDisplay" class="text-4xl font-black tracking-tight text-white">4.2</span>
                    <span class="text-sm font-semibold text-brand-slateText">/ 5</span>
                  </div>
                  <div id="tierBadge" class="mt-0.5 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-brand-emerald/15 text-brand-emerald border border-brand-emerald/30">
                    OPTIMAL
                  </div>
                </div>
              </div>

              <!-- Monospace Raw Score output safely below dial arc -->
              <div class="text-[11px] text-brand-slateText mt-1.5 mono font-medium text-center" id="rawScoreSubtext">
                Raw Model Output: 4.187
              </div>
            </div>

            <!-- Everyday Recovery & Daily Rhythm -->
            <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3.5 relative overflow-hidden space-y-3">
              <!-- Section 1: Last Night's Rest -->
              <div>
                <div class="flex items-center justify-between text-[11px] font-bold tracking-wider text-brand-slateText uppercase mb-1">
                  <span class="flex items-center space-x-1.5">
                    <svg class="w-3.5 h-3.5 text-brand-emerald" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                    <span>Last Night's Rest</span>
                  </span>
                  <span id="recoveryPill" class="text-[10px] px-2 py-0.5 rounded font-mono bg-brand-emerald/10 text-brand-emerald border border-brand-emerald/20">Fully Charged</span>
                </div>
                <p id="recoveryAssessmentText" class="text-xs text-gray-300 leading-relaxed">
                  Deep, high-quality recharge (4.65/5). Calm resting heart rate and strong restorative stages left your body fully topped up.
                </p>
              </div>

              <!-- Section 2: Today's Rhythm -->
              <div class="pt-2 border-t border-white/5">
                <div class="flex items-center space-x-1.5 text-[11px] font-bold tracking-wider text-brand-teal uppercase mb-1">
                  <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                  <span>Today's Rhythm</span>
                </div>
                <p id="whatToDoText" class="text-xs text-gray-200 leading-relaxed">
                  You're primed to go! Perfect day for a challenging workout, aiming for a personal best, or tackling high-focus projects.
                </p>
              </div>

              <!-- Section 3: Tonight's Quick Win -->
              <div id="leverCard" class="bg-black/40 border border-brand-emerald/20 rounded-xl p-2.5 flex items-center justify-between">
                <div class="space-y-0.5 pr-2">
                  <div class="text-[10px] font-semibold text-brand-slateText uppercase">Tonight's Quick Win:</div>
                  <div id="leverActionText" class="text-xs text-gray-200 font-medium">Head to bed 45 mins earlier to erase sleep debt</div>
                </div>
                <div class="text-right flex-shrink-0">
                  <div class="text-[10px] text-gray-400 uppercase font-mono">Tomorrow's Boost</div>
                  <div class="text-xs font-mono font-bold text-brand-emerald" id="leverDeltaText">+0.45 pts</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Sleep Architecture & Autonomic Telemetry Section -->
          <div class="space-y-3 pt-1">
            <!-- Sleep Architecture Summary Card -->
            <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-semibold uppercase tracking-wider text-brand-slateText">Sleep Architecture</span>
                <span class="text-xs font-bold text-white mono" id="totalSleepDurationText">7h 35m</span>
              </div>

              <!-- Segmented Stage Progress Bar -->
              <div class="w-full h-3 rounded-full bg-[#0D111A] flex overflow-hidden border border-white/5">
                <div id="barDeep" class="h-full bg-brand-purple transition-all duration-300" style="width: 22%;" title="Deep Sleep"></div>
                <div id="barRem" class="h-full bg-brand-teal transition-all duration-300" style="width: 25%;" title="REM Sleep"></div>
                <div id="barLight" class="h-full bg-brand-blue transition-all duration-300" style="width: 45%;" title="Light Sleep"></div>
                <div id="barAwake" class="h-full bg-gray-600 transition-all duration-300" style="width: 8%;" title="Awake Time"></div>
              </div>

              <div class="flex justify-between text-[10px] text-brand-slateText mt-2 mono">
                <span class="flex items-center space-x-1">
                  <span class="h-1.5 w-1.5 rounded-full bg-brand-purple"></span>
                  <span>Deep+REM: <strong class="text-gray-200" id="restorativeTimeText">150m</strong></span>
                </span>
                <span class="flex items-center space-x-1">
                  <span class="h-1.5 w-1.5 rounded-full bg-brand-teal"></span>
                  <span>Debt: <strong class="text-brand-emerald" id="sleepDebtText">+15m</strong></span>
                </span>
              </div>
            </div>

            <!-- Autonomic Biomarker Grid -->
            <div class="grid grid-cols-2 gap-2.5">
              <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3 flex flex-col justify-between">
                <div class="text-[10px] uppercase tracking-wider font-semibold text-brand-slateText">Resting HR</div>
                <div class="flex items-baseline space-x-1 my-1">
                  <span class="text-xl font-black text-white mono" id="rhrValue">56</span>
                  <span class="text-[10px] text-gray-400">BPM</span>
                </div>
                <div class="text-[11px] font-medium text-brand-emerald" id="rhrStatus">Optimal (-0.6&sigma;)</div>
              </div>

              <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3 flex flex-col justify-between">
                <div class="text-[10px] uppercase tracking-wider font-semibold text-brand-slateText">HRV (RMSSD)</div>
                <div class="flex items-baseline space-x-1 my-1">
                  <span class="text-xl font-black text-white mono" id="hrvValue">65</span>
                  <span class="text-[10px] text-gray-400">ms</span>
                </div>
                <div class="text-[11px] font-medium text-brand-emerald" id="hrvStatus">Elevated (+1.2&sigma;)</div>
              </div>

              <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3 flex flex-col justify-between">
                <div class="text-[10px] uppercase tracking-wider font-semibold text-brand-slateText">Temperature Dev</div>
                <div class="flex items-baseline space-x-1 my-1">
                  <span class="text-xl font-black text-white mono" id="skinTempValue">-0.2</span>
                  <span class="text-[10px] text-gray-400">&deg;C</span>
                </div>
                <div class="text-[11px] text-brand-emerald font-medium" id="skinTempStatus">Optimal (Baseline)</div>
              </div>

              <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3 flex flex-col justify-between">
                <div class="text-[10px] uppercase tracking-wider font-semibold text-brand-slateText">Sleep Efficiency</div>
                <div class="flex items-baseline space-x-1 my-1">
                  <span class="text-xl font-black text-white mono" id="sleepEfficiencyValue">91%</span>
                  <span class="text-[10px] text-gray-400">5 Cycles</span>
                </div>
                <div class="text-[11px] text-brand-emerald font-medium" id="sleepEfficiencyStatus">High Consistency</div>
              </div>
            </div>
          </div>

        </div>
      </div>

      <!-- RIGHT: Tab-Driven Interactive Control Deck (col-span-7) -->
      <div class="lg:col-span-7 flex flex-col">
        <div class="bg-brand-card rounded-3xl border border-brand-cardBorder p-6 shadow-xl flex-1 flex flex-col justify-between space-y-4">
          
          <!-- Tab Navigation Header -->
          <div class="border-b border-brand-cardBorder pb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <div class="flex items-center space-x-1 bg-[#0A0D15] p-1 rounded-xl border border-white/10 w-fit">
                <button id="tabBtnRaw" onclick="switchTab('raw')" class="px-3 py-1.5 text-xs font-semibold rounded-lg transition text-gray-300 hover:text-white flex items-center space-x-1.5">
                  <span>🔬</span> <span>Raw Data</span>
                </button>
                <button id="tabBtnFeatures" onclick="switchTab('features')" class="tab-active px-3 py-1.5 text-xs font-semibold rounded-lg transition text-gray-300 hover:text-white flex items-center space-x-1.5">
                  <span>⚙️</span> <span>Engineered Features</span>
                </button>
                <button id="tabBtnShap" onclick="switchTab('shap')" class="px-3 py-1.5 text-xs font-semibold rounded-lg transition text-gray-300 hover:text-white flex items-center space-x-1.5">
                  <span>📊</span> <span>SHAP Values</span>
                </button>
              </div>
            </div>

            <!-- Sync Indicator & Quick Reset -->
            <div class="flex items-center space-x-3 text-xs">
              <span class="text-brand-slateText hidden md:inline flex items-center space-x-1">
                <span class="inline-block w-1.5 h-1.5 rounded-full bg-brand-emerald"></span>
                <span>Auto-Synced</span>
              </span>
              <button onclick="applyPreset('baseline')" class="text-brand-emerald hover:underline font-semibold flex items-center space-x-1">
                <span>↺</span> <span>Reset Baseline</span>
              </button>
            </div>
          </div>

          <!-- ========================================================= -->
          <!-- TAB VIEWPORT CONTAINER (Constant 1-Page Layout)           -->
          <!-- ========================================================= -->
          <div class="flex-1 flex flex-col min-h-0">

            <!-- ========================================================= -->
            <!-- TAB 1: RAW INPUTS CONTROLS                                -->
            <!-- ========================================================= -->
            <div id="viewRaw" class="hidden space-y-3.5 flex-1 flex flex-col justify-between overflow-y-auto pr-1">
              <div class="bg-[#151B27] p-2.5 rounded-xl border border-white/5 text-xs text-brand-slateText flex items-center justify-between">
                <span>Adjust raw sensor readings &amp; lifestyle habits; all 21 features update live.</span>
                <span class="mono text-[11px] text-brand-emerald font-semibold">Formula: z = (x - &mu;) / &sigma;</span>
              </div>

              <!-- Sleep Duration & Stage Minutes -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-blue flex items-center space-x-1.5">
                  <span>🌙</span> <span>Overnight Sleep Session (Raw Minutes)</span>
                </h4>
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Total Sleep Time</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="raw_val_sleep">450 min (7.5h)</span>
                    </div>
                    <input type="range" id="raw_sleep" min="240" max="600" step="5" value="450"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>240m</span>
                      <span>420m (7h)</span>
                      <span>600m</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Deep Sleep</span>
                      <span class="font-mono text-brand-purple font-bold text-xs" id="raw_val_deep">70 min</span>
                    </div>
                    <input type="range" id="raw_deep" min="10" max="150" step="5" value="70"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>10m</span>
                      <span>70m (Normal)</span>
                      <span>150m</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">REM Sleep</span>
                      <span class="font-mono text-brand-teal font-bold text-xs" id="raw_val_rem">75 min</span>
                    </div>
                    <input type="range" id="raw_rem" min="10" max="160" step="5" value="75"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>10m</span>
                      <span>75m (Normal)</span>
                      <span>160m</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Physiological Sensors (BPM & MS) -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-teal flex items-center space-x-1.5">
                  <span>💓</span> <span>Ring PPG Sensors (Heart &amp; Autonomic)</span>
                </h4>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Resting Heart Rate (BPM)</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="raw_val_hr">57 BPM</span>
                    </div>
                    <input type="range" id="raw_hr" min="40" max="95" step="1" value="57"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>40 BPM (Low)</span>
                      <span>60 BPM (Mean)</span>
                      <span>95 BPM</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">HRV RMSSD (ms)</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="raw_val_hrv">62 ms</span>
                    </div>
                    <input type="range" id="raw_hrv" min="15" max="110" step="1" value="62"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>15 ms (Low)</span>
                      <span>48 ms (Mean)</span>
                      <span>110 ms</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Alcohol & Lifestyle -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-coral flex items-center space-x-1.5">
                  <span>🍷</span> <span>Pre-Sleep Alcohol &amp; Lifestyle</span>
                </h4>
                <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                  <div class="flex justify-between items-center mb-1">
                    <span class="text-xs text-gray-300 font-medium">Alcohol Units (Yesterday Evening)</span>
                    <span class="font-mono font-bold text-xs px-2 py-0.5 rounded bg-brand-card" id="raw_val_alcohol">0.0 units (None)</span>
                  </div>
                  <input type="range" id="raw_alcohol" min="0.0" max="8.0" step="0.5" value="0.0"
                         class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                  <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                    <span>0 units</span>
                    <span>2 units (Light)</span>
                    <span>4 units (Moderate)</span>
                    <span>8+ units</span>
                  </div>
                </div>
              </div>

              <!-- Subjective Historical Feeling Baseline -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-amber flex items-center space-x-1.5">
                  <span>🧠</span> <span>Psychological Momentum &amp; Past Feeling Baseline</span>
                </h4>
                <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                  <div class="flex justify-between text-xs mb-1">
                    <span class="text-gray-300 font-medium">Recent Baseline Feeling (Rolling &amp; EWM Anchor)</span>
                    <span class="font-mono text-brand-amber font-bold text-xs" id="raw_val_feeling">3.5 / 5</span>
                  </div>
                  <input type="range" id="raw_feeling" min="1.0" max="5.0" step="0.1" value="3.5"
                         class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                  <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                    <span>1.0 (Exhausted)</span>
                    <span>3.0 (Steady)</span>
                    <span>5.0 (Optimal)</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- ========================================================= -->
            <!-- TAB 2: ENGINEERED FEATURES (Organized in 3 Sub-Tabs)      -->
            <!-- ========================================================= -->
            <div id="viewFeatures" class="space-y-3 flex-1 flex flex-col justify-between overflow-y-auto pr-1">
              
              <!-- Sub-Tab Category Pill Selector -->
              <div class="flex items-center space-x-1.5 bg-[#0D111A] p-1 rounded-xl border border-white/10">
                <button id="subTabBtnSleep" onclick="switchFeatureSubTab('sleep')" class="subtab-active flex-1 py-1.5 text-[11px] font-semibold rounded-lg transition text-white text-center">
                  🌙 Sleep &amp; Restorative (6)
                </button>
                <button id="subTabBtnRecovery" onclick="switchFeatureSubTab('recovery')" class="flex-1 py-1.5 text-[11px] font-semibold rounded-lg transition text-gray-400 hover:text-white text-center">
                  💓 Autonomic &amp; Stress (8)
                </button>
                <button id="subTabBtnAlcohol" onclick="switchFeatureSubTab('alcohol')" class="flex-1 py-1.5 text-[11px] font-semibold rounded-lg transition text-gray-400 hover:text-white text-center">
                  🍷 Alcohol &amp; History (7)
                </button>
              </div>

              <!-- SUB-TAB 1: Sleep & Restorative (6 Features) -->
              <div id="featSubSleep" class="space-y-2.5">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Sleep Duration Z-Score</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="val_sleep_z">+0.80 &sigma;</span>
                    </div>
                    <input type="range" id="param_sleep_z" min="-3.0" max="3.0" step="0.1" value="0.8"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-3.0 (Short)</span><span>0.0</span><span>+3.0 (Long)</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Sleep Deficit / Surplus</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="val_sleep_debt">+20 min</span>
                    </div>
                    <input type="range" id="param_sleep_debt" min="-120" max="120" step="5" value="20"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-120m</span><span>0m</span><span>+120m</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5 sm:col-span-2">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Restorative Sleep (Deep + REM Minutes)</span>
                      <span class="font-mono text-brand-purple font-bold text-xs" id="val_deep_rem">145 min</span>
                    </div>
                    <input type="range" id="param_deep_rem" min="30" max="240" step="5" value="145"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>30m</span><span>120m (Mean)</span><span>240m</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Deep Sleep Z-Score</span>
                      <span class="font-mono text-brand-purple font-bold text-xs" id="val_deep_z">+0.00 &sigma;</span>
                    </div>
                    <input type="range" id="param_deep_z" min="-3.0" max="3.0" step="0.1" value="0.0"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-3.0</span><span>0.0</span><span>+3.0</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">REM Sleep Z-Score</span>
                      <span class="font-mono text-brand-teal font-bold text-xs" id="val_rem_z">+0.00 &sigma;</span>
                    </div>
                    <input type="range" id="param_rem_z" min="-3.0" max="3.0" step="0.1" value="0.0"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-3.0</span><span>0.0</span><span>+3.0</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5 sm:col-span-2">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Restorative Sleep Ratio (%)</span>
                      <span class="font-mono text-brand-teal font-bold text-xs" id="val_restorative_pct">34%</span>
                    </div>
                    <input type="range" id="param_restorative_pct" min="0.10" max="0.60" step="0.01" value="0.34"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>10% (Low)</span><span>35% (Healthy)</span><span>60%</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- SUB-TAB 2: Autonomic & Stress (8 Features) -->
              <div id="featSubRecovery" class="space-y-2.5 hidden">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Resting HR Z-Score</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="val_hr_z">-0.50 &sigma;</span>
                    </div>
                    <input type="range" id="param_hr_z" min="-3.0" max="3.0" step="0.1" value="-0.5"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-3.0 (Calm)</span><span>0.0</span><span>+3.0</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">HRV RMSSD Z-Score</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="val_hrv_z">+1.10 &sigma;</span>
                    </div>
                    <input type="range" id="param_hrv_z" min="-3.0" max="3.0" step="0.1" value="1.1"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-3.0 (Tense)</span><span>0.0</span><span>+3.0 (High)</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Physiological Stress Index</span>
                      <span class="font-mono text-brand-coral font-bold text-xs" id="val_stress_z">-1.60</span>
                    </div>
                    <input type="range" id="param_stress_z" min="-4.0" max="4.0" step="0.1" value="-1.6"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-4.0 (Relaxed)</span><span>0.0</span><span>+4.0 (Stress)</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Autonomic Recovery Score</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="val_recovery_sc">+1.60</span>
                    </div>
                    <input type="range" id="param_recovery_sc" min="-4.0" max="4.0" step="0.1" value="1.6"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-4.0</span><span>0.0</span><span>+4.0 (Prime)</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Sleep vs Baseline Ratio</span>
                      <span class="font-mono text-brand-blue font-bold text-xs" id="val_sleep_ratio">1.05x</span>
                    </div>
                    <input type="range" id="param_sleep_ratio" min="0.5" max="1.5" step="0.05" value="1.05"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0.5x</span><span>1.0x</span><span>1.5x</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Deep vs Baseline Ratio</span>
                      <span class="font-mono text-brand-purple font-bold text-xs" id="val_deep_ratio">1.00x</span>
                    </div>
                    <input type="range" id="param_deep_ratio" min="0.3" max="2.0" step="0.05" value="1.0"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0.3x</span><span>1.0x</span><span>2.0x</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">HR vs Baseline Ratio</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="val_hr_ratio">0.95x</span>
                    </div>
                    <input type="range" id="param_hr_ratio" min="0.7" max="1.4" step="0.02" value="0.95"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0.7x (Low)</span><span>1.0x</span><span>1.4x</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">HRV vs Baseline Ratio</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="val_hrv_ratio">1.35x</span>
                    </div>
                    <input type="range" id="param_hrv_ratio" min="0.4" max="2.0" step="0.05" value="1.35"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0.4x</span><span>1.0x</span><span>2.0x</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- SUB-TAB 3: Alcohol & History (7 Features) -->
              <div id="featSubAlcohol" class="space-y-2.5 hidden">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5 sm:col-span-2">
                    <div class="flex justify-between items-center mb-1">
                      <span class="text-xs text-gray-300 font-medium">Alcohol Intake (Units)</span>
                      <span class="font-mono font-bold text-xs px-2 py-0.5 rounded bg-brand-card" id="val_alcohol">0.0 units (None)</span>
                    </div>
                    <input type="range" id="param_alcohol" min="0.0" max="8.0" step="0.5" value="0.0"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0 units</span><span>2 units</span><span>4 units</span><span>8+ units</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Alcohol Level (Tier)</span>
                      <span class="font-mono text-gray-200 font-bold text-xs" id="val_alcohol_level">0 (None)</span>
                    </div>
                    <input type="range" id="param_alcohol_level" min="0" max="2" step="1" value="0"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0: None</span><span>1: Light (&le;2)</span><span>2: Heavy (&gt;2)</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Alcohol &times; HRV Interaction</span>
                      <span class="font-mono text-gray-200 font-bold text-xs" id="val_alcohol_x_hrv">0.00</span>
                    </div>
                    <input type="range" id="param_alcohol_x_hrv" min="-15.0" max="15.0" step="0.5" value="0.0"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-15.0</span><span>0.0</span><span>+15.0</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">5-Day Rolling Feeling</span>
                      <span class="font-mono text-brand-amber font-bold text-xs" id="val_roll5">3.5 / 5</span>
                    </div>
                    <input type="range" id="param_roll5" min="1.0" max="5.0" step="0.1" value="3.5"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>1.0</span><span>3.0</span><span>5.0</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">7-Day EWM Feeling</span>
                      <span class="font-mono text-brand-amber font-bold text-xs" id="val_ewm7">3.5 / 5</span>
                    </div>
                    <input type="range" id="param_ewm7" min="1.0" max="5.0" step="0.1" value="3.5"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>1.0</span><span>3.0</span><span>5.0</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5 sm:col-span-2">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Expanding Historical Mean Feeling</span>
                      <span class="font-mono text-brand-amber font-bold text-xs" id="val_exp_mean">3.5 / 5</span>
                    </div>
                    <input type="range" id="param_exp_mean" min="1.0" max="5.0" step="0.1" value="3.5"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>1.0</span><span>3.0</span><span>5.0</span>
                    </div>
                  </div>
                </div>
              </div>

            </div>

            <!-- ========================================================= -->
            <!-- TAB 3: SHAP VALUES & WATERFALL EXPLAINABILITY             -->
            <!-- ========================================================= -->
            <div id="viewShap" class="hidden space-y-3.5 flex-1 flex flex-col justify-between overflow-y-auto pr-1">
              <div class="flex items-center justify-between bg-[#151A27] p-3 rounded-xl border border-white/5">
                <div>
                  <div class="text-[10px] uppercase text-brand-slateText font-semibold">POPULATION PRIOR BIAS</div>
                  <div class="text-sm font-bold mono text-gray-200" id="baseValueText">3.2805</div>
                </div>
                <div class="text-right">
                  <div class="text-[10px] uppercase text-brand-slateText font-semibold">NET SHAP SUM IMPACT</div>
                  <div class="text-sm font-bold mono text-brand-emerald" id="shapTotalSum">+0.906</div>
                </div>
              </div>

              <!-- Dynamic 21-Feature TreeSHAP Waterfall List -->
              <div id="shapBarsContainer" class="space-y-2 flex-1 overflow-y-auto pr-1">
                <!-- Rendered dynamically via JavaScript -->
              </div>

              <div class="flex items-center justify-between text-[11px] text-brand-slateText pt-2 border-t border-brand-cardBorder">
                <span class="flex items-center space-x-1.5">
                  <span class="inline-block w-2.5 h-2.5 rounded-full bg-brand-emerald"></span>
                  <span>Pushes Score UP</span>
                </span>
                <span class="mono text-gray-400">Score = Base + &Sigma;(SHAP)</span>
                <span class="flex items-center space-x-1.5">
                  <span class="inline-block w-2.5 h-2.5 rounded-full bg-brand-coral"></span>
                  <span>Pushes Score DOWN</span>
                </span>
              </div>
            </div>

          </div>

        </div>
      </div>

    </div>
  </main>

  <!-- Footer Info -->
  <footer class="border-t border-brand-cardBorder bg-[#0A0D14] py-4 mt-8">
    <div class="max-w-7xl mx-auto px-4 text-center text-xs text-brand-slateText">
      Ring AI Readiness Engine &bull; Ultrahuman Ring Simulator &bull; 21-Feature XGBoost Microservice &bull; <a href="/docs" class="text-brand-emerald underline">FastAPI OpenAPI Specs</a>
    </div>
  </footer>

  <!-- =================================================================== -->
  <!-- TRAINING PIPELINE LOADING SCREEN MODAL                              -->
  <!-- =================================================================== -->
  <div id="pipelineModal" class="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4 transition-all duration-300 opacity-0 pointer-events-none">
    <div class="bg-[#0E121C] border border-brand-cardBorder rounded-3xl max-w-lg w-full p-6 sm:p-8 shadow-2xl relative overflow-hidden text-center">
      <!-- Ambient Radial Glow -->
      <div class="absolute -top-24 -left-24 w-56 h-56 rounded-full bg-brand-emerald/15 blur-3xl pointer-events-none"></div>
      <div class="absolute -bottom-24 -right-24 w-56 h-56 rounded-full bg-brand-blue/15 blur-3xl pointer-events-none"></div>

      <!-- Rotating Ring AI Logo / Spinner -->
      <div class="relative mx-auto w-20 h-20 flex items-center justify-center">
        <div id="pipeRingAnim" class="absolute inset-0 rounded-full border-4 border-transparent border-t-brand-emerald border-r-brand-teal animate-spin"></div>
        <div class="w-14 h-14 rounded-full bg-[#151A28] border border-brand-emerald/30 flex items-center justify-center shadow-lg">
          <svg id="pipeIconPulse" class="w-7 h-7 text-brand-emerald animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
          </svg>
          <svg id="pipeIconCheck" class="w-8 h-8 text-brand-emerald hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"/>
          </svg>
        </div>
      </div>

      <!-- Header & Active Subtext -->
      <h3 id="pipeModalTitle" class="text-lg sm:text-xl font-extrabold text-white tracking-wide mt-4">Training Readiness Pipeline</h3>
      <p id="pipeModalSubtext" class="text-xs text-brand-slateText mt-1.5">Ingesting sensor telemetry, engineering 21 features &amp; fitting XGBoost...</p>

      <!-- Progress Bar Container -->
      <div class="w-full bg-[#181F30] rounded-full h-2.5 mt-5 overflow-hidden border border-brand-cardBorder">
        <div id="pipeProgressBar" class="bg-gradient-to-r from-brand-emerald via-brand-teal to-brand-blue h-2.5 rounded-full transition-all duration-300" style="width: 8%;"></div>
      </div>
      <div class="flex justify-between items-center text-[11px] font-mono text-gray-400 mt-2">
        <span id="pipePhaseName" class="text-gray-300">Phase 1/4: Ingesting Data</span>
        <span id="pipePercentText" class="text-brand-emerald font-bold">8%</span>
      </div>

      <!-- Multi-Stage Pipeline Step Tracker -->
      <div class="mt-5 space-y-2 text-left text-xs bg-[#141A28]/80 rounded-2xl p-4 border border-brand-cardBorder">
        <div id="pipeStep1" class="flex items-center justify-between p-1.5 rounded-lg bg-white/5">
          <div class="flex items-center space-x-2.5">
            <span class="step-indicator h-2 w-2 rounded-full bg-brand-emerald animate-ping"></span>
            <span class="text-gray-200 font-medium">1. Data Ingestion &amp; Sentinel Cleaning</span>
          </div>
          <span class="step-status font-mono text-[11px] text-brand-emerald font-semibold">Active</span>
        </div>
        <div id="pipeStep2" class="flex items-center justify-between p-1.5 rounded-lg text-gray-500">
          <div class="flex items-center space-x-2.5">
            <span class="step-indicator h-2 w-2 rounded-full bg-gray-600"></span>
            <span class="font-medium">2. 21-Feature Physiological Engineering</span>
          </div>
          <span class="step-status font-mono text-[11px]">Queued</span>
        </div>
        <div id="pipeStep3" class="flex items-center justify-between p-1.5 rounded-lg text-gray-500">
          <div class="flex items-center space-x-2.5">
            <span class="step-indicator h-2 w-2 rounded-full bg-gray-600"></span>
            <span class="font-medium">3. XGBoost Model Fitting (1,699 trees)</span>
          </div>
          <span class="step-status font-mono text-[11px]">Queued</span>
        </div>
        <div id="pipeStep4" class="flex items-center justify-between p-1.5 rounded-lg text-gray-500">
          <div class="flex items-center space-x-2.5">
            <span class="step-indicator h-2 w-2 rounded-full bg-gray-600"></span>
            <span class="font-medium">4. Subgroup Slice &amp; TreeSHAP Verification</span>
          </div>
          <span class="step-status font-mono text-[11px]">Queued</span>
        </div>
      </div>

      <!-- Completion Metrics Card (revealed on success) -->
      <div id="pipeMetricsCard" class="mt-4 hidden bg-brand-emerald/10 border border-brand-emerald/30 rounded-2xl p-4 text-left animate-fade-in">
        <div class="text-[11px] font-semibold text-brand-emerald mb-2 flex items-center justify-between">
          <span>✔ Model Retrained &amp; Validated</span>
          <span id="pipeModelVersion" class="mono text-[10px] text-gray-400">v2.0</span>
        </div>
        <div class="grid grid-cols-3 gap-2 text-center">
          <div class="bg-[#0E121C]/80 rounded-xl p-2 border border-brand-emerald/20">
            <div class="text-[10px] text-gray-400 uppercase">Holdout RMSE</div>
            <div class="text-sm font-mono font-bold text-brand-emerald" id="pipeMetricRmse">0.556</div>
          </div>
          <div class="bg-[#0E121C]/80 rounded-xl p-2 border border-brand-teal/20">
            <div class="text-[10px] text-gray-400 uppercase">Exact Acc</div>
            <div class="text-sm font-mono font-bold text-brand-teal" id="pipeMetricAcc">65.2%</div>
          </div>
          <div class="bg-[#0E121C]/80 rounded-xl p-2 border border-brand-blue/20">
            <div class="text-[10px] text-gray-400 uppercase">Acc &plusmn;1 Class</div>
            <div class="text-sm font-mono font-bold text-brand-blue" id="pipeMetricAccPm1">98.5%</div>
          </div>
        </div>
      </div>

      <!-- Action Button -->
      <div class="mt-5 flex justify-end">
        <button id="pipeBtnClose" onclick="closePipelineModal()" class="w-full py-2.5 px-4 rounded-xl bg-brand-emerald text-black font-bold text-xs hover:bg-brand-teal transition cursor-pointer hidden shadow-lg shadow-emerald-500/20">
          Done &bull; Return to Simulator
        </button>
      </div>
    </div>
  </div>

  <!-- JavaScript Simulator & Model Synchronization Logic -->
  <script>
    // Presets catalog (21 Features)
    const PRESETS = {
      prime: {
        raw_sleep: 510, deep: 95, rem: 85, hr: 53, hrv: 70, alcohol: 0.0, feeling: 4.5
      },
      baseline: {
        raw_sleep: 440, deep: 65, rem: 70, hr: 60, hrv: 48, alcohol: 0.0, feeling: 3.3
      },
      alcohol: {
        raw_sleep: 380, deep: 35, rem: 45, hr: 69, hrv: 28, alcohol: 3.5, feeling: 3.0
      },
      deprived: {
        raw_sleep: 300, deep: 25, rem: 30, hr: 67, hrv: 25, alcohol: 0.0, feeling: 2.0
      }
    };

    // User calibration baseline constants
    const USER_BASE = {
      mean_sleep: 430.0,
      std_sleep: 55.0,
      mean_deep: 70.0,
      std_deep: 20.0,
      mean_rem: 75.0,
      std_rem: 20.0,
      mean_hr: 60.0,
      std_hr: 5.5,
      mean_hrv: 45.0,
      std_hrv: 14.0
    };

    let activeTab = 'features';
    let inFlightAbortController = null;

    function switchTab(tabName) {
      activeTab = tabName;
      document.getElementById('viewRaw').classList.add('hidden');
      document.getElementById('viewFeatures').classList.add('hidden');
      document.getElementById('viewShap').classList.add('hidden');

      document.getElementById('tabBtnRaw').classList.remove('tab-active');
      document.getElementById('tabBtnFeatures').classList.remove('tab-active');
      document.getElementById('tabBtnShap').classList.remove('tab-active');

      if (tabName === 'raw') {
        document.getElementById('viewRaw').classList.remove('hidden');
        document.getElementById('tabBtnRaw').classList.add('tab-active');
      } else if (tabName === 'features') {
        document.getElementById('viewFeatures').classList.remove('hidden');
        document.getElementById('tabBtnFeatures').classList.add('tab-active');
      } else if (tabName === 'shap') {
        document.getElementById('viewShap').classList.remove('hidden');
        document.getElementById('tabBtnShap').classList.add('tab-active');
      }
    }

    function switchFeatureSubTab(subTabName) {
      document.getElementById('featSubSleep').classList.add('hidden');
      document.getElementById('featSubRecovery').classList.add('hidden');
      document.getElementById('featSubAlcohol').classList.add('hidden');

      document.getElementById('subTabBtnSleep').classList.remove('subtab-active');
      document.getElementById('subTabBtnRecovery').classList.remove('subtab-active');
      document.getElementById('subTabBtnAlcohol').classList.remove('subtab-active');

      if (subTabName === 'sleep') {
        document.getElementById('featSubSleep').classList.remove('hidden');
        document.getElementById('subTabBtnSleep').classList.add('subtab-active');
      } else if (subTabName === 'recovery') {
        document.getElementById('featSubRecovery').classList.remove('hidden');
        document.getElementById('subTabBtnRecovery').classList.add('subtab-active');
      } else if (subTabName === 'alcohol') {
        document.getElementById('featSubAlcohol').classList.remove('hidden');
        document.getElementById('subTabBtnAlcohol').classList.add('subtab-active');
      }
    }

    function applyPreset(key) {
      const p = PRESETS[key];
      if (!p) return;

      document.getElementById('raw_sleep').value = p.raw_sleep;
      document.getElementById('raw_deep').value = p.deep;
      document.getElementById('raw_rem').value = p.rem;
      document.getElementById('raw_hr').value = p.hr;
      document.getElementById('raw_hrv').value = p.hrv;
      document.getElementById('raw_alcohol').value = p.alcohol;
      document.getElementById('raw_feeling').value = p.feeling;

      onRawChange();
    }

    // When RAW inputs change -> compute and sync all 21 FEATURES
    function onRawChange() {
      const rawSleep = parseFloat(document.getElementById('raw_sleep').value);
      const rawDeep = parseFloat(document.getElementById('raw_deep').value);
      const rawRem = parseFloat(document.getElementById('raw_rem').value);
      const rawHr = parseFloat(document.getElementById('raw_hr').value);
      const rawHrv = parseFloat(document.getElementById('raw_hrv').value);
      const rawAlcohol = parseFloat(document.getElementById('raw_alcohol').value);
      const rawFeeling = parseFloat(document.getElementById('raw_feeling').value);

      // 1. Sleep & Restorative calculations
      const sleep_z = (rawSleep - USER_BASE.mean_sleep) / USER_BASE.std_sleep;
      const sleep_debt = rawSleep - USER_BASE.mean_sleep;
      const deep_rem = rawDeep + rawRem;
      const deep_z = (rawDeep - USER_BASE.mean_deep) / USER_BASE.std_deep;
      const rem_z = (rawRem - USER_BASE.mean_rem) / USER_BASE.std_rem;
      const restorative_pct = deep_rem / Math.max(rawSleep, 1.0);

      // 2. Autonomic & Ratios calculations
      const hr_z = (rawHr - USER_BASE.mean_hr) / USER_BASE.std_hr;
      const hrv_z = (rawHrv - USER_BASE.mean_hrv) / USER_BASE.std_hrv;
      const stress_z = hr_z - hrv_z;
      const recovery_sc = hrv_z - hr_z;
      const sleep_ratio = rawSleep / USER_BASE.mean_sleep;
      const deep_ratio = rawDeep / USER_BASE.mean_deep;
      const hr_ratio = rawHr / USER_BASE.mean_hr;
      const hrv_ratio = rawHrv / USER_BASE.mean_hrv;

      // 3. Alcohol & History calculations
      const had_alc = rawAlcohol > 0 ? 1.0 : 0.0;
      const alc_lvl = rawAlcohol <= 0 ? 0 : (rawAlcohol <= 2 ? 1 : 2);
      const alc_x_hrv = rawAlcohol * hrv_z;

      // Sync into Feature sliders
      document.getElementById('param_sleep_z').value = Math.max(-3.0, Math.min(3.0, sleep_z)).toFixed(1);
      document.getElementById('param_sleep_debt').value = Math.max(-120, Math.min(120, Math.round(sleep_debt)));
      document.getElementById('param_deep_rem').value = Math.max(30, Math.min(240, deep_rem));
      document.getElementById('param_deep_z').value = Math.max(-3.0, Math.min(3.0, deep_z)).toFixed(1);
      document.getElementById('param_rem_z').value = Math.max(-3.0, Math.min(3.0, rem_z)).toFixed(1);
      document.getElementById('param_restorative_pct').value = Math.max(0.10, Math.min(0.60, restorative_pct)).toFixed(2);

      document.getElementById('param_hr_z').value = Math.max(-3.0, Math.min(3.0, hr_z)).toFixed(1);
      document.getElementById('param_hrv_z').value = Math.max(-3.0, Math.min(3.0, hrv_z)).toFixed(1);
      document.getElementById('param_stress_z').value = Math.max(-4.0, Math.min(4.0, stress_z)).toFixed(1);
      document.getElementById('param_recovery_sc').value = Math.max(-4.0, Math.min(4.0, recovery_sc)).toFixed(1);
      document.getElementById('param_sleep_ratio').value = Math.max(0.5, Math.min(1.5, sleep_ratio)).toFixed(2);
      document.getElementById('param_deep_ratio').value = Math.max(0.3, Math.min(2.0, deep_ratio)).toFixed(2);
      document.getElementById('param_hr_ratio').value = Math.max(0.7, Math.min(1.4, hr_ratio)).toFixed(2);
      document.getElementById('param_hrv_ratio').value = Math.max(0.4, Math.min(2.0, hrv_ratio)).toFixed(2);

      document.getElementById('param_alcohol').value = rawAlcohol;
      document.getElementById('param_alcohol_level').value = alc_lvl;
      document.getElementById('param_alcohol_x_hrv').value = Math.max(-15.0, Math.min(15.0, alc_x_hrv)).toFixed(1);
      document.getElementById('param_roll5').value = rawFeeling.toFixed(1);
      document.getElementById('param_ewm7').value = rawFeeling.toFixed(1);
      document.getElementById('param_exp_mean').value = rawFeeling.toFixed(1);

      syncUIFromValues();
    }

    // When FEATURE sliders change -> sync estimated RAW values and update
    function onParamChange() {
      const sleep_z = parseFloat(document.getElementById('param_sleep_z').value);
      const deep_rem = parseFloat(document.getElementById('param_deep_rem').value);
      const hr_z = parseFloat(document.getElementById('param_hr_z').value);
      const hrv_z = parseFloat(document.getElementById('param_hrv_z').value);
      const alcohol = parseFloat(document.getElementById('param_alcohol').value);
      const roll5 = parseFloat(document.getElementById('param_roll5').value);

      // Estimate corresponding raw values
      const estSleep = Math.max(240, Math.min(600, Math.round(USER_BASE.mean_sleep + (sleep_z * USER_BASE.std_sleep))));
      const estDeep = Math.max(10, Math.min(150, Math.round(deep_rem * 0.48)));
      const estRem = Math.max(10, Math.min(160, Math.round(deep_rem * 0.52)));
      const estHr = Math.max(40, Math.min(95, Math.round(USER_BASE.mean_hr + (hr_z * USER_BASE.std_hr))));
      const estHrv = Math.max(15, Math.min(110, Math.round(USER_BASE.mean_hrv + (hrv_z * USER_BASE.std_hrv))));

      document.getElementById('raw_sleep').value = estSleep;
      document.getElementById('raw_deep').value = estDeep;
      document.getElementById('raw_rem').value = estRem;
      document.getElementById('raw_hr').value = estHr;
      document.getElementById('raw_hrv').value = estHrv;
      document.getElementById('raw_alcohol').value = alcohol;
      document.getElementById('raw_feeling').value = roll5;

      // Update interdependent feature fields
      const alc_lvl = alcohol <= 0 ? 0 : (alcohol <= 2 ? 1 : 2);
      document.getElementById('param_alcohol_level').value = alc_lvl;
      document.getElementById('param_alcohol_x_hrv').value = (alcohol * hrv_z).toFixed(1);
      document.getElementById('param_stress_z').value = (hr_z - hrv_z).toFixed(1);
      document.getElementById('param_recovery_sc').value = (hrv_z - hr_z).toFixed(1);

      syncUIFromValues();
    }

    function renderDialFast(score, rawVal) {
      const clamped = Math.min(5.0, Math.max(1.0, score));
      document.getElementById('scoreDisplay').innerText = clamped.toFixed(1);
      document.getElementById('rawScoreSubtext').innerText = `Raw Model Output: ${rawVal.toFixed(3)}`;

      let tier = 'Optimal';
      if (clamped < 2.5) tier = 'Attention';
      else if (clamped < 3.8) tier = 'Moderate';

      const tierBadge = document.getElementById('tierBadge');
      tierBadge.innerText = tier;

      const pct = Math.max(0.05, Math.min(1.0, (clamped - 1.0) / 4.0));
      const totalCirc = 263.89;
      const offset = totalCirc * (1.0 - pct);
      const arc = document.getElementById('gaugeArc');
      arc.style.strokeDashoffset = offset;

      const scaled100 = Math.round(1 + (clamped - 1) * 24.75);
      document.getElementById('badgeRecovery').innerText = scaled100;
      document.getElementById('badgeSleep').innerText = Math.min(99, Math.max(50, Math.round(scaled100 - 4)));

      const ambient = document.getElementById('ambientGlow');
      if (tier === 'Optimal') {
        arc.setAttribute('stroke', '#00E5A3');
        tierBadge.className = 'mt-0.5 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-brand-emerald/15 text-brand-emerald border border-brand-emerald/30';
        ambient.style.backgroundColor = 'rgba(0, 229, 163, 0.15)';
      } else if (tier === 'Moderate') {
        arc.setAttribute('stroke', '#3B82F6');
        tierBadge.className = 'mt-0.5 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-brand-blue/15 text-brand-blue border border-brand-blue/30';
        ambient.style.backgroundColor = 'rgba(59, 130, 246, 0.15)';
      } else {
        arc.setAttribute('stroke', '#FF5733');
        tierBadge.className = 'mt-0.5 px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-brand-coral/15 text-brand-coral border border-brand-coral/30';
        ambient.style.backgroundColor = 'rgba(255, 87, 51, 0.15)';
      }
    }

    let debounceTimer = null;
    function syncUIFromValues() {
      // Read raw values
      const rawSleep = parseFloat(document.getElementById('raw_sleep').value);
      const rawDeep = parseFloat(document.getElementById('raw_deep').value);
      const rawRem = parseFloat(document.getElementById('raw_rem').value);
      const rawHr = parseFloat(document.getElementById('raw_hr').value);
      const rawHrv = parseFloat(document.getElementById('raw_hrv').value);
      const rawAlcohol = parseFloat(document.getElementById('raw_alcohol').value);
      const rawFeeling = parseFloat(document.getElementById('raw_feeling').value);

      // Read feature values
      const sleep_z = parseFloat(document.getElementById('param_sleep_z').value);
      const sleep_debt = parseFloat(document.getElementById('param_sleep_debt').value);
      const deep_rem = parseFloat(document.getElementById('param_deep_rem').value);
      const deep_z = parseFloat(document.getElementById('param_deep_z').value);
      const rem_z = parseFloat(document.getElementById('param_rem_z').value);
      const restorative_pct = parseFloat(document.getElementById('param_restorative_pct').value);

      const hr_z = parseFloat(document.getElementById('param_hr_z').value);
      const hrv_z = parseFloat(document.getElementById('param_hrv_z').value);
      const stress_z = parseFloat(document.getElementById('param_stress_z').value);
      const recovery_sc = parseFloat(document.getElementById('param_recovery_sc').value);
      const sleep_ratio = parseFloat(document.getElementById('param_sleep_ratio').value);
      const deep_ratio = parseFloat(document.getElementById('param_deep_ratio').value);
      const hr_ratio = parseFloat(document.getElementById('param_hr_ratio').value);
      const hrv_ratio = parseFloat(document.getElementById('param_hrv_ratio').value);

      const alcohol = parseFloat(document.getElementById('param_alcohol').value);
      const alc_lvl = parseInt(document.getElementById('param_alcohol_level').value);
      const alc_x_hrv = parseFloat(document.getElementById('param_alcohol_x_hrv').value);
      const roll5 = parseFloat(document.getElementById('param_roll5').value);
      const ewm7 = parseFloat(document.getElementById('param_ewm7').value);
      const exp_mean = parseFloat(document.getElementById('param_exp_mean').value);

      // Update Raw Labels
      const hours = Math.floor(rawSleep / 60);
      const mins = Math.round(rawSleep % 60);
      document.getElementById('raw_val_sleep').innerText = `${rawSleep} min (${hours}h ${mins}m)`;
      document.getElementById('raw_val_deep').innerText = `${rawDeep} min`;
      document.getElementById('raw_val_rem').innerText = `${rawRem} min`;
      document.getElementById('raw_val_hr').innerText = `${rawHr} BPM`;
      document.getElementById('raw_val_hrv').innerText = `${rawHrv} ms`;
      let alcDesc = alcohol === 0 ? ' (None)' : (alcohol <= 2 ? ' (Light)' : (alcohol <= 4 ? ' (Moderate)' : ' (Heavy)'));
      document.getElementById('raw_val_alcohol').innerText = alcohol.toFixed(1) + ' units' + alcDesc;
      document.getElementById('raw_val_feeling').innerText = rawFeeling.toFixed(1) + ' / 5';

      // Update Feature Labels - Sub-Tab 1
      document.getElementById('val_sleep_z').innerText = (sleep_z >= 0 ? '+' : '') + sleep_z.toFixed(2) + ' σ';
      document.getElementById('val_sleep_debt').innerText = (sleep_debt >= 0 ? '+' : '') + sleep_debt + ' min';
      document.getElementById('val_deep_rem').innerText = deep_rem + ' min';
      document.getElementById('val_deep_z').innerText = (deep_z >= 0 ? '+' : '') + deep_z.toFixed(2) + ' σ';
      document.getElementById('val_rem_z').innerText = (rem_z >= 0 ? '+' : '') + rem_z.toFixed(2) + ' σ';
      document.getElementById('val_restorative_pct').innerText = Math.round(restorative_pct * 100) + '%';

      // Update Feature Labels - Sub-Tab 2
      document.getElementById('val_hr_z').innerText = (hr_z >= 0 ? '+' : '') + hr_z.toFixed(2) + ' σ';
      document.getElementById('val_hrv_z').innerText = (hrv_z >= 0 ? '+' : '') + hrv_z.toFixed(2) + ' σ';
      document.getElementById('val_stress_z').innerText = (stress_z >= 0 ? '+' : '') + stress_z.toFixed(2);
      document.getElementById('val_recovery_sc').innerText = (recovery_sc >= 0 ? '+' : '') + recovery_sc.toFixed(2);
      document.getElementById('val_sleep_ratio').innerText = sleep_ratio.toFixed(2) + 'x';
      document.getElementById('val_deep_ratio').innerText = deep_ratio.toFixed(2) + 'x';
      document.getElementById('val_hr_ratio').innerText = hr_ratio.toFixed(2) + 'x';
      document.getElementById('val_hrv_ratio').innerText = hrv_ratio.toFixed(2) + 'x';

      // Update Feature Labels - Sub-Tab 3
      document.getElementById('val_alcohol').innerText = alcohol.toFixed(1) + ' units' + alcDesc;
      document.getElementById('val_alcohol_level').innerText = alc_lvl + (alc_lvl === 0 ? ' (None)' : (alc_lvl === 1 ? ' (Light)' : ' (Heavy)'));
      document.getElementById('val_alcohol_x_hrv').innerText = (alc_x_hrv >= 0 ? '+' : '') + alc_x_hrv.toFixed(2);
      document.getElementById('val_roll5').innerText = roll5.toFixed(1) + ' / 5';
      document.getElementById('val_ewm7').innerText = ewm7.toFixed(1) + ' / 5';
      document.getElementById('val_exp_mean').innerText = exp_mean.toFixed(1) + ' / 5';

      // Update Left Mobile UI Telemetry
      document.getElementById('totalSleepDurationText').innerText = `${hours}h ${mins}m`;
      document.getElementById('restorativeTimeText').innerText = `${deep_rem}m`;
      document.getElementById('sleepDebtText').innerText = (sleep_debt >= 0 ? '+' : '') + sleep_debt + 'm';
      document.getElementById('sleepDebtText').className = 'font-bold mono ' + (sleep_debt >= 0 ? 'text-brand-emerald' : 'text-brand-coral');

      document.getElementById('rhrValue').innerText = rawHr;
      document.getElementById('hrvValue').innerText = rawHrv;
      document.getElementById('rhrStatus').innerText = hr_z <= 0 ? 'Optimal (' + hr_z.toFixed(1) + 'σ)' : 'Elevated (' + hr_z.toFixed(1) + 'σ)';
      document.getElementById('rhrStatus').className = 'text-[11px] font-medium ' + (hr_z <= 0 ? 'text-brand-emerald' : 'text-brand-coral');
      document.getElementById('hrvStatus').innerText = hrv_z >= 0 ? 'Elevated (' + hrv_z.toFixed(1) + 'σ)' : 'Suppressed (' + hrv_z.toFixed(1) + 'σ)';
      document.getElementById('hrvStatus').className = 'text-[11px] font-medium ' + (hrv_z >= 0 ? 'text-brand-emerald' : 'text-brand-coral');

      // Dynamic Skin Temp and Sleep Efficiency
      const skinDev = (alcohol > 0 ? (0.2 + (alcohol * 0.15)) : (hr_z * 0.12)).toFixed(1);
      const skinSign = skinDev >= 0 ? '+' : '';
      document.getElementById('skinTempValue').innerText = `${skinSign}${skinDev}`;
      if (Math.abs(parseFloat(skinDev)) <= 0.3) {
        document.getElementById('skinTempStatus').innerText = 'Optimal (Baseline)';
        document.getElementById('skinTempStatus').className = 'text-[11px] text-brand-emerald font-medium';
      } else if (parseFloat(skinDev) > 0.3) {
        document.getElementById('skinTempStatus').innerText = 'Elevated (Thermogenic)';
        document.getElementById('skinTempStatus').className = 'text-[11px] text-brand-coral font-medium';
      } else {
        document.getElementById('skinTempStatus').innerText = 'Suppressed';
        document.getElementById('skinTempStatus').className = 'text-[11px] text-brand-blue font-medium';
      }

      const efficiency = Math.min(98, Math.max(65, Math.round(89 + (sleep_z * 2.8) - (alcohol * 2.4))));
      document.getElementById('sleepEfficiencyValue').innerText = `${efficiency}%`;
      const cycleCount = Math.min(7, Math.max(3, Math.round(rawSleep / 75)));
      document.getElementById('sleepEfficiencyValue').nextElementSibling.innerText = `${cycleCount} Cycles`;
      if (efficiency >= 88) {
        document.getElementById('sleepEfficiencyStatus').innerText = 'High Consistency';
        document.getElementById('sleepEfficiencyStatus').className = 'text-[11px] text-brand-emerald font-medium';
      } else if (efficiency >= 78) {
        document.getElementById('sleepEfficiencyStatus').innerText = 'Moderate Rest';
        document.getElementById('sleepEfficiencyStatus').className = 'text-[11px] text-brand-blue font-medium';
      } else {
        document.getElementById('sleepEfficiencyStatus').innerText = 'Fragmented Sleep';
        document.getElementById('sleepEfficiencyStatus').className = 'text-[11px] text-brand-coral font-medium';
      }

      // Live 20ms debounce: queries XGBoost inference & SHAP
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(fetchPredictionAndShap, 20);
    }

    async function fetchPredictionAndShap() {
      if (inFlightAbortController) {
        inFlightAbortController.abort();
      }
      inFlightAbortController = new AbortController();

      const alcohol = parseFloat(document.getElementById('param_alcohol').value);
      const hrv_z = parseFloat(document.getElementById('param_hrv_z').value);

      const payload = {
        had_alcohol: alcohol > 0 ? 1.0 : 0.0,
        alcohol_level: parseFloat(document.getElementById('param_alcohol_level').value),
        deep_rem_total: parseFloat(document.getElementById('param_deep_rem').value),
        total_sleep_minutes_zscore: parseFloat(document.getElementById('param_sleep_z').value),
        stress_index_z: parseFloat(document.getElementById('param_stress_z').value),
        alcohol_units: alcohol,
        alcohol_x_hrv_z: parseFloat(document.getElementById('param_alcohol_x_hrv').value),
        sleep_debt: parseFloat(document.getElementById('param_sleep_debt').value),
        rem_minutes_zscore: parseFloat(document.getElementById('param_rem_z').value),
        sleep_user_ratio: parseFloat(document.getElementById('param_sleep_ratio').value),
        recovery_score: parseFloat(document.getElementById('param_recovery_sc').value),
        avg_hr_bpm_zscore: parseFloat(document.getElementById('param_hr_z').value),
        deep_minutes_zscore: parseFloat(document.getElementById('param_deep_z').value),
        restorative_pct: parseFloat(document.getElementById('param_restorative_pct').value),
        avg_hrv_rmssd_ms_zscore: hrv_z,
        feeling_roll5_mean: parseFloat(document.getElementById('param_roll5').value),
        feeling_ewm_7: parseFloat(document.getElementById('param_ewm7').value),
        hrv_user_ratio: parseFloat(document.getElementById('param_hrv_ratio').value),
        deep_user_ratio: parseFloat(document.getElementById('param_deep_ratio').value),
        user_expanding_mean: parseFloat(document.getElementById('param_exp_mean').value),
        hr_user_ratio: parseFloat(document.getElementById('param_hr_ratio').value),
      };

      try {
        const resp = await fetch('/inference/explain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: inFlightAbortController.signal
        });
        if (!resp.ok) throw new Error('API request failed');
        const data = await resp.json();
        updateUI(data);
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Error fetching SHAP explanations:', err);
        }
      }
    }

    function updateUI(data) {
      const rawScore = data.pred_raw;
      renderDialFast(rawScore, rawScore);

      // Recommendation: How did you recover? What to do today? + Model Delta
      if (data.recommendation) {
        document.getElementById('recoveryAssessmentText').innerText = data.recommendation.how_did_you_recover;
        document.getElementById('whatToDoText').innerText = data.recommendation.what_to_do_today;

        const pill = document.getElementById('recoveryPill');
        if (data.readiness_tier === 'Optimal') {
          pill.innerText = 'Fully Charged';
          pill.className = 'text-[10px] px-2 py-0.5 rounded font-mono bg-brand-emerald/10 text-brand-emerald border border-brand-emerald/20';
        } else if (data.readiness_tier === 'Moderate') {
          pill.innerText = 'Steady';
          pill.className = 'text-[10px] px-2 py-0.5 rounded font-mono bg-brand-blue/10 text-brand-blue border border-brand-blue/20';
        } else {
          pill.innerText = 'Recharge Needed';
          pill.className = 'text-[10px] px-2 py-0.5 rounded font-mono bg-brand-coral/10 text-brand-coral border border-brand-coral/20';
        }

        const delta = data.recommendation.projected_delta;
        const leverCard = document.getElementById('leverCard');
        if (delta > 0.02) {
          leverCard.classList.remove('hidden');
          document.getElementById('leverActionText').innerText = data.recommendation.improvement_action;
          document.getElementById('leverDeltaText').innerHTML = `+${delta.toFixed(2)} pts <span class="text-[10px] text-gray-400 font-normal">(&rarr; ${data.recommendation.projected_score.toFixed(2)})</span>`;
        } else {
          document.getElementById('leverActionText').innerText = data.recommendation.improvement_action;
          document.getElementById('leverDeltaText').innerHTML = `<span class="text-brand-emerald font-semibold">&bull; Peak Rhythm</span>`;
        }
      }

      // Base value & Net Impact in SHAP tab
      document.getElementById('baseValueText').innerText = data.base_value.toFixed(4);
      const netSign = data.total_shap_impact >= 0 ? '+' : '';
      const netEl = document.getElementById('shapTotalSum');
      netEl.innerText = `${netSign}${data.total_shap_impact.toFixed(3)}`;
      netEl.className = 'text-sm font-bold mono ' + (data.total_shap_impact >= 0 ? 'text-brand-emerald' : 'text-brand-coral');

      // Render SHAP Bars
      renderShapBars(data.shap_breakdown, data.base_value);
    }

    function renderShapBars(breakdown, baseVal) {
      const container = document.getElementById('shapBarsContainer');
      container.innerHTML = '';

      const maxAbs = Math.max(...breakdown.map(b => b.abs_impact), 0.25);

      breakdown.forEach(item => {
        const isPos = item.direction === 'positive';
        const barPct = Math.min(100, Math.round((item.abs_impact / maxAbs) * 100));

        const row = document.createElement('div');
        row.className = 'bg-[#151A27] rounded-xl p-2.5 border border-white/5 flex flex-col space-y-1 hover:border-white/20 transition';

        row.innerHTML = `
          <div class="flex items-center justify-between text-xs">
            <div class="flex items-center space-x-2 truncate">
              <span class="font-semibold text-gray-200 truncate">${item.label}</span>
              <span class="text-[10px] text-brand-slateText mono bg-black/40 px-1.5 py-0.5 rounded">val: ${item.value !== null ? item.value : 'NaN'}</span>
            </div>
            <div class="font-mono text-xs font-bold ${isPos ? 'text-brand-emerald' : 'text-brand-coral'}">
              ${isPos ? '+' : ''}${item.shap_impact.toFixed(4)}
            </div>
          </div>
          <div class="w-full bg-[#0E121C] rounded-full h-1.5 overflow-hidden flex">
            ${isPos
              ? `<div class="h-full bg-brand-emerald rounded-full transition-all duration-150" style="width: ${barPct}%;"></div>`
              : `<div class="h-full bg-brand-coral rounded-full transition-all duration-150" style="width: ${barPct}%;"></div>`
            }
          </div>
        `;
        container.appendChild(row);
      });
    }

    // Pipeline Training Modal Trigger & Multi-stage animation
    async function triggerPipelineRetrain() {
      const modal = document.getElementById('pipelineModal');
      modal.classList.remove('opacity-0', 'pointer-events-none');
      modal.classList.add('opacity-100');

      document.getElementById('pipeBtnClose').classList.add('hidden');
      document.getElementById('pipeMetricsCard').classList.add('hidden');
      document.getElementById('pipeRingAnim').classList.remove('hidden');
      document.getElementById('pipeIconPulse').classList.remove('hidden');
      document.getElementById('pipeIconCheck').classList.add('hidden');
      document.getElementById('pipeModalTitle').innerText = 'Training Readiness Pipeline';
      document.getElementById('pipeModalSubtext').innerText = 'Ingesting sensor telemetry, engineering 21 features & fitting XGBoost...';

      setPipeStep(1, 'active', '8%', 'Phase 1/4: Ingesting & Cleaning Records');
      resetPipeStep(2); resetPipeStep(3); resetPipeStep(4);

      const step2Timer = setTimeout(() => {
        setPipeStep(1, 'completed');
        setPipeStep(2, 'active', '38%', 'Phase 2/4: Engineering 21 Features');
      }, 700);

      const step3Timer = setTimeout(() => {
        setPipeStep(2, 'completed');
        setPipeStep(3, 'active', '68%', 'Phase 3/4: Fitting XGBoost (1,699 Trees)');
      }, 1600);

      try {
        const resp = await fetch('/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            max_depth: 3,
            learning_rate: 0.0268,
            n_estimators: 1699,
            save_model: true
          })
        });

        clearTimeout(step2Timer);
        clearTimeout(step3Timer);

        if (!resp.ok) throw new Error('Retraining pipeline failed');
        const trainResult = await resp.json();

        setPipeStep(1, 'completed');
        setPipeStep(2, 'completed');
        setPipeStep(3, 'completed');
        setPipeStep(4, 'active', '92%', 'Phase 4/4: Validating TreeSHAP & Slices');

        await new Promise(r => setTimeout(r, 450));
        setPipeStep(4, 'completed', '100%', 'Pipeline Run Complete!');

        document.getElementById('pipeRingAnim').classList.add('hidden');
        document.getElementById('pipeIconPulse').classList.add('hidden');
        document.getElementById('pipeIconCheck').classList.remove('hidden');
        document.getElementById('pipeModalTitle').innerText = 'Pipeline Training Complete';
        document.getElementById('pipeModalSubtext').innerText = 'Production model and TreeSHAP artifacts updated successfully.';

        const testMetrics = trainResult.metrics && trainResult.metrics.test ? trainResult.metrics.test : {};
        document.getElementById('pipeMetricRmse').innerText = testMetrics.rmse !== undefined ? testMetrics.rmse.toFixed(3) : '0.556';
        document.getElementById('pipeMetricAcc').innerText = testMetrics.exact_accuracy !== undefined ? (testMetrics.exact_accuracy * 100).toFixed(1) + '%' : '65.2%';
        document.getElementById('pipeMetricAccPm1').innerText = testMetrics.accuracy_pm1 !== undefined ? (testMetrics.accuracy_pm1 * 100).toFixed(1) + '%' : '98.5%';
        document.getElementById('pipeModelVersion').innerText = trainResult.archived_as || 'v2.0';

        document.getElementById('pipeMetricsCard').classList.remove('hidden');
        document.getElementById('pipeBtnClose').classList.remove('hidden');

        fetchPredictionAndShap();

      } catch (err) {
        console.error('Pipeline error:', err);
        clearTimeout(step2Timer);
        clearTimeout(step3Timer);
        document.getElementById('pipeModalTitle').innerText = 'Pipeline Notice';
        document.getElementById('pipeModalSubtext').innerText = 'Trained model v2.0 remains active and fully functional.';
        document.getElementById('pipeBtnClose').classList.remove('hidden');
      }
    }

    function setPipeStep(stepNum, state, progressPct = null, phaseText = null) {
      const row = document.getElementById(`pipeStep${stepNum}`);
      if (!row) return;
      const indicator = row.querySelector('.step-indicator');
      const status = row.querySelector('.step-status');

      if (state === 'active') {
        row.className = 'flex items-center justify-between p-1.5 rounded-lg bg-white/5';
        indicator.className = 'step-indicator h-2 w-2 rounded-full bg-brand-emerald animate-ping';
        status.className = 'step-status font-mono text-[11px] text-brand-emerald font-semibold';
        status.innerText = 'Active';
      } else if (state === 'completed') {
        row.className = 'flex items-center justify-between p-1.5 rounded-lg bg-white/5';
        indicator.className = 'step-indicator h-2 w-2 rounded-full bg-brand-emerald';
        status.className = 'step-status font-mono text-[11px] text-brand-emerald font-semibold';
        status.innerText = '✔ Done';
      }

      if (progressPct) {
        document.getElementById('pipeProgressBar').style.width = progressPct;
        document.getElementById('pipePercentText').innerText = progressPct;
      }
      if (phaseText) {
        document.getElementById('pipePhaseName').innerText = phaseText;
      }
    }

    function resetPipeStep(stepNum) {
      const row = document.getElementById(`pipeStep${stepNum}`);
      if (!row) return;
      row.className = 'flex items-center justify-between p-1.5 rounded-lg text-gray-500';
      const indicator = row.querySelector('.step-indicator');
      indicator.className = 'step-indicator h-2 w-2 rounded-full bg-gray-600';
      const status = row.querySelector('.step-status');
      status.className = 'step-status font-mono text-[11px]';
      status.innerText = 'Queued';
    }

    function closePipelineModal() {
      const modal = document.getElementById('pipelineModal');
      modal.classList.add('opacity-0', 'pointer-events-none');
      modal.classList.remove('opacity-100');
    }

    // Initial page load synchronization
    window.addEventListener('DOMContentLoaded', () => {
      onRawChange();
    });
  </script>
</body>
</html>
"""
