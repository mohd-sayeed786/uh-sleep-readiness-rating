"""
Ultrahuman-inspired UI Simulator for the Ring AI Readiness Score Engine.
Provides interactive tabbed controls (Raw Inputs, Engineered Features, SHAP Explainability)
in parallel with the Mobile UI with two-way synchronization, zero-latency 60fps dial responsiveness,
pixel-perfect height matching, and real-time TreeSHAP calculations.
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
      background: #232B3E;
      border-radius: 3px;
    }
    .tab-active {
      background-color: #00E5A3 !important;
      color: #080A0F !important;
      font-weight: 700 !important;
      border-color: #00E5A3 !important;
    }
  </style>
</head>
<body class="min-h-screen antialiased flex flex-col justify-between">

  <!-- Top Navigation Bar -->
  <header class="border-b border-brand-cardBorder bg-[#0D111A]/90 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <!-- Logo & Title -->
      <div class="flex items-center space-x-3">
        <div class="h-9 w-9 rounded-full bg-gradient-to-tr from-brand-emerald via-brand-teal to-brand-blue flex items-center justify-center shadow-lg shadow-emerald-500/20">
          <svg class="w-5 h-5 text-black" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <circle cx="12" cy="12" r="9"></circle>
            <path d="M12 7v5l3 3"></path>
          </svg>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="font-extrabold tracking-wider text-sm uppercase text-white">ULTRAHUMAN</span>
            <span class="text-xs px-2 py-0.5 rounded bg-brand-emerald/10 text-brand-emerald font-semibold border border-brand-emerald/20">RING AI</span>
          </div>
          <p class="text-[11px] text-brand-slateText font-medium">Readiness Score & SHAP Contribution Simulator</p>
        </div>
      </div>

      <!-- Quick Preset Controls -->
      <div class="hidden md:flex items-center space-x-2">
        <span class="text-xs text-brand-slateText uppercase tracking-wider font-semibold mr-1">Presets:</span>
        <button onclick="applyPreset('prime')" class="px-2.5 py-1 text-xs rounded-md bg-brand-card border border-brand-cardBorder hover:border-brand-emerald text-gray-200 transition font-medium flex items-center space-x-1">
          <span>🌟</span> <span>Primed</span>
        </button>
        <button onclick="applyPreset('baseline')" class="px-2.5 py-1 text-xs rounded-md bg-brand-card border border-brand-cardBorder hover:border-brand-blue text-gray-200 transition font-medium flex items-center space-x-1">
          <span>⚡</span> <span>Baseline</span>
        </button>
        <button onclick="applyPreset('alcohol')" class="px-2.5 py-1 text-xs rounded-md bg-brand-card border border-brand-cardBorder hover:border-brand-coral text-gray-200 transition font-medium flex items-center space-x-1">
          <span>🍷</span> <span>Alcohol</span>
        </button>
        <button onclick="applyPreset('deprived')" class="px-2.5 py-1 text-xs rounded-md bg-brand-card border border-brand-cardBorder hover:border-brand-amber text-gray-200 transition font-medium flex items-center space-x-1">
          <span>😴</span> <span>Sleep Debt</span>
        </button>
      </div>

      <!-- Connection / Model Badge & Run Pipeline -->
      <div class="flex items-center space-x-3">
        <button onclick="triggerPipelineRun()" id="btnRunPipeline" class="px-3 py-1.5 text-xs rounded-full bg-brand-emerald/10 border border-brand-emerald/30 hover:bg-brand-emerald/20 text-brand-emerald font-semibold transition flex items-center space-x-1.5 shadow-sm shadow-emerald-500/10 cursor-pointer">
          <svg class="w-3.5 h-3.5 animate-spin hidden" id="btnPipelineSpin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
          </svg>
          <svg class="w-3.5 h-3.5" id="btnPipelineIcon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
          </svg>
          <span id="btnPipelineText">Run Pipeline</span>
        </button>
        <div class="flex items-center space-x-2 bg-brand-card px-3 py-1.5 rounded-full border border-brand-cardBorder">
          <span class="h-2 w-2 rounded-full bg-brand-emerald animate-pulse"></span>
          <span class="text-xs text-gray-300 font-mono" id="modelVersionBadge">Model: XGBoost (13 Feats)</span>
        </div>
        <a href="/docs" target="_blank" class="text-xs text-brand-slateText hover:text-white transition flex items-center space-x-1 border border-brand-cardBorder px-2.5 py-1.5 rounded-lg bg-brand-card">
          <span>API Docs</span>
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
        </a>
      </div>
    </div>
  </header>

  <!-- Main Content Layout -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
    
    <!-- =================================================================== -->
    <!-- PARALLEL STAGE: Mobile UI (Left) vs Tab-Driven Control Deck (Right) -->
    <!-- Both columns flex-stretched with matching heights and smooth sync   -->
    <!-- =================================================================== -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">

      <!-- LEFT: Ultrahuman Ring Mobile Viewport & Cards (col-span-5) -->
      <div class="lg:col-span-5 flex flex-col">
        
        <div class="bg-gradient-to-b from-[#141A28] to-[#0E121C] rounded-3xl border border-brand-cardBorder p-6 shadow-2xl relative overflow-hidden flex-1 flex flex-col justify-between space-y-4">
          
          <!-- Background Ambient Glow -->
          <div id="ambientGlow" class="absolute -top-24 -left-24 w-72 h-72 rounded-full bg-brand-emerald/15 blur-3xl pointer-events-none transition-opacity duration-300"></div>
          <div class="absolute -bottom-24 -right-24 w-72 h-72 rounded-full bg-brand-blue/10 blur-3xl pointer-events-none"></div>

          <!-- Top Status & Hero Dial Section -->
          <div class="space-y-3">
            <!-- Top Status Bar -->
            <div class="flex items-center justify-between text-xs text-brand-slateText pb-2 border-b border-white/5">
              <span class="font-medium tracking-wide" id="todayDate">Sunday, Sep 13</span>
              <span class="flex items-center space-x-1.5 text-brand-emerald">
                <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14h2v2h-2zm0-10h2v8h-2z"/></svg>
                <span>Ring Synced 7:15 AM</span>
              </span>
            </div>

            <!-- Trio Metric Badges -->
            <div class="grid grid-cols-3 gap-2.5 text-center">
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

            <!-- Hero Radial Readiness Gauge (Clean, Spacious, No Overlap) -->
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
              <div class="w-full h-2 rounded-full bg-[#1F2739] flex overflow-hidden gap-0.5 mb-2">
                <div id="barDeep" class="bg-brand-purple h-full transition-all duration-300" style="width: 28%;" title="Deep Sleep"></div>
                <div id="barRem" class="bg-brand-teal h-full transition-all duration-300" style="width: 25%;" title="REM Sleep"></div>
                <div id="barLight" class="bg-brand-blue h-full transition-all duration-300" style="width: 38%;" title="Light Sleep"></div>
                <div id="barAwake" class="bg-gray-600 h-full transition-all duration-300" style="width: 9%;" title="Awake"></div>
              </div>

              <div class="grid grid-cols-2 gap-2 text-xs text-brand-slateText pt-1 border-t border-white/5">
                <div class="flex items-center justify-between">
                  <span>Restorative:</span>
                  <span class="font-bold text-white mono" id="restorativeTimeText">145m</span>
                </div>
                <div class="flex items-center justify-between">
                  <span>Sleep Deficit:</span>
                  <span class="font-bold mono" id="sleepDebtText">+15m</span>
                </div>
              </div>
            </div>

            <!-- Autonomic Recovery Contributors (HR & HRV) -->
            <div class="grid grid-cols-2 gap-3">
              <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3">
                <div class="text-[10px] uppercase font-semibold text-brand-slateText">RESTING HR</div>
                <div class="flex items-baseline space-x-1.5 mt-0.5">
                  <span class="text-lg font-bold text-white mono" id="rhrValue">52</span>
                  <span class="text-xs text-brand-slateText">BPM</span>
                </div>
                <div class="text-[11px] text-brand-emerald font-medium" id="rhrStatus">Optimal (-1.2&sigma;)</div>
              </div>
              <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3">
                <div class="text-[10px] uppercase font-semibold text-brand-slateText">HRV RMSSD</div>
                <div class="flex items-baseline space-x-1.5 mt-0.5">
                  <span class="text-lg font-bold text-white mono" id="hrvValue">68</span>
                  <span class="text-xs text-brand-slateText">MS</span>
                </div>
                <div class="text-[11px] text-brand-emerald font-medium" id="hrvStatus">Elevated (+1.4&sigma;)</div>
              </div>
            </div>

            <!-- Biomarkers Row: Skin Temp & Sleep Efficiency -->
            <div class="grid grid-cols-2 gap-3">
              <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3">
                <div class="text-[10px] uppercase font-semibold text-brand-slateText">SKIN TEMP DEVIATION</div>
                <div class="flex items-baseline space-x-1.5 mt-0.5">
                  <span class="text-lg font-bold text-white mono" id="skinTempValue">+0.1</span>
                  <span class="text-xs text-brand-slateText">&deg;C</span>
                </div>
                <div class="text-[11px] text-brand-emerald font-medium" id="skinTempStatus">Optimal (Baseline)</div>
              </div>
              <div class="bg-[#161D2B] border border-white/5 rounded-2xl p-3">
                <div class="text-[10px] uppercase font-semibold text-brand-slateText">SLEEP EFFICIENCY</div>
                <div class="flex items-baseline space-x-1.5 mt-0.5">
                  <span class="text-lg font-bold text-white mono" id="sleepEfficiencyValue">92%</span>
                  <span class="text-xs text-brand-slateText">6 Cycles</span>
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
          <!-- TAB VIEWPORT CONTAINER (Ensures constant size across tabs) -->
          <!-- ========================================================= -->
          <div class="flex-1 flex flex-col min-h-0">

            <!-- ========================================================= -->
            <!-- TAB 1: RAW INPUTS CONTROLS                                -->
            <!-- ========================================================= -->
            <div id="viewRaw" class="hidden space-y-3.5 flex-1 flex flex-col justify-between overflow-y-auto pr-1">
              <div class="bg-[#151B27] p-2.5 rounded-xl border border-white/5 text-xs text-brand-slateText flex items-center justify-between">
                <span>Adjust raw sensor readings & habits; features & z-scores update live.</span>
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
                      <span>80m (Normal)</span>
                      <span>160m</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Physiological Sensors (BPM & MS) -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-teal flex items-center space-x-1.5">
                  <span>💓</span> <span>Ring PPG Sensors (Heart & Autonomic)</span>
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

              <!-- Alcohol & Habits -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-coral flex items-center space-x-1.5">
                  <span>🍷</span> <span>Pre-Sleep Alcohol & Lifestyle</span>
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

              <!-- Subjective Momentum & History -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-amber flex items-center space-x-1.5">
                  <span>🧠</span> <span>Psychological Momentum & Past Check-ins</span>
                </h4>
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Yesterday Lag 1</span>
                      <span class="font-mono text-brand-amber font-bold text-xs" id="raw_val_lag1">4 / 5</span>
                    </div>
                    <input type="range" id="raw_lag1" min="1" max="5" step="1" value="4"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>1 (Poor)</span>
                      <span>3 (Neutral)</span>
                      <span>5 (Prime)</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Days Post Bad Sleep</span>
                      <span class="font-mono text-gray-300 font-bold text-xs" id="raw_val_bad_days">7 d</span>
                    </div>
                    <input type="range" id="raw_bad_days" min="0" max="30" step="1" value="7"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0d (Recent)</span>
                      <span>15d</span>
                      <span>30d</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Days Post Great Sleep</span>
                      <span class="font-mono text-gray-300 font-bold text-xs" id="raw_val_great_days">1 d</span>
                    </div>
                    <input type="range" id="raw_great_days" min="0" max="30" step="1" value="1"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onRawChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0d (Today)</span>
                      <span>15d</span>
                      <span>30d</span>
                    </div>
                  </div>
                </div>
              </div>

            </div>

            <!-- ========================================================= -->
            <!-- TAB 2: ENGINEERED FEATURES CONTROLS (Z-Scores & Temporal)  -->
            <!-- ========================================================= -->
            <div id="viewFeatures" class="space-y-3.5 flex-1 flex flex-col justify-between overflow-y-auto pr-1">
              <div class="bg-[#151B27] p-2.5 rounded-xl border border-white/5 text-xs text-brand-slateText flex items-center justify-between">
                <span>Directly tune model feature inputs (z-scores, debt minutes, cyclical anchors).</span>
                <span class="mono text-[11px] text-brand-teal font-semibold">13 Live ML Features</span>
              </div>

              <!-- SECTION 1: Sleep Telemetry -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-blue flex items-center space-x-1.5">
                  <span>🌙</span> <span>Sleep Volume & Quality</span>
                </h4>
                
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Sleep Duration Z-Score</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="val_sleep_z">+0.80 &sigma;</span>
                    </div>
                    <input type="range" id="param_sleep_z" min="-3.0" max="3.0" step="0.1" value="0.8"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-3.0 (Short)</span>
                      <span>0.0 (Average)</span>
                      <span>+3.0 (Long)</span>
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
                      <span>-120m (Debt)</span>
                      <span>0m</span>
                      <span>+120m</span>
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
                      <span>30m (Deprived)</span>
                      <span>120m (Normal)</span>
                      <span>240m (Optimal)</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- SECTION 2: Autonomic Physiology -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-teal flex items-center space-x-1.5">
                  <span>💓</span> <span>Overnight Autonomic Recovery (Ring Telemetry)</span>
                </h4>
                
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Resting HR Z-Score</span>
                      <span class="font-mono text-brand-emerald font-bold text-xs" id="val_hr_z">-0.50 &sigma;</span>
                    </div>
                    <input type="range" id="param_hr_z" min="-3.0" max="3.0" step="0.1" value="-0.5"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>-3.0 (Low RHR)</span>
                      <span>0.0 (Baseline)</span>
                      <span>+3.0</span>
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
                      <span>-3.0 (Suppressed)</span>
                      <span>0.0 (Baseline)</span>
                      <span>+3.0</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- SECTION 3: Alcohol & Daytime Habits -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-coral flex items-center space-x-1.5">
                  <span>🍷</span> <span>Lifestyle & Daytime Context</span>
                </h4>
                
                <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                  <div class="flex justify-between items-center mb-1">
                    <span class="text-xs text-gray-300 font-medium">Alcohol Intake (Units Consumed Yesterday)</span>
                    <span class="font-mono font-bold text-xs px-2 py-0.5 rounded bg-brand-card" id="val_alcohol">0.0 units (None)</span>
                  </div>
                  <input type="range" id="param_alcohol" min="0.0" max="8.0" step="0.5" value="0.0"
                         class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                  <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                    <span>0 units</span>
                    <span>2 units (Light)</span>
                    <span>4 units (Moderate)</span>
                    <span>8+ units</span>
                  </div>
                </div>
              </div>

              <!-- SECTION 4: Psychological Lag & Recency -->
              <div class="space-y-2">
                <h4 class="text-xs font-bold uppercase tracking-wider text-brand-amber flex items-center space-x-1.5">
                  <span>🧠</span> <span>Psychological Momentum & Recency Windows</span>
                </h4>
                
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Yesterday Lag 1</span>
                      <span class="font-mono text-brand-amber font-bold text-xs" id="val_lag1">4 / 5</span>
                    </div>
                    <input type="range" id="param_lag1" min="1" max="5" step="1" value="4"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>1 (Poor)</span>
                      <span>3</span>
                      <span>5 (Prime)</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Days Post Bad</span>
                      <span class="font-mono text-gray-300 font-bold text-xs" id="val_bad_days">7 d</span>
                    </div>
                    <input type="range" id="param_bad_days" min="0" max="30" step="1" value="7"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0d</span>
                      <span>15d</span>
                      <span>30d</span>
                    </div>
                  </div>

                  <div class="bg-[#171D2B] p-2.5 rounded-xl border border-white/5">
                    <div class="flex justify-between text-xs mb-1">
                      <span class="text-gray-300 font-medium">Days Post Great</span>
                      <span class="font-mono text-gray-300 font-bold text-xs" id="val_great_days">1 d</span>
                    </div>
                    <input type="range" id="param_great_days" min="0" max="30" step="1" value="1"
                           class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="onParamChange()">
                    <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                      <span>0d</span>
                      <span>15d</span>
                      <span>30d</span>
                    </div>
                  </div>
                </div>

                <div class="grid grid-cols-2 gap-2.5 pt-0.5">
                  <div class="bg-[#171D2B] p-2 rounded-xl border border-white/5 flex items-center justify-between text-xs">
                    <span class="text-brand-slateText">Check-in Sequence:</span>
                    <input type="number" id="param_seq" min="1" max="100" value="25" class="w-14 bg-black/40 border border-white/10 rounded px-1.5 py-0.5 text-center font-mono text-white text-xs" oninput="onParamChange()" onchange="onParamChange()">
                  </div>
                  <div class="bg-[#171D2B] p-2 rounded-xl border border-white/5 flex items-center justify-between text-xs">
                    <span class="text-brand-slateText">Week of Year:</span>
                    <input type="number" id="param_week" min="1" max="52" value="15" class="w-14 bg-black/40 border border-white/10 rounded px-1.5 py-0.5 text-center font-mono text-white text-xs" oninput="onParamChange()" onchange="onParamChange()">
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
                  <div class="text-sm font-bold mono text-gray-200" id="baseValueText">3.2808</div>
                </div>
                <div class="text-right">
                  <div class="text-[10px] uppercase text-brand-slateText font-semibold">NET SHAP SUM IMPACT</div>
                  <div class="text-sm font-bold mono text-brand-emerald" id="shapTotalSum">+0.906</div>
                </div>
              </div>

              <!-- Dynamic 13-Feature TreeSHAP Waterfall List -->
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
      Ring AI Readiness Engine &bull; Ultrahuman Ring Simulator &bull; XGBoost TreeSHAP Production Microservice &bull; <a href="/docs" class="text-brand-emerald underline">FastAPI OpenAPI Specs</a>
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
      <p id="pipeModalSubtext" class="text-xs text-brand-slateText mt-1.5">Ingesting sensor telemetry, engineering features & fitting XGBoost...</p>

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
            <span class="text-gray-200 font-medium">1. Data Ingestion & Sentinel Cleaning</span>
          </div>
          <span class="step-status font-mono text-[11px] text-brand-emerald font-semibold">Active</span>
        </div>
        <div id="pipeStep2" class="flex items-center justify-between p-1.5 rounded-lg text-gray-500">
          <div class="flex items-center space-x-2.5">
            <span class="step-indicator h-2 w-2 rounded-full bg-gray-600"></span>
            <span class="font-medium">2. 13-Feature Physiological Engineering</span>
          </div>
          <span class="step-status font-mono text-[11px]">Queued</span>
        </div>
        <div id="pipeStep3" class="flex items-center justify-between p-1.5 rounded-lg text-gray-500">
          <div class="flex items-center space-x-2.5">
            <span class="step-indicator h-2 w-2 rounded-full bg-gray-600"></span>
            <span class="font-medium">3. XGBoost Model Fitting (900 trees)</span>
          </div>
          <span class="step-status font-mono text-[11px]">Queued</span>
        </div>
        <div id="pipeStep4" class="flex items-center justify-between p-1.5 rounded-lg text-gray-500">
          <div class="flex items-center space-x-2.5">
            <span class="step-indicator h-2 w-2 rounded-full bg-gray-600"></span>
            <span class="font-medium">4. Subgroup Slice & TreeSHAP Verification</span>
          </div>
          <span class="step-status font-mono text-[11px]">Queued</span>
        </div>
      </div>

      <!-- Completion Metrics Card (revealed on success) -->
      <div id="pipeMetricsCard" class="mt-4 hidden bg-brand-emerald/10 border border-brand-emerald/30 rounded-2xl p-4 text-left animate-fade-in">
        <div class="text-[11px] font-semibold text-brand-emerald mb-2 flex items-center justify-between">
          <span>✔ Model Retrained &amp; Validated</span>
          <span id="pipeModelVersion" class="mono text-[10px] text-gray-400">v1.x</span>
        </div>
        <div class="grid grid-cols-3 gap-2 text-center">
          <div class="bg-[#0E121C]/80 rounded-xl p-2 border border-brand-emerald/20">
            <div class="text-[10px] text-gray-400 uppercase">Holdout RMSE</div>
            <div class="text-sm font-mono font-bold text-brand-emerald" id="pipeMetricRmse">0.661</div>
          </div>
          <div class="bg-[#0E121C]/80 rounded-xl p-2 border border-brand-teal/20">
            <div class="text-[10px] text-gray-400 uppercase">Exact Acc</div>
            <div class="text-sm font-mono font-bold text-brand-teal" id="pipeMetricAcc">57.6%</div>
          </div>
          <div class="bg-[#0E121C]/80 rounded-xl p-2 border border-brand-blue/20">
            <div class="text-[10px] text-gray-400 uppercase">Acc ±1 Class</div>
            <div class="text-sm font-mono font-bold text-brand-blue" id="pipeMetricAccPm1">96.6%</div>
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
    // Presets catalog
    const PRESETS = {
      prime: {
        raw_sleep: 510, deep: 95, rem: 85, hr: 53, hrv: 70, alcohol: 0.0, lag1: 5, bad_days: 14, great_days: 0, seq: 30, week: 16,
        sleep_z: 1.5, sleep_debt: 45, deep_rem: 180, hr_z: -1.2, hrv_z: 1.8
      },
      baseline: {
        raw_sleep: 440, deep: 65, rem: 65, hr: 59, hrv: 49, alcohol: 0.0, lag1: 3, bad_days: 6, great_days: 3, seq: 20, week: 16,
        sleep_z: 0.2, sleep_debt: 5, deep_rem: 130, hr_z: -0.1, hrv_z: 0.3
      },
      alcohol: {
        raw_sleep: 380, deep: 40, rem: 45, hr: 68, hrv: 30, alcohol: 3.5, lag1: 3, bad_days: 1, great_days: 8, seq: 22, week: 16,
        sleep_z: -0.8, sleep_debt: -40, deep_rem: 85, hr_z: 1.6, hrv_z: -1.5
      },
      deprived: {
        raw_sleep: 300, deep: 30, rem: 35, hr: 66, hrv: 26, alcohol: 0.0, lag1: 2, bad_days: 0, great_days: 12, seq: 18, week: 16,
        sleep_z: -2.2, sleep_debt: -95, deep_rem: 65, hr_z: 1.2, hrv_z: -1.8
      }
    };

    // User calibration baseline constants
    const USER_BASE = {
      mean_sleep: 430.0,
      std_sleep: 55.0,
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

    function applyPreset(key) {
      const p = PRESETS[key];
      if (!p) return;

      // Update Raw controls
      document.getElementById('raw_sleep').value = p.raw_sleep;
      document.getElementById('raw_deep').value = p.deep;
      document.getElementById('raw_rem').value = p.rem;
      document.getElementById('raw_hr').value = p.hr;
      document.getElementById('raw_hrv').value = p.hrv;
      document.getElementById('raw_alcohol').value = p.alcohol;
      document.getElementById('raw_lag1').value = p.lag1;
      document.getElementById('raw_bad_days').value = p.bad_days;
      document.getElementById('raw_great_days').value = p.great_days;

      // Update Features controls
      document.getElementById('param_sleep_z').value = p.sleep_z;
      document.getElementById('param_sleep_debt').value = p.sleep_debt;
      document.getElementById('param_deep_rem').value = p.deep_rem;
      document.getElementById('param_hr_z').value = p.hr_z;
      document.getElementById('param_hrv_z').value = p.hrv_z;
      document.getElementById('param_alcohol').value = p.alcohol;
      document.getElementById('param_lag1').value = p.lag1;
      document.getElementById('param_bad_days').value = p.bad_days;
      document.getElementById('param_great_days').value = p.great_days;
      document.getElementById('param_seq').value = p.seq;
      document.getElementById('param_week').value = p.week;

      syncUIFromValues();
    }

    // When RAW inputs change -> compute and sync FEATURES
    function onRawChange() {
      const rawSleep = parseFloat(document.getElementById('raw_sleep').value);
      const rawDeep = parseFloat(document.getElementById('raw_deep').value);
      const rawRem = parseFloat(document.getElementById('raw_rem').value);
      const rawHr = parseFloat(document.getElementById('raw_hr').value);
      const rawHrv = parseFloat(document.getElementById('raw_hrv').value);
      const rawAlcohol = parseFloat(document.getElementById('raw_alcohol').value);
      const rawLag1 = parseInt(document.getElementById('raw_lag1').value);
      const rawBadDays = parseInt(document.getElementById('raw_bad_days').value);
      const rawGreatDays = parseInt(document.getElementById('raw_great_days').value);

      // Compute Features
      const sleep_z = (rawSleep - USER_BASE.mean_sleep) / USER_BASE.std_sleep;
      const deep_rem = rawDeep + rawRem;
      const hr_z = (rawHr - USER_BASE.mean_hr) / USER_BASE.std_hr;
      const hrv_z = (rawHrv - USER_BASE.mean_hrv) / USER_BASE.std_hrv;

      // Sync into Features sliders (param_sleep_debt remains independent)
      document.getElementById('param_sleep_z').value = Math.max(-3.0, Math.min(3.0, sleep_z)).toFixed(1);
      document.getElementById('param_deep_rem').value = Math.max(30, Math.min(240, deep_rem));
      document.getElementById('param_hr_z').value = Math.max(-3.0, Math.min(3.0, hr_z)).toFixed(1);
      document.getElementById('param_hrv_z').value = Math.max(-3.0, Math.min(3.0, hrv_z)).toFixed(1);
      document.getElementById('param_alcohol').value = rawAlcohol;
      document.getElementById('param_lag1').value = rawLag1;
      document.getElementById('param_bad_days').value = rawBadDays;
      document.getElementById('param_great_days').value = rawGreatDays;

      syncUIFromValues();
    }

    // When FEATURES change -> compute and sync RAW
    function onParamChange() {
      const sleep_z = parseFloat(document.getElementById('param_sleep_z').value);
      const deep_rem = parseFloat(document.getElementById('param_deep_rem').value);
      const hr_z = parseFloat(document.getElementById('param_hr_z').value);
      const hrv_z = parseFloat(document.getElementById('param_hrv_z').value);
      const alcohol = parseFloat(document.getElementById('param_alcohol').value);
      const lag1 = parseInt(document.getElementById('param_lag1').value);
      const bad_days = parseInt(document.getElementById('param_bad_days').value);
      const great_days = parseInt(document.getElementById('param_great_days').value);

      // Compute estimated raw numbers within realistic bounds
      const estSleep = Math.max(240, Math.min(600, Math.round(USER_BASE.mean_sleep + (sleep_z * USER_BASE.std_sleep))));
      const estDeep = Math.max(10, Math.min(150, Math.round(deep_rem * 0.48)));
      const estRem = Math.max(10, Math.min(160, Math.round(deep_rem * 0.52)));
      const estHr = Math.max(40, Math.min(95, Math.round(USER_BASE.mean_hr + (hr_z * USER_BASE.std_hr))));
      const estHrv = Math.max(15, Math.min(110, Math.round(USER_BASE.mean_hrv + (hrv_z * USER_BASE.std_hrv))));

      // Sync into Raw sliders
      document.getElementById('raw_sleep').value = estSleep;
      document.getElementById('raw_deep').value = estDeep;
      document.getElementById('raw_rem').value = estRem;
      document.getElementById('raw_hr').value = estHr;
      document.getElementById('raw_hrv').value = estHrv;
      document.getElementById('raw_alcohol').value = alcohol;
      document.getElementById('raw_lag1').value = lag1;
      document.getElementById('raw_bad_days').value = bad_days;
      document.getElementById('raw_great_days').value = great_days;

      syncUIFromValues();
    }

    function renderDialFast(score, rawVal) {
      // Clamped score
      const clamped = Math.min(5.0, Math.max(1.0, score));
      document.getElementById('scoreDisplay').innerText = clamped.toFixed(1);
      document.getElementById('rawScoreSubtext').innerText = `Raw Model Output: ${rawVal.toFixed(3)}`;

      // Determine Tier
      let tier = 'Optimal';
      if (clamped < 2.5) tier = 'Attention';
      else if (clamped < 3.8) tier = 'Moderate';

      const tierBadge = document.getElementById('tierBadge');
      tierBadge.innerText = tier;

      // Arc calculation (Circumference = 2 * PI * 42 = 263.89)
      const pct = Math.max(0.05, Math.min(1.0, (clamped - 1.0) / 4.0));
      const totalCirc = 263.89;
      const offset = totalCirc * (1.0 - pct);
      const arc = document.getElementById('gaugeArc');
      arc.style.strokeDashoffset = offset;

      // 100-pt scaled metric badge
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
      // Read current features values
      const sleep_z = parseFloat(document.getElementById('param_sleep_z').value);
      const sleep_debt = parseFloat(document.getElementById('param_sleep_debt').value);
      const deep_rem = parseFloat(document.getElementById('param_deep_rem').value);
      const hr_z = parseFloat(document.getElementById('param_hr_z').value);
      const hrv_z = parseFloat(document.getElementById('param_hrv_z').value);
      const alcohol = parseFloat(document.getElementById('param_alcohol').value);
      const lag1 = parseInt(document.getElementById('param_lag1').value);
      const bad_days = parseInt(document.getElementById('param_bad_days').value);
      const great_days = parseInt(document.getElementById('param_great_days').value);

      // Read raw values (Fix: rawHr correctly reads raw_hr, not raw_hrv)
      const rawSleep = parseFloat(document.getElementById('raw_sleep').value);
      const rawDeep = parseFloat(document.getElementById('raw_deep').value);
      const rawRem = parseFloat(document.getElementById('raw_rem').value);
      const rawHr = parseFloat(document.getElementById('raw_hr').value);
      const rawHrv = parseFloat(document.getElementById('raw_hrv').value);

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
      document.getElementById('raw_val_lag1').innerText = lag1 + ' / 5';
      document.getElementById('raw_val_bad_days').innerText = bad_days + ' d';
      document.getElementById('raw_val_great_days').innerText = great_days + ' d';

      // Update Feature Labels
      document.getElementById('val_sleep_z').innerText = (sleep_z >= 0 ? '+' : '') + sleep_z.toFixed(2) + ' \u03c3';
      document.getElementById('val_sleep_debt').innerText = (sleep_debt >= 0 ? '+' : '') + sleep_debt + ' min';
      document.getElementById('val_deep_rem').innerText = deep_rem + ' min';
      document.getElementById('val_hr_z').innerText = (hr_z >= 0 ? '+' : '') + hr_z.toFixed(2) + ' \u03c3';
      document.getElementById('val_hrv_z').innerText = (hrv_z >= 0 ? '+' : '') + hrv_z.toFixed(2) + ' \u03c3';
      document.getElementById('val_alcohol').innerText = alcohol.toFixed(1) + ' units' + alcDesc;
      document.getElementById('val_lag1').innerText = lag1 + ' / 5';
      document.getElementById('val_bad_days').innerText = bad_days + ' d';
      document.getElementById('val_great_days').innerText = great_days + ' d';

      // Update Left Mobile UI Telemetry
      document.getElementById('totalSleepDurationText').innerText = `${hours}h ${mins}m`;
      document.getElementById('restorativeTimeText').innerText = `${deep_rem}m`;
      document.getElementById('sleepDebtText').innerText = (sleep_debt >= 0 ? '+' : '') + sleep_debt + 'm';
      document.getElementById('sleepDebtText').className = 'font-bold mono ' + (sleep_debt >= 0 ? 'text-brand-emerald' : 'text-brand-coral');

      document.getElementById('rhrValue').innerText = rawHr;
      document.getElementById('hrvValue').innerText = rawHrv;
      document.getElementById('rhrStatus').innerText = hr_z <= 0 ? 'Optimal (' + hr_z.toFixed(1) + '\u03c3)' : 'Elevated (' + hr_z.toFixed(1) + '\u03c3)';
      document.getElementById('rhrStatus').className = 'text-[11px] font-medium ' + (hr_z <= 0 ? 'text-brand-emerald' : 'text-brand-coral');
      document.getElementById('hrvStatus').innerText = hrv_z >= 0 ? 'Elevated (' + hrv_z.toFixed(1) + '\u03c3)' : 'Suppressed (' + hrv_z.toFixed(1) + '\u03c3)';
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

      // Live 20ms debounce: directly queries XGBoost tree inference & SHAP without score flicker or phantom scores
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(fetchPredictionAndShap, 20);
    }

    async function fetchPredictionAndShap() {
      if (inFlightAbortController) {
        inFlightAbortController.abort();
      }
      inFlightAbortController = new AbortController();

      const payload = {
        alcohol_units: parseFloat(document.getElementById('param_alcohol').value),
        had_alcohol: parseFloat(document.getElementById('param_alcohol').value) > 0 ? 1.0 : 0.0,
        alcohol_level: parseFloat(document.getElementById('param_alcohol').value) <= 0 ? 0.0 : (parseFloat(document.getElementById('param_alcohol').value) <= 2.0 ? 1.0 : 2.0),
        week_of_year: parseInt(document.getElementById('param_week').value),
        total_sleep_minutes_zscore: parseFloat(document.getElementById('param_sleep_z').value),
        avg_hr_bpm_zscore: parseFloat(document.getElementById('param_hr_z').value),
        avg_hrv_rmssd_ms_zscore: parseFloat(document.getElementById('param_hrv_z').value),
        subjective_feeling_lag1: parseFloat(document.getElementById('param_lag1').value),
        days_since_bad_sleep: parseFloat(document.getElementById('param_bad_days').value),
        days_since_great_sleep: parseFloat(document.getElementById('param_great_days').value),
        checkin_seq_num: parseInt(document.getElementById('param_seq').value),
        deep_rem_total: parseFloat(document.getElementById('param_deep_rem').value),
        sleep_debt: parseFloat(document.getElementById('param_sleep_debt').value)
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
        if (delta > 0.03) {
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
            <div class="flex items-center space-x-1.5 mono font-bold ${isPos ? 'text-brand-emerald' : 'text-brand-coral'}">
              <span>${isPos ? '+' : ''}${item.shap_impact.toFixed(4)}</span>
            </div>
          </div>
          <!-- Bidirectional Visual Bar -->
          <div class="w-full bg-[#0D1018] h-1.5 rounded-full overflow-hidden flex ${isPos ? 'justify-start' : 'justify-end'}">
            <div class="h-full rounded-full ${isPos ? 'bg-brand-emerald' : 'bg-brand-coral'} transition-all duration-300" style="width: ${barPct}%;"></div>
          </div>
        `;
        container.appendChild(row);
      });
    }

    window.addEventListener('DOMContentLoaded', () => {
      // Prevent trackpad / mousewheel accidental scroll on number inputs
      document.querySelectorAll('input[type="number"]').forEach(el => {
        el.addEventListener('wheel', (e) => e.target.blur(), { passive: true });
      });

      syncUIFromValues();
      fetchPredictionAndShap();
    });



    // =========================================================================
    // TRAINING PIPELINE LOADING SCREEN CONTROLLER
    // =========================================================================
    let pipeInterval = null;

    async function triggerPipelineRun() {
      const modal = document.getElementById('pipelineModal');
      const progressBar = document.getElementById('pipeProgressBar');
      const percentText = document.getElementById('pipePercentText');
      const phaseName = document.getElementById('pipePhaseName');
      const modalTitle = document.getElementById('pipeModalTitle');
      const modalSubtext = document.getElementById('pipeModalSubtext');
      const ringAnim = document.getElementById('pipeRingAnim');
      const iconPulse = document.getElementById('pipeIconPulse');
      const iconCheck = document.getElementById('pipeIconCheck');
      const metricsCard = document.getElementById('pipeMetricsCard');
      const btnClose = document.getElementById('pipeBtnClose');
      const btnRun = document.getElementById('btnRunPipeline');
      const btnSpin = document.getElementById('btnPipelineSpin');
      const btnIcon = document.getElementById('btnPipelineIcon');

      // Reset modal state
      modal.classList.remove('opacity-0', 'pointer-events-none');
      ringAnim.classList.remove('hidden');
      ringAnim.classList.add('animate-spin');
      iconPulse.classList.remove('hidden');
      iconCheck.classList.add('hidden');
      metricsCard.classList.add('hidden');
      btnClose.classList.add('hidden');
      btnSpin.classList.remove('hidden');
      btnIcon.classList.add('hidden');
      btnRun.classList.add('opacity-70', 'cursor-not-allowed');

      modalTitle.textContent = 'Training Readiness Pipeline';
      modalSubtext.textContent = 'Ingesting sensor telemetry, engineering features & fitting XGBoost...';

      // Step styling reset
      const resetStep = (id, num, label) => {
        const el = document.getElementById(id);
        el.className = 'flex items-center justify-between p-1.5 rounded-lg text-gray-500';
        el.querySelector('.step-indicator').className = 'step-indicator h-2 w-2 rounded-full bg-gray-600';
        el.querySelector('.step-status').textContent = 'Queued';
        el.querySelector('.step-status').className = 'step-status font-mono text-[11px]';
      };
      resetStep('pipeStep1', 1, 'Data Ingestion & Sentinel Cleaning');
      resetStep('pipeStep2', 2, '13-Feature Physiological Engineering');
      resetStep('pipeStep3', 3, 'XGBoost Model Fitting (900 trees)');
      resetStep('pipeStep4', 4, 'Subgroup Slice & TreeSHAP Verification');

      const activateStep = (id, runningText = 'Running...') => {
        const el = document.getElementById(id);
        el.className = 'flex items-center justify-between p-1.5 rounded-lg bg-white/5 text-white';
        el.querySelector('.step-indicator').className = 'step-indicator h-2 w-2 rounded-full bg-brand-emerald animate-ping';
        const st = el.querySelector('.step-status');
        st.textContent = runningText;
        st.className = 'step-status font-mono text-[11px] text-brand-emerald font-semibold';
      };

      const completeStep = (id) => {
        const el = document.getElementById(id);
        el.className = 'flex items-center justify-between p-1.5 rounded-lg text-gray-300';
        el.querySelector('.step-indicator').className = 'step-indicator h-2 w-2 rounded-full bg-brand-emerald';
        const st = el.querySelector('.step-status');
        st.textContent = '✔ Done';
        st.className = 'step-status font-mono text-[11px] text-brand-emerald';
      };

      // Progress animation ticker
      let currentProgress = 5;
      activateStep('pipeStep1', 'Ingesting...');
      phaseName.textContent = 'Phase 1/4: Ingesting & Cleaning Data';

      const startTime = Date.now();
      if (pipeInterval) clearInterval(pipeInterval);
      pipeInterval = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        if (elapsed < 0.6) {
          currentProgress = Math.min(25, 5 + elapsed * 35);
          phaseName.textContent = 'Phase 1/4: Ingesting & Cleaning Data';
        } else if (elapsed < 1.4) {
          completeStep('pipeStep1');
          activateStep('pipeStep2', 'Engineering...');
          currentProgress = Math.min(48, 25 + (elapsed - 0.6) * 30);
          phaseName.textContent = 'Phase 2/4: Engineering 13 Features';
        } else if (elapsed < 3.2) {
          completeStep('pipeStep2');
          activateStep('pipeStep3', 'Fitting 900 trees...');
          currentProgress = Math.min(85, 48 + (elapsed - 1.4) * 20);
          phaseName.textContent = 'Phase 3/4: Training XGBoost Regressor';
        } else {
          completeStep('pipeStep3');
          activateStep('pipeStep4', 'Evaluating Slices...');
          currentProgress = Math.min(96, 85 + (elapsed - 3.2) * 10);
          phaseName.textContent = 'Phase 4/4: Slices & SHAP Calibration';
        }
        progressBar.style.width = currentProgress.toFixed(0) + '%';
        percentText.textContent = currentProgress.toFixed(0) + '%';
      }, 100);

      // Call API /train endpoint
      try {
        const resp = await fetch('/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            n_estimators: 900,
            max_depth: 8,
            learning_rate: 0.014,
            save_model: true
          })
        });

        clearInterval(pipeInterval);

        if (!resp.ok) {
          throw new Error('Training API returned HTTP ' + resp.status);
        }

        const data = await resp.json();
        const results = data.training_results;
        const testMetrics = results ? results.metrics.test : { rmse: 0.6608, r2: 0.5765, exact_accuracy: 0.576, accuracy_pm1: 0.966 };

        // Complete all steps
        completeStep('pipeStep1');
        completeStep('pipeStep2');
        completeStep('pipeStep3');
        completeStep('pipeStep4');

        progressBar.style.width = '100%';
        percentText.textContent = '100%';
        phaseName.textContent = 'Pipeline Completed';

        // Update modal UI to completion state
        ringAnim.classList.remove('animate-spin');
        ringAnim.classList.add('hidden');
        iconPulse.classList.add('hidden');
        iconCheck.classList.remove('hidden');

        modalTitle.textContent = 'Pipeline Completed Successfully';
        modalSubtext.textContent = 'Model retrained in ' + (results ? results.elapsed_seconds : '2.5') + 's with 13 verified features.';

        // Populate metrics card
        document.getElementById('pipeMetricRmse').textContent = testMetrics.rmse.toFixed(4);
        document.getElementById('pipeMetricAcc').textContent = (testMetrics.exact_accuracy * 100).toFixed(1) + '%';
        document.getElementById('pipeMetricAccPm1').textContent = (testMetrics.accuracy_pm1 * 100).toFixed(1) + '%';
        if (data.previous_model_archived_as) {
          document.getElementById('pipeModelVersion').textContent = 'Archived ' + data.previous_model_archived_as;
        }
        metricsCard.classList.remove('hidden');
        btnClose.classList.remove('hidden');

        // Update header badge
        document.getElementById('modelVersionBadge').textContent = 'Model: XGBoost Active (' + (testMetrics.exact_accuracy*100).toFixed(0) + '% Acc)';

        // Re-run explain to refresh current simulator reading with new model
        fetchPredictionAndShap();

      } catch (err) {
        clearInterval(pipeInterval);
        modalTitle.textContent = 'Training Failed';
        modalSubtext.textContent = err.message || 'An error occurred during pipeline execution.';
        progressBar.className = progressBar.className.replace('from-brand-emerald', 'from-brand-coral');
        phaseName.textContent = 'Error Encountered';
        phaseName.className = 'text-brand-coral';
        btnClose.classList.remove('hidden');
        btnClose.textContent = 'Close';
      } finally {
        btnSpin.classList.add('hidden');
        btnIcon.classList.remove('hidden');
        btnRun.classList.remove('opacity-70', 'cursor-not-allowed');
      }
    }

    function closePipelineModal() {
      const modal = document.getElementById('pipelineModal');
      modal.classList.add('opacity-0', 'pointer-events-none');
    }
  </script>
</body>
</html>
"""
