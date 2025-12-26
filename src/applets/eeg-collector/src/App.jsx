import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Download } from 'lucide-react';

const EEGDataCollector = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentTrial, setCurrentTrial] = useState(0);
  const [phase, setPhase] = useState('rest'); // rest, cue, imagery
  const [timeLeft, setTimeLeft] = useState(5);
  const [trialSequence, setTrialSequence] = useState([]);
  const [sessionStarted, setSessionStarted] = useState(false);
  
  const intervalRef = useRef(null);
  const startTimeRef = useRef(null);
  const pausedTimeRef = useRef(0);

  // Phase durations in seconds
  const PHASE_DURATIONS = {
    rest: 5,
    cue: 1,
    imagery: 3
  };

  // Total trials: 80 rest + 80 thumb + 80 index + 80 pinky = 320
  const TRIALS_PER_CLASS = 80;
  const CLASSES = ['rest', 'thumb', 'index', 'pinky'];

  // Initialize trial sequence
  useEffect(() => {
    if (!sessionStarted) {
      const sequence = [];
      CLASSES.forEach(cls => {
        for (let i = 0; i < TRIALS_PER_CLASS; i++) {
          sequence.push(cls);
        }
      });
      
      // Fisher-Yates shuffle
      for (let i = sequence.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sequence[i], sequence[j]] = [sequence[j], sequence[i]];
      }
      
      setTrialSequence(sequence);
    }
  }, [sessionStarted]);

  // Timer logic
  useEffect(() => {
    if (isRunning && !isPaused) {
      intervalRef.current = setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 0.1) {
            advancePhase();
            return getNextPhaseDuration();
          }
          return prev - 0.1;
        });
      }, 100);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning, isPaused, phase, currentTrial]);

  const getNextPhaseDuration = () => {
    const phases = ['rest', 'cue', 'imagery'];
    const currentIndex = phases.indexOf(phase);
    const nextPhase = phases[(currentIndex + 1) % phases.length];
    return PHASE_DURATIONS[nextPhase];
  };

  const advancePhase = () => {
    const currentClass = trialSequence[currentTrial];
    
    // For rest trials, skip cue and imagery phases
    if (currentClass === 'rest') {
      if (phase === 'rest') {
        // Move to next trial after rest period
        if (currentTrial >= trialSequence.length - 1) {
          completeSession();
        } else {
          setCurrentTrial(prev => prev + 1);
          setPhase('rest');
        }
      }
      return;
    }

    // For motor imagery trials (thumb, index, pinky)
    switch (phase) {
      case 'rest':
        setPhase('cue');
        break;
      case 'cue':
        setPhase('imagery');
        break;
      case 'imagery':
        if (currentTrial >= trialSequence.length - 1) {
          completeSession();
        } else {
          setCurrentTrial(prev => prev + 1);
          setPhase('rest');
        }
        break;
    }
  };

  const completeSession = () => {
    setIsRunning(false);
    setIsPaused(false);
    alert('Session complete! Download the trial sequence file.');
  };

  const startSession = () => {
    if (!sessionStarted) {
      setSessionStarted(true);
    }
    setIsRunning(true);
    setIsPaused(false);
    startTimeRef.current = Date.now();
  };

  const pauseSession = () => {
    setIsPaused(!isPaused);
    if (!isPaused) {
      pausedTimeRef.current = Date.now();
    }
  };

  const resetSession = () => {
    setIsRunning(false);
    setIsPaused(false);
    setCurrentTrial(0);
    setPhase('rest');
    setTimeLeft(5);
    setSessionStarted(false);
    setTrialSequence([]);
  };

  const downloadSequence = () => {
    const content = trialSequence.join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `eeg_trial_sequence_${new Date().getTime()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getCurrentClass = () => trialSequence[currentTrial] || 'rest';
  const currentClass = getCurrentClass();

  const getPhaseDisplay = () => {
    if (currentClass === 'rest') {
      return 'Rest Period';
    }
    
    switch (phase) {
      case 'rest':
        return 'Rest';
      case 'cue':
        return `Cue: ${currentClass.toUpperCase()}`;
      case 'imagery':
        return 'Motor Imagery';
      default:
        return '';
    }
  };

  const getBackgroundColor = () => {
    if (currentClass === 'rest') return 'bg-gray-100';
    if (phase === 'rest') return 'bg-gray-100';
    if (phase === 'cue') return 'bg-blue-100';
    if (phase === 'imagery') return 'bg-green-100';
    return 'bg-white';
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            EEG Motor Imagery Data Collection
          </h1>
          <p className="text-gray-600 mb-4">
            320 trials: 80 rest, 80 thumb, 80 index, 80 pinky (randomized)
          </p>
          
          <div className="flex gap-4 mb-6">
            <button
              onClick={startSession}
              disabled={isRunning && !isPaused}
              className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
            >
              <Play size={20} />
              {sessionStarted ? 'Resume' : 'Start Session'}
            </button>
            
            <button
              onClick={pauseSession}
              disabled={!isRunning}
              className="flex items-center gap-2 px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
            >
              <Pause size={20} />
              {isPaused ? 'Resume' : 'Pause'}
            </button>
            
            <button
              onClick={resetSession}
              className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
            >
              <RotateCcw size={20} />
              Reset
            </button>
            
            <button
              onClick={downloadSequence}
              disabled={!sessionStarted}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
            >
              <Download size={20} />
              Download Sequence
            </button>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Trial</div>
              <div className="text-2xl font-bold text-gray-800">
                {currentTrial + 1} / {trialSequence.length}
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Current Class</div>
              <div className="text-2xl font-bold text-gray-800 capitalize">
                {currentClass}
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Phase</div>
              <div className="text-2xl font-bold text-gray-800 capitalize">
                {phase.replace('-', ' ')}
              </div>
            </div>
          </div>
        </div>

        <div className={`relative ${getBackgroundColor()} rounded-lg shadow-lg transition-colors duration-300`} 
             style={{ height: '500px' }}>
          <div className="absolute top-6 right-6 text-right">
            <div className="text-sm text-gray-600 mb-1">Time Remaining</div>
            <div className="text-5xl font-bold text-gray-800">
              {timeLeft.toFixed(1)}s
            </div>
          </div>

          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-6xl font-bold text-gray-800 mb-4">
                {getPhaseDisplay()}
              </div>
              
              {phase === 'cue' && currentClass !== 'rest' && (
                <div className="text-4xl text-gray-600 mt-8">
                  Prepare to imagine {currentClass} movement
                </div>
              )}
              
              {phase === 'imagery' && currentClass !== 'rest' && (
                <div className="text-4xl text-gray-600 mt-8">
                  Imagine moving your {currentClass}
                </div>
              )}

              {!isRunning && !sessionStarted && (
                <div className="text-2xl text-gray-500 mt-8">
                  Press "Start Session" to begin
                </div>
              )}

              {isPaused && (
                <div className="text-2xl text-yellow-600 mt-8 font-bold">
                  PAUSED
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-3">Instructions</h2>
          <ul className="space-y-2 text-gray-700">
            <li><strong>Rest Period (5s):</strong> Relax and prepare for the next trial</li>
            <li><strong>Cue Phase (1s):</strong> View which finger to imagine (thumb/index/pinky)</li>
            <li><strong>Motor Imagery (3s):</strong> Imagine moving the specified finger</li>
            <li><strong>Rest Trials:</strong> Only show 5s rest period, then proceed to next trial</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default EEGDataCollector;