//Go to src/applets/eeg-collector and run "npm run dev" to start the app

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
  const phaseRef = useRef(phase);
  const currentTrialRef = useRef(currentTrial);
  const isAdvancingRef = useRef(false);

  // Keep refs in sync with state
  useEffect(() => {
    phaseRef.current = phase;
    isAdvancingRef.current = false; // Reset flag when phase actually changes
  }, [phase]);

  useEffect(() => {
    currentTrialRef.current = currentTrial;
  }, [currentTrial]);

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
          const newTime = prev - 0.1;
          if (newTime <= 0) {
            // Prevent multiple advancePhase calls while waiting for state update
            if (!isAdvancingRef.current) {
              isAdvancingRef.current = true;
              setTimeout(() => advancePhase(), 0);
            }
            return 0;
          }
          return newTime;
        });
      }, 100);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning, isPaused]);

  const advancePhase = () => {
    const currentPhase = phaseRef.current;
    const trial = currentTrialRef.current;
    
    // All trials follow the same sequence: rest -> cue -> imagery
    switch (currentPhase) {
      case 'rest':
        setPhase('cue');
        setTimeLeft(PHASE_DURATIONS.cue);
        break;
      case 'cue':
        setPhase('imagery');
        setTimeLeft(PHASE_DURATIONS.imagery);
        break;
      case 'imagery':
        if (trial >= trialSequence.length - 1) {
          completeSession();
        } else {
          setCurrentTrial(prev => prev + 1);
          setPhase('rest');
          setTimeLeft(PHASE_DURATIONS.rest);
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
  };

  const pauseSession = () => {
    setIsPaused(!isPaused);
  };

  const resetSession = () => {
    setIsRunning(false);
    setIsPaused(false);
    setCurrentTrial(0);
    setPhase('rest');
    setTimeLeft(5);
    setSessionStarted(false);
    setTrialSequence([]);
    isAdvancingRef.current = false;
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
    switch (phase) {
      case 'rest':
        return 'Rest';
      case 'cue':
        return `Cue: ${currentClass.toUpperCase()}`;
      case 'imagery':
        return currentClass === 'rest' ? 'Rest (Motor Imagery Period)' : 'Motor Imagery';
      default:
        return '';
    }
  };

  const getBackgroundColor = () => {
    if (phase === 'rest') return 'bg-gray-100';
    if (phase === 'cue') return 'bg-blue-100';
    if (phase === 'imagery') {
      return currentClass === 'rest' ? 'bg-gray-100' : 'bg-green-100';
    }
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
              <div className="text-xs text-gray-500 mt-1">
                (Trial {currentTrial + 1}: {trialSequence[currentTrial]})
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
              
              {phase === 'cue' && (
                <div className="text-4xl text-gray-600 mt-8">
                  {currentClass === 'rest' 
                    ? 'Prepare to rest' 
                    : `Prepare to imagine ${currentClass} movement`}
                </div>
              )}
              
              {phase === 'imagery' && (
                <div className="text-4xl text-gray-600 mt-8">
                  {currentClass === 'rest' 
                    ? 'Continue resting - no imagery' 
                    : `Imagine moving your ${currentClass}`}
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
            <li><strong>Cue Phase (1s):</strong> View the class (rest/thumb/index/pinky)</li>
            <li><strong>Motor Imagery (3s):</strong> For finger classes, imagine moving that finger. For rest class, continue resting with no imagery</li>
            <li><strong>All trials follow the same structure:</strong> Rest (5s) → Cue (1s) → Motor Imagery period (3s)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default EEGDataCollector;