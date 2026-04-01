"use client";

import { Play, Pause, SkipBack, SkipForward, RotateCcw } from "lucide-react";

interface StepControlsProps {
  currentStep: number;
  totalSteps: number;
  onPrev: () => void;
  onNext: () => void;
  onReset: () => void;
  isPlaying: boolean;
  onToggleAutoPlay: () => void;
  stepTitle: string;
  stepDescription: string;
}

export function StepControls({
  currentStep, totalSteps, onPrev, onNext, onReset,
  isPlaying, onToggleAutoPlay, stepTitle, stepDescription,
}: StepControlsProps) {
  return (
    <div className="mt-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 p-4">
      <div className="mb-3">
        <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">{stepTitle}</h4>
        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">{stepDescription}</p>
      </div>
      <div className="flex items-center gap-2">
        <button onClick={onReset} className="p-1.5 rounded-lg hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors">
          <RotateCcw size={14} />
        </button>
        <button onClick={onPrev} disabled={currentStep === 0} className="p-1.5 rounded-lg hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors disabled:opacity-30">
          <SkipBack size={14} />
        </button>
        <button onClick={onToggleAutoPlay} className="p-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors">
          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button onClick={onNext} disabled={currentStep === totalSteps - 1} className="p-1.5 rounded-lg hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors disabled:opacity-30">
          <SkipForward size={14} />
        </button>
        <div className="ml-auto flex items-center gap-1.5">
          {Array.from({ length: totalSteps }, (_, i) => (
            <div key={i} className={`w-2 h-2 rounded-full transition-colors ${i <= currentStep ? "bg-blue-500" : "bg-zinc-300 dark:bg-zinc-600"}`} />
          ))}
          <span className="ml-2 text-xs text-zinc-400">{currentStep + 1}/{totalSteps}</span>
        </div>
      </div>
    </div>
  );
}
