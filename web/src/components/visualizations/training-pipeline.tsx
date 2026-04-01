"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/ui/step-controls";

const STAGES = [
  { id: "base", label: "基座模型", sub: "Qwen/LLaMA", x: 80, color: "#6B7280" },
  { id: "pt", label: "增量预训练", sub: "PT", x: 230, color: "#3B82F6" },
  { id: "sft", label: "有监督微调", sub: "SFT", x: 380, color: "#10B981" },
  { id: "align", label: "偏好对齐", sub: "RLHF/DPO", x: 530, color: "#8B5CF6" },
  { id: "deploy", label: "部署上线", sub: "推理服务", x: 680, color: "#F59E0B" },
];

const DATA_LABELS = [
  { text: "医疗文本语料", x: 230, color: "#3B82F6" },
  { text: "指令微调数据", x: 380, color: "#10B981" },
  { text: "人类偏好数据", x: 530, color: "#8B5CF6" },
];

const STEP_INFO = [
  { title: "MedicalGPT 训练全流程", desc: "从基座模型到医疗问答，经历4个核心阶段。点击播放查看动画演示。" },
  { title: "① 选择基座模型", desc: "选一个通用大模型作为起点，如 Qwen-2.5、LLaMA-3。它已经\"读\"过互联网上的大量文本。" },
  { title: "② 增量预训练 (PT)", desc: "喂入海量医疗文本（论文、百科、病历），让模型理解医疗领域的语言分布。目标：Next Token Prediction。" },
  { title: "③ 有监督微调 (SFT)", desc: "用「问题→回答」格式的医疗指令数据训练。让模型从\"会读\"变成\"会答\"。240万条医疗数据。" },
  { title: "④ 偏好对齐 (RLHF/DPO)", desc: "用人类偏好数据教模型区分「好回答」和「坏回答」。遵循 Helpful、Honest、Harmless 原则。" },
  { title: "⑤ 部署上线", desc: "训练完成！通过 Gradio/vLLM 部署，提供医疗问答服务。可以回答\"小孩发烧怎么办\"等问题。" },
];

export default function TrainingPipeline() {
  const { currentStep, totalSteps, next, prev, reset, isPlaying, toggleAutoPlay } =
    useSteppedVisualization({ totalSteps: 6, autoPlayInterval: 3000 });

  return (
    <div className="w-full rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden">
      <div className="px-6 py-4 border-b border-zinc-100 dark:border-zinc-800">
        <h3 className="text-lg font-bold">🏥 MedicalGPT 训练 Pipeline 动画</h3>
        <p className="text-xs text-zinc-500 mt-1">交互式步进演示 — 点击播放或手动切换每个阶段</p>
      </div>

      <div className="p-6">
        <svg viewBox="0 0 760 220" className="w-full" style={{ maxHeight: 280 }}>
          <defs>
            <filter id="glow-blue"><feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            <filter id="glow-green"><feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            <marker id="arrow" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
              <polygon points="0 0, 10 3.5, 0 7" fill="#94A3B8" />
            </marker>
            <marker id="arrow-active" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
              <polygon points="0 0, 10 3.5, 0 7" fill="#3B82F6" />
            </marker>
          </defs>

          {/* Connection arrows */}
          {STAGES.slice(0, -1).map((stage, i) => {
            const nextStage = STAGES[i + 1];
            const isActive = currentStep > i + 1;
            const isAnimating = currentStep === i + 2;
            return (
              <g key={`arrow-${i}`}>
                <motion.line
                  x1={stage.x + 55} y1={100}
                  x2={nextStage.x - 55} y2={100}
                  stroke={isActive || isAnimating ? "#3B82F6" : "#D4D4D8"}
                  strokeWidth={isAnimating ? 3 : 2}
                  strokeDasharray={isAnimating ? "6 4" : "none"}
                  markerEnd={isActive || isAnimating ? "url(#arrow-active)" : "url(#arrow)"}
                  animate={{ opacity: isActive || isAnimating ? 1 : 0.4 }}
                />
                {isAnimating && (
                  <motion.circle
                    cx={stage.x + 55} cy={100} r={4}
                    fill="#3B82F6"
                    animate={{ cx: [stage.x + 55, nextStage.x - 55] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                  />
                )}
              </g>
            );
          })}

          {/* Data labels on top */}
          <AnimatePresence>
            {DATA_LABELS.map((d, i) => {
              const show = currentStep === i + 2;
              return show ? (
                <motion.g key={d.text} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                  <rect x={d.x - 55} y={28} width={110} height={24} rx={12} fill={d.color} opacity={0.15} />
                  <text x={d.x} y={44} textAnchor="middle" fontSize={11} fill={d.color} fontWeight={600}>{d.text}</text>
                  <line x1={d.x} y1={52} x2={d.x} y2={68} stroke={d.color} strokeWidth={1.5} strokeDasharray="3 2" />
                </motion.g>
              ) : null;
            })}
          </AnimatePresence>

          {/* Stage nodes */}
          {STAGES.map((stage, i) => {
            const isActive = currentStep >= i + 1;
            const isCurrent = currentStep === i + 1;
            return (
              <motion.g key={stage.id} animate={{ scale: isCurrent ? 1.05 : 1 }} style={{ transformOrigin: `${stage.x}px 100px` }}>
                <motion.rect
                  x={stage.x - 50} y={70} width={100} height={60} rx={12}
                  fill={isActive ? stage.color : "#E4E4E7"}
                  opacity={isActive ? 1 : 0.5}
                  animate={{
                    filter: isCurrent ? "url(#glow-blue)" : "none",
                  }}
                  transition={{ duration: 0.5 }}
                />
                <text x={stage.x} y={95} textAnchor="middle" fontSize={12} fontWeight={700} fill={isActive ? "white" : "#71717A"}>
                  {stage.label}
                </text>
                <text x={stage.x} y={112} textAnchor="middle" fontSize={10} fill={isActive ? "rgba(255,255,255,0.8)" : "#A1A1AA"}>
                  {stage.sub}
                </text>

                {isCurrent && (
                  <motion.rect
                    x={stage.x - 50} y={70} width={100} height={60} rx={12}
                    fill="none" stroke="white" strokeWidth={2}
                    initial={{ opacity: 0 }} animate={{ opacity: [0, 0.8, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                )}
              </motion.g>
            );
          })}

          {/* Bottom: result indicator */}
          <AnimatePresence>
            {currentStep === 5 && (
              <motion.g initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
                <rect x={605} y={150} width={150} height={40} rx={20} fill="#10B981" opacity={0.15} />
                <text x={680} y={175} textAnchor="middle" fontSize={12} fill="#10B981" fontWeight={700}>✅ 医疗问答就绪</text>
              </motion.g>
            )}
          </AnimatePresence>
        </svg>
      </div>

      <div className="px-6 pb-6">
        <StepControls
          currentStep={currentStep} totalSteps={totalSteps}
          onPrev={prev} onNext={next} onReset={reset}
          isPlaying={isPlaying} onToggleAutoPlay={toggleAutoPlay}
          stepTitle={STEP_INFO[currentStep].title}
          stepDescription={STEP_INFO[currentStep].desc}
        />
      </div>
    </div>
  );
}
