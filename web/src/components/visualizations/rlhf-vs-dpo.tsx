"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/ui/step-controls";

const STEP_INFO = [
  { title: "RLHF vs DPO 对比", desc: "两种让模型学习人类偏好的方法，让我们对比看看它们的区别。" },
  { title: "RLHF：第一步 — 训练奖励模型", desc: "收集人类偏好数据（对同一问题的好回答和坏回答），训练一个奖励模型来给回答打分。" },
  { title: "RLHF：第二步 — PPO强化学习", desc: "用奖励模型的分数作为奖励信号，通过PPO算法优化策略模型。需要同时运行4个模型！" },
  { title: "DPO：直接优化", desc: "DPO 直接从偏好数据中学习，跳过了奖励模型和PPO！只需要2个模型，训练更简单、更稳定。" },
  { title: "对比总结", desc: "RLHF效果更精细但复杂；DPO更简单但对数据质量要求更高。MedicalGPT两种方法都支持！" },
];

export default function RLHFvsDPO() {
  const { currentStep, totalSteps, next, prev, reset, isPlaying, toggleAutoPlay } =
    useSteppedVisualization({ totalSteps: 5, autoPlayInterval: 4000 });

  return (
    <div className="w-full rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden">
      <div className="px-6 py-4 border-b border-zinc-100 dark:border-zinc-800">
        <h3 className="text-lg font-bold">⚖️ RLHF vs DPO 对齐方法对比</h3>
        <p className="text-xs text-zinc-500 mt-1">两条路通向同一个目标 — 让模型的回答更符合人类偏好</p>
      </div>

      <div className="p-6">
        <svg viewBox="0 0 700 320" className="w-full" style={{ maxHeight: 380 }}>
          {/* RLHF side */}
          <AnimatePresence>
            {currentStep >= 1 && (
              <motion.g initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
                <rect x={10} y={10} width={330} height={290} rx={12} fill="#FEF2F2" stroke="#FECACA" strokeWidth={1} />
                <text x={175} y={35} textAnchor="middle" fontSize={14} fontWeight={800} fill="#DC2626">RLHF 路线</text>
                <text x={175} y={52} textAnchor="middle" fontSize={9} fill="#F87171">复杂但精细</text>

                {/* Step 1: Preference data */}
                <rect x={30} y={70} width={120} height={45} rx={8} fill="#FCA5A5" />
                <text x={90} y={90} textAnchor="middle" fontSize={10} fontWeight={600} fill="#7F1D1D">👍👎 偏好数据</text>
                <text x={90} y={104} textAnchor="middle" fontSize={8} fill="#991B1B">chosen / rejected</text>

                {/* Arrow */}
                <line x1={150} y1={92} x2={175} y2={92} stroke="#DC2626" strokeWidth={1.5} markerEnd="url(#arrow-red)" />

                {/* Step 2: Reward Model */}
                <motion.rect x={180} y={70} width={120} height={45} rx={8}
                  fill={currentStep >= 1 ? "#EF4444" : "#E4E4E7"}
                  animate={currentStep === 1 ? { opacity: [0.7, 1, 0.7] } : { opacity: 1 }}
                  transition={{ duration: 1.5, repeat: currentStep === 1 ? Infinity : 0 }}
                />
                <text x={240} y={90} textAnchor="middle" fontSize={10} fontWeight={700} fill="white">奖励模型 RM</text>
                <text x={240} y={104} textAnchor="middle" fontSize={8} fill="rgba(255,255,255,0.8)">学习打分</text>
              </motion.g>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {currentStep >= 2 && (
              <motion.g initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                {/* Arrow down */}
                <line x1={240} y1={115} x2={240} y2={140} stroke="#DC2626" strokeWidth={1.5} />

                {/* PPO training area */}
                <rect x={30} y={145} width={290} height={140} rx={8} fill="rgba(239,68,68,0.1)" stroke="#FECACA" strokeDasharray="4 2" />
                <text x={175} y={165} textAnchor="middle" fontSize={10} fontWeight={700} fill="#DC2626">PPO 强化学习训练</text>

                {/* 4 models */}
                {[
                  { label: "策略模型", sub: "Actor", x: 65, color: "#EF4444" },
                  { label: "参考模型", sub: "Ref", x: 145, color: "#F97316" },
                  { label: "奖励模型", sub: "RM", x: 225, color: "#DC2626" },
                  { label: "价值模型", sub: "Critic", x: 295, color: "#B91C1C" },
                ].map((m, i) => (
                  <motion.g key={m.sub}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.15 }}
                  >
                    <rect x={m.x - 30} y={178} width={60} height={40} rx={6} fill={m.color} />
                    <text x={m.x} y={195} textAnchor="middle" fontSize={8} fontWeight={600} fill="white">{m.label}</text>
                    <text x={m.x} y={208} textAnchor="middle" fontSize={7} fill="rgba(255,255,255,0.7)">{m.sub}</text>
                  </motion.g>
                ))}

                <text x={175} y={240} textAnchor="middle" fontSize={9} fill="#EF4444">⚠️ 需同时加载 4 个模型，显存需求巨大</text>
                <text x={175} y={256} textAnchor="middle" fontSize={9} fill="#EF4444">⚠️ 训练不稳定，超参数敏感</text>

                {/* Circular arrows for PPO loop */}
                <motion.circle cx={175} y={198} r={55} fill="none" stroke="#DC2626" strokeWidth={1} strokeDasharray="4 4" opacity={0.4}
                  animate={{ strokeDashoffset: [0, -20] }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }} />
              </motion.g>
            )}
          </AnimatePresence>

          {/* DPO side */}
          <AnimatePresence>
            {currentStep >= 3 && (
              <motion.g initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
                <rect x={360} y={10} width={330} height={290} rx={12} fill="#F0FDF4" stroke="#BBF7D0" strokeWidth={1} />
                <text x={525} y={35} textAnchor="middle" fontSize={14} fontWeight={800} fill="#16A34A">DPO 路线</text>
                <text x={525} y={52} textAnchor="middle" fontSize={9} fill="#22C55E">简单而高效</text>

                {/* Preference data */}
                <rect x={380} y={70} width={120} height={45} rx={8} fill="#86EFAC" />
                <text x={440} y={90} textAnchor="middle" fontSize={10} fontWeight={600} fill="#14532D">👍👎 偏好数据</text>
                <text x={440} y={104} textAnchor="middle" fontSize={8} fill="#166534">chosen / rejected</text>

                {/* Direct arrow */}
                <motion.line x1={500} y1={92} x2={540} y2={92} stroke="#16A34A" strokeWidth={2}
                  animate={{ strokeDashoffset: [10, 0] }} transition={{ duration: 0.8, repeat: Infinity }}
                  strokeDasharray="5 3"
                />
                <text x={520} y={84} textAnchor="middle" fontSize={8} fill="#16A34A" fontWeight={600}>直接!</text>

                {/* Only 2 models */}
                <rect x={380} y={145} width={290} height={100} rx={8} fill="rgba(22,163,74,0.1)" stroke="#BBF7D0" strokeDasharray="4 2" />
                <text x={525} y={165} textAnchor="middle" fontSize={10} fontWeight={700} fill="#16A34A">DPO 直接优化</text>

                {[
                  { label: "策略模型", sub: "Policy", x: 460, color: "#16A34A" },
                  { label: "参考模型", sub: "Ref", x: 590, color: "#15803D" },
                ].map((m, i) => (
                  <motion.g key={m.sub}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + i * 0.15 }}
                  >
                    <rect x={m.x - 45} y={178} width={90} height={45} rx={6} fill={m.color} />
                    <text x={m.x} y={198} textAnchor="middle" fontSize={10} fontWeight={600} fill="white">{m.label}</text>
                    <text x={m.x} y={213} textAnchor="middle" fontSize={8} fill="rgba(255,255,255,0.7)">{m.sub}</text>
                  </motion.g>
                ))}

                <text x={525} y={250} textAnchor="middle" fontSize={9} fill="#16A34A">✅ 只需 2 个模型，显存友好</text>
                <text x={525} y={266} textAnchor="middle" fontSize={9} fill="#16A34A">✅ 训练简单稳定，效果优秀</text>
              </motion.g>
            )}
          </AnimatePresence>

          {/* VS badge */}
          <AnimatePresence>
            {currentStep >= 3 && (
              <motion.g initial={{ scale: 0 }} animate={{ scale: 1 }}>
                <circle cx={350} cy={150} r={20} fill="#18181B" />
                <text x={350} y={155} textAnchor="middle" fontSize={12} fontWeight={800} fill="white">VS</text>
              </motion.g>
            )}
          </AnimatePresence>

          {/* Comparison summary */}
          {currentStep >= 4 && (
            <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <rect x={550} y={70} width={130} height={60} rx={8} fill="#F5F3FF" stroke="#DDD6FE" />
              <text x={615} y={92} textAnchor="middle" fontSize={9} fontWeight={700} fill="#7C3AED">MedicalGPT</text>
              <text x={615} y={107} textAnchor="middle" fontSize={8} fill="#8B5CF6">两种都支持 ✓</text>
              <text x={615} y={122} textAnchor="middle" fontSize={7} fill="#A78BFA">推荐新手用 DPO</text>
            </motion.g>
          )}

          <defs>
            <marker id="arrow-red" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="6" markerHeight="5" orient="auto-start-reverse">
              <polygon points="0 0, 10 3.5, 0 7" fill="#DC2626" />
            </marker>
          </defs>
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
