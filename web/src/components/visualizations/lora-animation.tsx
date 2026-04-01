"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/ui/step-controls";

const STEP_INFO = [
  { title: "LoRA 低秩适配", desc: "LoRA 是参数高效微调的核心方法。让我们用动画看看它是怎么工作的。" },
  { title: "① 原始权重矩阵 W₀", desc: "预训练模型有巨大的权重矩阵（如 4096×4096），包含了模型学到的所有知识。" },
  { title: "② 冻结 W₀", desc: "LoRA 的关键：冻结原始权重，不再更新它。这保留了预训练知识，避免灾难性遗忘。" },
  { title: "③ 添加低秩矩阵 BA", desc: "新增两个小矩阵：B (d×r) 和 A (r×d)，其中 r << d。例如 r=8 时，参数量仅为原来的 0.4%！" },
  { title: "④ 训练 B 和 A", desc: "只训练 B 和 A 这两个小矩阵。梯度只流过它们，显存需求大幅降低。" },
  { title: "⑤ 合并权重", desc: "推理时 W' = W₀ + BA，可以合并成一个矩阵，不增加推理延迟。这就是 LoRA 的精髓！" },
];

export default function LoRAAnimation() {
  const { currentStep, totalSteps, next, prev, reset, isPlaying, toggleAutoPlay } =
    useSteppedVisualization({ totalSteps: 6, autoPlayInterval: 3000 });

  const showW0 = currentStep >= 1;
  const frozen = currentStep >= 2;
  const showBA = currentStep >= 3;
  const training = currentStep >= 4;
  const merged = currentStep >= 5;

  return (
    <div className="w-full rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden">
      <div className="px-6 py-4 border-b border-zinc-100 dark:border-zinc-800">
        <h3 className="text-lg font-bold">🧩 LoRA 低秩适配原理动画</h3>
        <p className="text-xs text-zinc-500 mt-1">W&apos; = W₀ + BA — 不改整本书，只贴便签纸</p>
      </div>

      <div className="p-6 flex justify-center">
        <svg viewBox="0 0 600 300" className="w-full" style={{ maxHeight: 350 }}>
          <defs>
            <linearGradient id="grad-frozen" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#6B7280" />
              <stop offset="100%" stopColor="#9CA3AF" />
            </linearGradient>
            <linearGradient id="grad-lora" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3B82F6" />
              <stop offset="100%" stopColor="#8B5CF6" />
            </linearGradient>
            <linearGradient id="grad-merged" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#10B981" />
              <stop offset="100%" stopColor="#3B82F6" />
            </linearGradient>
          </defs>

          {/* W₀ big matrix */}
          <AnimatePresence>
            {showW0 && !merged && (
              <motion.g initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}>
                <motion.rect
                  x={50} y={50} width={140} height={140} rx={8}
                  fill={frozen ? "url(#grad-frozen)" : "#3B82F6"}
                  animate={{ opacity: frozen ? 0.6 : 1 }}
                />
                {/* Grid lines */}
                {[0, 1, 2, 3].map(i => (
                  <g key={`grid-${i}`}>
                    <line x1={50 + (i + 1) * 35} y1={50} x2={50 + (i + 1) * 35} y2={190} stroke="rgba(255,255,255,0.2)" strokeWidth={0.5} />
                    <line x1={50} y1={50 + (i + 1) * 35} x2={190} y2={50 + (i + 1) * 35} stroke="rgba(255,255,255,0.2)" strokeWidth={0.5} />
                  </g>
                ))}
                <text x={120} y={125} textAnchor="middle" fontSize={16} fontWeight={800} fill="white">W₀</text>
                <text x={120} y={145} textAnchor="middle" fontSize={9} fill="rgba(255,255,255,0.8)">
                  {frozen ? "🔒 已冻结" : "d × d"}
                </text>
                <text x={120} y={210} textAnchor="middle" fontSize={11} fill="#71717A">
                  {frozen ? "原始权重（不更新）" : "原始权重矩阵"}
                </text>
              </motion.g>
            )}
          </AnimatePresence>

          {/* Plus sign */}
          {showBA && !merged && (
            <motion.text x={220} y={128} fontSize={28} fontWeight={800} fill="#71717A" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>+</motion.text>
          )}

          {/* B matrix */}
          <AnimatePresence>
            {showBA && !merged && (
              <motion.g initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
                <motion.rect
                  x={260} y={50} width={40} height={140} rx={6}
                  fill="url(#grad-lora)"
                  animate={training ? { filter: "brightness(1.2)" } : {}}
                />
                <text x={280} y={125} textAnchor="middle" fontSize={14} fontWeight={700} fill="white">B</text>
                <text x={280} y={145} textAnchor="middle" fontSize={8} fill="rgba(255,255,255,0.8)">d×r</text>
                {training && (
                  <motion.rect x={260} y={50} width={40} height={140} rx={6} fill="none" stroke="#FCD34D" strokeWidth={2}
                    animate={{ opacity: [0, 1, 0] }} transition={{ duration: 1, repeat: Infinity }} />
                )}
              </motion.g>
            )}
          </AnimatePresence>

          {/* × sign */}
          {showBA && !merged && (
            <motion.text x={316} y={128} fontSize={16} fontWeight={700} fill="#71717A" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>×</motion.text>
          )}

          {/* A matrix */}
          <AnimatePresence>
            {showBA && !merged && (
              <motion.g initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
                <motion.rect
                  x={340} y={90} width={140} height={40} rx={6}
                  fill="url(#grad-lora)"
                  animate={training ? { filter: "brightness(1.2)" } : {}}
                />
                <text x={410} y={115} textAnchor="middle" fontSize={14} fontWeight={700} fill="white">A</text>
                <text x={410} y={100} textAnchor="middle" fontSize={8} fill="rgba(255,255,255,0.8)">r×d</text>
                {training && (
                  <motion.rect x={340} y={90} width={140} height={40} rx={6} fill="none" stroke="#FCD34D" strokeWidth={2}
                    animate={{ opacity: [0, 1, 0] }} transition={{ duration: 1, repeat: Infinity, delay: 0.5 }} />
                )}
              </motion.g>
            )}
          </AnimatePresence>

          {/* r label */}
          {showBA && !merged && (
            <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
              <text x={410} y={160} textAnchor="middle" fontSize={11} fill="#8B5CF6" fontWeight={600}>r = 8 (远小于 d)</text>
              <text x={410} y={178} textAnchor="middle" fontSize={10} fill="#71717A">参数量仅 0.4%</text>
            </motion.g>
          )}

          {/* Training gradient flow */}
          {training && !merged && (
            <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <text x={340} y={60} fontSize={10} fill="#F59E0B" fontWeight={600}>⚡ 梯度只流过 B 和 A</text>
              <motion.line x1={280} y1={45} x2={280} y2={50} stroke="#F59E0B" strokeWidth={2}
                strokeDasharray="4 2" animate={{ strokeDashoffset: [8, 0] }} transition={{ duration: 0.5, repeat: Infinity }} />
              <motion.line x1={410} y1={85} x2={410} y2={90} stroke="#F59E0B" strokeWidth={2}
                strokeDasharray="4 2" animate={{ strokeDashoffset: [8, 0] }} transition={{ duration: 0.5, repeat: Infinity }} />
            </motion.g>
          )}

          {/* Merged result */}
          <AnimatePresence>
            {merged && (
              <motion.g initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}>
                <rect x={180} y={40} width={160} height={160} rx={12} fill="url(#grad-merged)" />
                {[0, 1, 2, 3].map(i => (
                  <g key={`mgrid-${i}`}>
                    <line x1={180 + (i + 1) * 40} y1={40} x2={180 + (i + 1) * 40} y2={200} stroke="rgba(255,255,255,0.15)" strokeWidth={0.5} />
                    <line x1={180} y1={40 + (i + 1) * 40} x2={340} y2={40 + (i + 1) * 40} stroke="rgba(255,255,255,0.15)" strokeWidth={0.5} />
                  </g>
                ))}
                <text x={260} y={115} textAnchor="middle" fontSize={16} fontWeight={800} fill="white">W&apos; = W₀ + BA</text>
                <text x={260} y={140} textAnchor="middle" fontSize={10} fill="rgba(255,255,255,0.8)">合并后无额外延迟</text>
                <text x={260} y={225} textAnchor="middle" fontSize={12} fill="#10B981" fontWeight={600}>✅ 推理时与全参数微调等价</text>

                <motion.rect x={180} y={40} width={160} height={160} rx={12} fill="none" stroke="white" strokeWidth={2}
                  initial={{ opacity: 0 }} animate={{ opacity: [0, 0.6, 0] }} transition={{ duration: 2, repeat: Infinity }} />
              </motion.g>
            )}
          </AnimatePresence>

          {/* Comparison box */}
          {showBA && (
            <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
              <rect x={500} y={50} width={90} height={170} rx={8} fill="#F4F4F5" stroke="#E4E4E7" />
              <text x={545} y={72} textAnchor="middle" fontSize={9} fontWeight={700} fill="#52525B">显存对比</text>
              <rect x={520} y={85} width={16} height={100} rx={3} fill="#EF4444" opacity={0.7} />
              <text x={528} y={80} textAnchor="middle" fontSize={7} fill="#71717A">全参</text>
              <rect x={545} y={145} width={16} height={40} rx={3} fill="#3B82F6" />
              <text x={553} y={140} textAnchor="middle" fontSize={7} fill="#71717A">LoRA</text>
              <rect x={570} y={161} width={16} height={24} rx={3} fill="#10B981" />
              <text x={578} y={156} textAnchor="middle" fontSize={7} fill="#71717A">QLoRA</text>
              <text x={545} y={205} textAnchor="middle" fontSize={8} fill="#71717A">60G → 16G → 6G</text>
            </motion.g>
          )}
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
