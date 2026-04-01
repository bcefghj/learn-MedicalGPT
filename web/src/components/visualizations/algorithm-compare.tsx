"use client";

import { motion } from "framer-motion";
import { useState } from "react";

const ALGORITHMS = [
  {
    id: "ppo",
    name: "PPO (RLHF)",
    color: "#EF4444",
    models: 4,
    complexity: 95,
    stability: 60,
    memory: 90,
    effect: 88,
    desc: "精细但笨重",
    icon: "🎯",
    pros: ["效果精细", "控制力强"],
    cons: ["4个模型", "训练不稳", "显存巨大"],
  },
  {
    id: "dpo",
    name: "DPO",
    color: "#3B82F6",
    models: 2,
    complexity: 40,
    stability: 85,
    memory: 50,
    effect: 82,
    desc: "简单而高效",
    icon: "⚡",
    pros: ["只需2模型", "实现简单", "训练稳定"],
    cons: ["数据要求高", "依赖SFT"],
  },
  {
    id: "orpo",
    name: "ORPO",
    color: "#10B981",
    models: 1,
    complexity: 30,
    stability: 80,
    memory: 35,
    effect: 78,
    desc: "一步到位",
    icon: "🚀",
    pros: ["无需参考模型", "SFT+对齐一步", "防遗忘"],
    cons: ["效果略弱", "较新"],
  },
  {
    id: "grpo",
    name: "GRPO",
    color: "#8B5CF6",
    models: 1,
    complexity: 45,
    stability: 82,
    memory: 40,
    effect: 90,
    desc: "前沿之选",
    icon: "🧠",
    pros: ["无需Critic", "显存省50%", "推理任务强"],
    cons: ["需多次采样", "较新方法"],
  },
];

function Bar({ value, color, label }: { value: number; color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-zinc-500 w-10 text-right">{label}</span>
      <div className="flex-1 h-3 bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          whileInView={{ width: `${value}%` }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>
      <span className="text-[10px] font-mono text-zinc-400 w-6">{value}</span>
    </div>
  );
}

export default function AlgorithmCompare() {
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="w-full rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden">
      <div className="px-6 py-4 border-b border-zinc-100 dark:border-zinc-800">
        <h3 className="text-lg font-bold">📊 对齐算法对比 — 点击卡片查看详情</h3>
        <p className="text-xs text-zinc-500 mt-1">PPO vs DPO vs ORPO vs GRPO，面试必背的算法选择决策</p>
      </div>

      <div className="p-6 grid grid-cols-2 md:grid-cols-4 gap-3">
        {ALGORITHMS.map((algo) => (
          <motion.div
            key={algo.id}
            onClick={() => setSelected(selected === algo.id ? null : algo.id)}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.98 }}
            className={`cursor-pointer rounded-xl border-2 p-4 transition-colors ${
              selected === algo.id
                ? "border-current shadow-lg"
                : "border-zinc-200 dark:border-zinc-700"
            }`}
            style={{ borderColor: selected === algo.id ? algo.color : undefined }}
          >
            <div className="text-2xl mb-2">{algo.icon}</div>
            <h4 className="text-sm font-bold" style={{ color: algo.color }}>{algo.name}</h4>
            <p className="text-[10px] text-zinc-500 mt-0.5">{algo.desc}</p>
            <div className="mt-3 text-xs font-mono" style={{ color: algo.color }}>
              {algo.models} 模型
            </div>
          </motion.div>
        ))}
      </div>

      {/* Detail panel */}
      {selected && (() => {
        const algo = ALGORITHMS.find(a => a.id === selected)!;
        return (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="px-6 pb-6"
          >
            <div className="rounded-xl p-4" style={{ backgroundColor: algo.color + "10" }}>
              <h4 className="font-bold mb-3" style={{ color: algo.color }}>{algo.icon} {algo.name} 详细指标</h4>

              <div className="space-y-2 mb-4">
                <Bar value={algo.complexity} color={algo.color} label="复杂度" />
                <Bar value={algo.stability} color={algo.color} label="稳定性" />
                <Bar value={algo.memory} color={algo.color} label="显存" />
                <Bar value={algo.effect} color={algo.color} label="效果" />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs font-bold text-green-600 mb-1">✅ 优势</p>
                  {algo.pros.map(p => (
                    <p key={p} className="text-xs text-zinc-600 dark:text-zinc-400">• {p}</p>
                  ))}
                </div>
                <div>
                  <p className="text-xs font-bold text-red-500 mb-1">⚠️ 局限</p>
                  {algo.cons.map(c => (
                    <p key={c} className="text-xs text-zinc-600 dark:text-zinc-400">• {c}</p>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        );
      })()}
    </div>
  );
}
