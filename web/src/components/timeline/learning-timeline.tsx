"use client";

import { motion } from "framer-motion";

const PHASES = [
  {
    phase: "Phase 1",
    title: "基础入门",
    color: "#3B82F6",
    bg: "bg-blue-50 dark:bg-blue-950/30",
    border: "border-blue-200 dark:border-blue-800",
    dot: "bg-blue-500",
    lessons: [
      { id: "L01", name: "什么是大语言模型", tag: "零基础", motto: "大模型 = 超级学霸" },
      { id: "L02", name: "Transformer 架构", tag: "核心", motto: "注意力 = 知道该看哪" },
      { id: "L03", name: "环境搭建与工具链", tag: "实操", motto: "磨刀不误砍柴工" },
      { id: "L04", name: "MedicalGPT 全景", tag: "地图", motto: "先看地图再走路" },
    ],
  },
  {
    phase: "Phase 2",
    title: "训练 Pipeline 核心",
    color: "#10B981",
    bg: "bg-emerald-50 dark:bg-emerald-950/30",
    border: "border-emerald-200 dark:border-emerald-800",
    dot: "bg-emerald-500",
    lessons: [
      { id: "L05", name: "增量预训练 PT", tag: "核心", motto: "让模型读医学论文" },
      { id: "L06", name: "有监督微调 SFT", tag: "核心", motto: "读书不够，得做题" },
      { id: "L07", name: "LoRA 与 QLoRA", tag: "核心", motto: "只需贴便签纸" },
      { id: "L08", name: "奖励模型 RM", tag: "进阶", motto: "训练老师来打分" },
      { id: "L09", name: "PPO 与 RLHF", tag: "进阶", motto: "用评分改进回答" },
      { id: "L10", name: "DPO 直接偏好优化", tag: "进阶", motto: "直接从对比中学" },
      { id: "L11", name: "ORPO 与 GRPO", tag: "前沿", motto: "一步到位学对齐" },
    ],
  },
  {
    phase: "Phase 3",
    title: "数据与工程",
    color: "#F59E0B",
    bg: "bg-amber-50 dark:bg-amber-950/30",
    border: "border-amber-200 dark:border-amber-800",
    dot: "bg-amber-500",
    lessons: [
      { id: "L12", name: "医疗数据集详解", tag: "数据", motto: "数据决定上限" },
      { id: "L13", name: "数据处理与质量", tag: "数据", motto: "Garbage in, out" },
      { id: "L14", name: "分布式训练", tag: "工程", motto: "八张卡一起上" },
      { id: "L15", name: "评估与部署", tag: "工程", motto: "不上线等于白训" },
    ],
  },
  {
    phase: "Phase 4",
    title: "实战进阶",
    color: "#8B5CF6",
    bg: "bg-purple-50 dark:bg-purple-950/30",
    border: "border-purple-200 dark:border-purple-800",
    dot: "bg-purple-500",
    lessons: [
      { id: "L16", name: "Colab 全流程实战", tag: "实战", motto: "动手才是真功夫" },
      { id: "L17", name: "RAG 检索增强", tag: "应用", motto: "先查资料再回答" },
      { id: "L18", name: "源码逐行精读", tag: "深入", motto: "读源码分水岭" },
    ],
  },
  {
    phase: "Phase 5",
    title: "面试冲刺",
    color: "#EF4444",
    bg: "bg-red-50 dark:bg-red-950/30",
    border: "border-red-200 dark:border-red-800",
    dot: "bg-red-500",
    lessons: [
      { id: "L19", name: "简历包装", tag: "求职", motto: "STAR + 量化数据" },
      { id: "L20", name: "面试通关 100+题", tag: "求职", motto: "全部答案都在这" },
    ],
  },
];

export default function LearningTimeline() {
  return (
    <div className="w-full max-w-3xl mx-auto">
      {PHASES.map((phase, pi) => (
        <motion.div
          key={phase.phase}
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ delay: pi * 0.1, duration: 0.5 }}
          className="mb-8"
        >
          {/* Phase header */}
          <div className="flex items-center gap-3 mb-4">
            <div className={`w-10 h-10 rounded-xl ${phase.dot} flex items-center justify-center`}>
              <span className="text-white text-sm font-bold">{pi + 1}</span>
            </div>
            <div>
              <span className="text-xs font-mono tracking-wider uppercase" style={{ color: phase.color }}>
                {phase.phase}
              </span>
              <h3 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">{phase.title}</h3>
            </div>
          </div>

          {/* Lessons */}
          <div className="ml-5 border-l-2 pl-6" style={{ borderColor: phase.color + "40" }}>
            {phase.lessons.map((lesson, li) => (
              <motion.div
                key={lesson.id}
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: li * 0.05 }}
                className={`relative mb-3 p-3 rounded-lg ${phase.bg} border ${phase.border} hover:scale-[1.02] transition-transform cursor-default`}
              >
                {/* Timeline dot */}
                <div
                  className={`absolute -left-[33px] top-4 w-3 h-3 rounded-full ${phase.dot} ring-2 ring-white dark:ring-zinc-950`}
                />

                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold" style={{ color: phase.color }}>{lesson.id}</span>
                      <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">{lesson.name}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                        style={{ backgroundColor: phase.color + "20", color: phase.color }}>
                        {lesson.tag}
                      </span>
                    </div>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1 italic">&ldquo;{lesson.motto}&rdquo;</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
