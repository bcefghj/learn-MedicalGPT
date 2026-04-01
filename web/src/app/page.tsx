"use client";

import { motion } from "framer-motion";
import { Github, BookOpen, GraduationCap, Sparkles } from "lucide-react";
import TrainingPipeline from "@/components/visualizations/training-pipeline";
import LoRAAnimation from "@/components/visualizations/lora-animation";
import RLHFvsDPO from "@/components/visualizations/rlhf-vs-dpo";
import AlgorithmCompare from "@/components/visualizations/algorithm-compare";
import LearningTimeline from "@/components/timeline/learning-timeline";

export default function Home() {
  return (
    <main className="min-h-screen bg-white dark:bg-zinc-950">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-emerald-50 dark:from-zinc-950 dark:via-zinc-950 dark:to-zinc-900" />
        <div className="absolute inset-0">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 rounded-full bg-blue-400/20"
              style={{ left: `${Math.random() * 100}%`, top: `${Math.random() * 100}%` }}
              animate={{ opacity: [0, 0.5, 0], scale: [0, 1.5, 0] }}
              transition={{ duration: 3 + Math.random() * 3, repeat: Infinity, delay: Math.random() * 3 }}
            />
          ))}
        </div>

        <div className="relative max-w-5xl mx-auto px-6 py-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 text-sm font-medium mb-6"
              animate={{ scale: [1, 1.02, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Sparkles size={14} />
              从零基础到面试通关
            </motion.div>

            <h1 className="text-4xl md:text-6xl font-black tracking-tight">
              <span className="text-zinc-900 dark:text-zinc-100">Learn </span>
              <span className="bg-gradient-to-r from-blue-600 via-emerald-500 to-purple-600 bg-clip-text text-transparent">
                MedicalGPT
              </span>
            </h1>

            <p className="mt-4 text-lg text-zinc-500 dark:text-zinc-400 max-w-2xl mx-auto">
              <strong className="text-zinc-700 dark:text-zinc-300">20 节课</strong>，彻底搞懂医疗大模型训练全流程
              <br />
              PT → SFT → LoRA → RLHF → DPO → GRPO
              <br />
              <span className="text-blue-500">学完写进简历，面试对答如流</span>
            </p>

            <div className="flex items-center justify-center gap-4 mt-8">
              <a
                href="https://github.com/bcefghj/learn-MedicalGPT"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 font-semibold text-sm hover:scale-105 transition-transform"
              >
                <Github size={16} />
                GitHub
              </a>
              <a
                href="#pipeline"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-500 text-white font-semibold text-sm hover:scale-105 transition-transform"
              >
                <BookOpen size={16} />
                开始学习
              </a>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="grid grid-cols-4 gap-4 mt-12 max-w-xl mx-auto"
          >
            {[
              { num: "20", label: "节课程" },
              { num: "12K+", label: "行内容" },
              { num: "100+", label: "面试题" },
              { num: "5", label: "阶段" },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-2xl font-black text-zinc-900 dark:text-zinc-100">{stat.num}</div>
                <div className="text-xs text-zinc-400">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Training Pipeline Animation */}
      <section id="pipeline" className="max-w-5xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              🏥 训练 Pipeline 全流程动画
            </h2>
            <p className="text-sm text-zinc-500 mt-2">点击播放按钮，逐步理解从基座模型到医疗问答的完整训练过程</p>
          </div>
          <TrainingPipeline />
        </motion.div>
      </section>

      {/* LoRA Animation */}
      <section className="max-w-5xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              🧩 LoRA 低秩适配动画
            </h2>
            <p className="text-sm text-zinc-500 mt-2">面试高频考点！用动画直觉理解为什么 LoRA 能用 0.4% 的参数实现微调</p>
          </div>
          <LoRAAnimation />
        </motion.div>
      </section>

      {/* RLHF vs DPO */}
      <section className="max-w-5xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              ⚖️ RLHF vs DPO 对比动画
            </h2>
            <p className="text-sm text-zinc-500 mt-2">两种偏好对齐方法，一个复杂精细一个简单高效</p>
          </div>
          <RLHFvsDPO />
        </motion.div>
      </section>

      {/* Algorithm Compare */}
      <section className="max-w-5xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              📊 对齐算法全家桶
            </h2>
            <p className="text-sm text-zinc-500 mt-2">PPO / DPO / ORPO / GRPO — 点击卡片查看详细对比</p>
          </div>
          <AlgorithmCompare />
        </motion.div>
      </section>

      {/* Learning Timeline */}
      <section className="max-w-5xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="text-center mb-10">
            <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              🗺️ 学习路线图
            </h2>
            <p className="text-sm text-zinc-500 mt-2">5 个阶段，20 节课，从零基础到面试通关的完整路径</p>
          </div>
          <LearningTimeline />
        </motion.div>
      </section>

      {/* CTA Footer */}
      <section className="max-w-5xl mx-auto px-6 py-16">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="rounded-2xl bg-gradient-to-r from-blue-600 to-purple-600 p-8 text-center text-white"
        >
          <GraduationCap size={40} className="mx-auto mb-4 opacity-80" />
          <h2 className="text-2xl font-bold mb-2">准备好了吗？</h2>
          <p className="text-blue-100 mb-6">
            学完这 20 节课，你能自信地在面试中讲清楚 MedicalGPT 的每一个技术细节。
          </p>
          <a
            href="https://github.com/bcefghj/learn-MedicalGPT"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-white text-blue-600 font-bold text-sm hover:scale-105 transition-transform"
          >
            <Github size={16} />
            去 GitHub 开始学习
          </a>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-100 dark:border-zinc-800 py-8 text-center text-xs text-zinc-400">
        <p>Learn MedicalGPT — 从零基础到面试通关的医疗大模型学习指南</p>
        <p className="mt-1">
          基于{" "}
          <a href="https://github.com/shibing624/MedicalGPT" className="underline hover:text-blue-500" target="_blank" rel="noopener noreferrer">
            shibing624/MedicalGPT
          </a>
          {" "}项目 | MIT License
        </p>
      </footer>
    </main>
  );
}
