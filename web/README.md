# 🌐 Learn MedicalGPT — 交互式学习平台

带动画和可视化的 MedicalGPT 学习网页。

## 功能特色

- 🎬 **训练 Pipeline 动画** — 逐步演示 PT → SFT → RLHF/DPO 全流程
- 🧩 **LoRA 原理动画** — 矩阵分解的直觉可视化
- ⚖️ **RLHF vs DPO 对比** — 两种对齐方法并排动画对比
- 📊 **算法对比面板** — PPO/DPO/ORPO/GRPO 交互式对比
- 🗺️ **学习路线时间轴** — 5阶段20节课的可视化路线图
- 🌙 **暗色模式** — 深色/浅色主题自适应

## 技术栈

- [Next.js](https://nextjs.org/) — React 框架
- [framer-motion](https://www.framer.com/motion/) — 动画库
- [Tailwind CSS v4](https://tailwindcss.com/) — CSS 框架
- [Lucide React](https://lucide.dev/) — 图标库

## 快速启动

```sh
cd web
npm install
npm run dev      # 访问 http://localhost:3000
```

## 构建静态部署

```sh
npm run build    # 输出到 out/ 目录
```

可部署到 GitHub Pages / Vercel / Netlify。
