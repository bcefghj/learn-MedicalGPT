import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Learn MedicalGPT — 从零基础到面试通关",
  description: "20节课彻底搞懂医疗大模型训练全流程，交互式可视化学习平台",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className="antialiased">{children}</body>
    </html>
  );
}
