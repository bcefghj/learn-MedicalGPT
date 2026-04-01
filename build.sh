#!/bin/bash
# build.sh — 生成 HTML 和 PDF 格式的文档
# 使用方法: chmod +x build.sh && ./build.sh

set -e

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="$PROJ_DIR/docs"

mkdir -p "$DOCS_DIR"

echo "=== 开始构建文档 ==="

# 收集所有 Markdown 文件（按顺序）
MD_FILES=(
    "$PROJ_DIR/README.md"
    "$PROJ_DIR/resources/README.md"
    "$PROJ_DIR/job-market/README.md"
    "$PROJ_DIR/interview/README.md"
    "$PROJ_DIR/minimind/01-项目全景.md"
    "$PROJ_DIR/minimind/02-从零搭建.md"
    "$PROJ_DIR/minimind/03-源码精读.md"
    "$PROJ_DIR/minimind/04-简历包装.md"
    "$PROJ_DIR/minimind/05-STAR面试稿.md"
    "$PROJ_DIR/minimind/06-面试问答100题.md"
    "$PROJ_DIR/cheatsheets/README.md"
)

# 生成合并的 Markdown
echo "--- 合并 Markdown ---"
MERGED="$DOCS_DIR/learn-MedicalGPT-全文.md"
echo "" > "$MERGED"
for f in "${MD_FILES[@]}"; do
    if [ -f "$f" ]; then
        cat "$f" >> "$MERGED"
        printf "\n\n---\n\n" >> "$MERGED"
    fi
done
echo "  已生成: $MERGED"

# 生成 HTML
echo "--- 生成 HTML ---"
if command -v pandoc &> /dev/null; then
    pandoc "$MERGED" \
        -o "$DOCS_DIR/learn-MedicalGPT-全文.html" \
        --standalone \
        --toc \
        --toc-depth=2 \
        --metadata title="Learn MedicalGPT — 从零基础到写进简历" \
        -V lang=zh-CN
    echo "  已生成: $DOCS_DIR/learn-MedicalGPT-全文.html"
else
    echo "  pandoc 未安装，跳过 HTML 生成"
    echo "  安装方法: brew install pandoc"
fi

# 生成 PDF（需要额外安装 PDF 引擎）
echo "--- 生成 PDF ---"
if command -v pandoc &> /dev/null; then
    if command -v xelatex &> /dev/null; then
        pandoc "$MERGED" \
            -o "$DOCS_DIR/learn-MedicalGPT-全文.pdf" \
            --pdf-engine=xelatex \
            -V CJKmainfont="PingFang SC" \
            -V geometry:margin=1in \
            --toc \
            --metadata title="Learn MedicalGPT — 从零基础到写进简历" \
            && echo "  已生成: $DOCS_DIR/learn-MedicalGPT-全文.pdf" \
            || echo "  PDF 生成失败，请检查中文字体配置"
    elif command -v wkhtmltopdf &> /dev/null; then
        pandoc "$MERGED" \
            -o "$DOCS_DIR/learn-MedicalGPT-全文.pdf" \
            --pdf-engine=wkhtmltopdf \
            --toc \
            --metadata title="Learn MedicalGPT — 从零基础到写进简历" \
            && echo "  已生成: $DOCS_DIR/learn-MedicalGPT-全文.pdf" \
            || echo "  PDF 生成失败"
    else
        echo "  未找到 PDF 引擎（xelatex 或 wkhtmltopdf）"
        echo "  安装方法："
        echo "    macOS: brew install --cask mactex  # 推荐，支持中文"
        echo "    macOS: brew install --cask wkhtmltopdf  # 轻量替代"
        echo "    Ubuntu: sudo apt install texlive-xetex texlive-lang-chinese"
        echo ""
        echo "  替代方案：用浏览器打开 HTML 文件，Ctrl+P 打印为 PDF"
    fi
fi

echo ""
echo "=== 构建完成 ==="
echo "输出目录: $DOCS_DIR/"
ls -lh "$DOCS_DIR/"
