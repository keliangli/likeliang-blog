# NEXUS 技术博客

一个基于 Hugo + Giscus + GitHub Pages 的静态技术博客。

## 特性

- 🚀 **Hugo 静态生成** - 极速构建，SEO 友好
- 💬 **Giscus 评论** - 基于 GitHub Discussions 的免费评论系统
- 🎨 **赛博朋克主题** - 深色科技感设计
- 📱 **响应式布局** - 完美适配各种设备
- 🌍 **GitHub Pages 托管** - 免费、稳定、全球 CDN

## 本地开发

### 前置要求

- [Hugo Extended](https://gohugo.io/installation/) (v0.145.0+)
- [Git](https://git-scm.com/)

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/nexus-blog.git
cd nexus-blog
```

2. 安装主题
```bash
git submodule update --init --recursive
```

3. 本地预览
```bash
hugo server -D
```

访问 http://localhost:1313 查看效果。

## 创建新文章

```bash
hugo new content posts/文章标题.md
```

## 部署

博客使用 GitHub Actions 自动部署到 GitHub Pages：

1. 推送代码到 `main` 分支
2. GitHub Actions 自动构建并部署
3. 访问 `https://yourusername.github.io/nexus-blog`

## Giscus 评论配置

1. 确保仓库已开启 Discussions 功能
2. 访问 [Giscus 配置页面](https://giscus.app/zh-CN)
3. 输入你的仓库信息，获取配置参数
4. 更新 `hugo.toml` 中的 Giscus 配置：

```toml
[params.giscus]
  repo = "yourusername/nexus-blog"
  repoId = "YOUR_REPO_ID"
  category = "Announcements"
  categoryId = "YOUR_CATEGORY_ID"
```

## 目录结构

```
nexus-blog/
├── archetypes/          # 文章模板
├── assets/              # 资源文件
├── content/             # 博客内容
│   ├── posts/          # 文章
│   └── about/          # 关于页面
├── layouts/             # HTML 模板
│   └── partials/       # 模板片段
│       ├── giscus.html # Giscus 评论
│       └── head-additions.html # 自定义样式
├── static/              # 静态资源
├── themes/              # 主题
│   └── ananke/         # Ananke 主题
├── .github/
│   └── workflows/
│       └── deploy.yml  # GitHub Actions 部署
├── hugo.toml           # Hugo 配置
└── README.md           # 本文件
```

## 自定义

### 修改主题颜色

编辑 `layouts/partials/head-additions.html` 中的 CSS 变量。

### 添加新页面

```bash
hugo new content 页面名称/_index.md
```

### 配置菜单

在 `hugo.toml` 中修改 `[menu]` 部分。

## 技术栈

- [Hugo](https://gohugo.io/) - 静态网站生成器
- [Ananke](https://github.com/theNewDynamic/gohugo-theme-ananke) - Hugo 主题
- [Giscus](https://giscus.app/) - 基于 GitHub Discussions 的评论系统
- [GitHub Pages](https://pages.github.com/) - 静态网站托管
- [GitHub Actions](https://github.com/features/actions) - CI/CD

## License

MIT License © 2024 NEXUS
