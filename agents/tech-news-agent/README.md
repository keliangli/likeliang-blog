# 技术新闻本地智能体

这个智能体会自动收集 GitHub 上与以下主题相关的最新热点仓库，并生成 Hugo 文章：

- 大模型
- 大模型推理技术
- 智能体技术
- 大模型训练技术

生成文章位置：

- `content/tech-news/tech-news-YYYY-MM-DD.md`

文章标题格式：

- `技术新闻 YYYY-MM-DD`

## 1. 运行要求

- Windows PowerShell 5.1+
- Git for Windows（用于 `curl.exe` 与 `git`）
- 已配置好该仓库远程推送权限（可选自动推送）

## 2. 可选环境变量

- `GITHUB_TOKEN`：提升 GitHub API 限额，建议配置
- `GIT_CURL_PATH`：如果 `curl.exe` 不在默认位置，可手动指定

## 3. 手动执行

在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\agents\tech-news-agent\run.ps1
```

仅生成文件，不提交：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\agents\tech-news-agent\run.ps1 -SkipGitCommit
```

生成并自动推送：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\agents\tech-news-agent\run.ps1 -AutoPush
```

## 4. 定时执行（每日）

安装计划任务（默认每日 08:40）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\agents\tech-news-agent\install-schedule.ps1
```

指定执行时间：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\agents\tech-news-agent\install-schedule.ps1 -RunAt "07:30"
```

## 5. 配置说明

配置文件：

- `agents/tech-news-agent/agent-config.json`

你可以在该文件中调整：

- 抓取主题与查询关键词
- 每篇文章的标签、作者名、分类

