param(
    [string]$RunDate = (Get-Date -Format "yyyy-MM-dd"),
    [int]$LookbackDays = 7,
    [int]$PerTopic = 6,
    [switch]$SkipGitCommit,
    [switch]$AutoPush
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-CurlPath {
    if ($env:GIT_CURL_PATH -and (Test-Path $env:GIT_CURL_PATH)) {
        return $env:GIT_CURL_PATH
    }
    $defaultCurl = "C:\Program Files\Git\mingw64\bin\curl.exe"
    if (Test-Path $defaultCurl) {
        return $defaultCurl
    }
    throw "未找到 curl.exe。请安装 Git for Windows，或设置环境变量 GIT_CURL_PATH。"
}

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Invoke-GitHubApi([string]$Url, [string]$CurlPath) {
    $args = @(
        "-sS",
        "--retry", "2",
        "-H", "User-Agent: keliangli-tech-news-agent",
        "-H", "Accept: application/vnd.github+json"
    )

    if ($env:GITHUB_TOKEN) {
        $args += @("-H", "Authorization: Bearer $($env:GITHUB_TOKEN)")
    }

    $args += $Url
    $response = & $CurlPath @args
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub API 请求失败: $Url"
    }

    return ($response | ConvertFrom-Json)
}

function Escape-MarkdownCell([string]$Text) {
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return "-"
    }
    $clean = $Text -replace "\r?\n", " "
    $clean = $clean -replace "\|", "/"
    return $clean.Trim()
}

function Load-State([string]$StatePath) {
    $state = @{
        lastRun = $null
        repos   = @{}
    }

    if (-not (Test-Path $StatePath)) {
        return $state
    }

    $raw = Get-Content $StatePath -Encoding UTF8 -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $state
    }

    $json = $raw | ConvertFrom-Json
    $state.lastRun = $json.lastRun

    $repos = @{}
    if ($json.repos) {
        foreach ($p in $json.repos.PSObject.Properties) {
            $repos[$p.Name] = @{
                stars = [int]$p.Value.stars
            }
        }
    }
    $state.repos = $repos
    return $state
}

function Save-State([string]$StatePath, [hashtable]$RepoMap, [string]$NowIso) {
    $stateDir = Split-Path $StatePath -Parent
    Ensure-Directory $stateDir

    $orderedRepos = [ordered]@{}
    foreach ($key in ($RepoMap.Keys | Sort-Object)) {
        $orderedRepos[$key] = @{
            stars = [int]$RepoMap[$key].stars
        }
    }

    $payload = @{
        lastRun = $NowIso
        repos   = $orderedRepos
    }

    $payload | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $StatePath
}

function Build-PostContent(
    [string]$RunDate,
    [string]$Timestamp,
    [pscustomobject]$Config,
    [System.Collections.ArrayList]$TopicResults,
    [System.Collections.ArrayList]$AllRepos,
    [int]$NewRepoCount,
    [string]$SinceDate
) {
    $title = "{0} {1}" -f $Config.post.titlePrefix, $RunDate
    $description = "{0} GitHub 热点追踪：大模型、大模型推理、智能体、大模型训练。" -f $RunDate
    $category = $Config.post.category
    $author = $Config.post.author
    $tags = @($Config.post.tags)

    $topStar = $AllRepos | Sort-Object stars -Descending | Select-Object -First 1
    $topGrowth = $AllRepos | Sort-Object deltaStars -Descending | Select-Object -First 1

    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("---")
    [void]$sb.AppendLine(("title: ""{0}""" -f $title))
    [void]$sb.AppendLine(("date: {0}" -f $Timestamp))
    [void]$sb.AppendLine("draft: false")
    [void]$sb.AppendLine(("author: ""{0}""" -f $author))
    [void]$sb.AppendLine(("categories: [""{0}""]" -f $category))
    [void]$sb.AppendLine(("tags: [""{0}""]" -f ($tags -join '", "')))
    [void]$sb.AppendLine(("description: ""{0}""" -f $description))
    [void]$sb.AppendLine("---")
    [void]$sb.AppendLine()
    [void]$sb.AppendLine("## 今日综述")
    [void]$sb.AppendLine()
    [void]$sb.AppendLine(("- 抓取窗口：`{0}` 至 `{1}`（近 {2} 天）" -f $SinceDate, $RunDate, $LookbackDays))
    [void]$sb.AppendLine(("- 抓取范围：`{0}` 个技术方向，共 `{1}` 个热点仓库" -f $TopicResults.Count, $AllRepos.Count))
    [void]$sb.AppendLine(("- 新入榜仓库：`{0}` 个" -f $NewRepoCount))
    if ($topStar) {
        [void]$sb.AppendLine(("- 最高星标：[{0}]({1})（⭐ {2}）" -f $topStar.fullName, $topStar.url, $topStar.stars))
    }
    if ($topGrowth) {
        [void]$sb.AppendLine(("- 增长最快：[{0}]({1})（较上次 +{2}⭐）" -f $topGrowth.fullName, $topGrowth.url, $topGrowth.deltaStars))
    }

    [void]$sb.AppendLine()
    [void]$sb.AppendLine("## 热点解析")
    [void]$sb.AppendLine()

    foreach ($topic in $TopicResults) {
        [void]$sb.AppendLine(("### {0}" -f $topic.name))
        [void]$sb.AppendLine()

        if ($topic.repos.Count -eq 0) {
            [void]$sb.AppendLine("- 本次没有抓到满足条件的仓库。")
            [void]$sb.AppendLine()
            continue
        }

        $topicTopStar = $topic.repos | Sort-Object stars -Descending | Select-Object -First 1
        $topicTopGrowth = $topic.repos | Sort-Object deltaStars -Descending | Select-Object -First 1

        [void]$sb.AppendLine(("- 热度头部：[{0}]({1})（⭐ {2}）" -f $topicTopStar.fullName, $topicTopStar.url, $topicTopStar.stars))
        [void]$sb.AppendLine(("- 增速关注：[{0}]({1})（+{2}⭐）" -f $topicTopGrowth.fullName, $topicTopGrowth.url, $topicTopGrowth.deltaStars))
        [void]$sb.AppendLine()
        [void]$sb.AppendLine("| 仓库 | 星标 | 较上次 | 语言 | 最近更新 | 核心说明 |")
        [void]$sb.AppendLine("|:---|---:|---:|:---:|:---:|:---|")

        foreach ($repo in $topic.repos) {
            $lang = if ([string]::IsNullOrWhiteSpace($repo.language)) { "-" } else { $repo.language }
            $updated = (Get-Date $repo.updatedAt).ToString("yyyy-MM-dd")
            $desc = Escape-MarkdownCell $repo.description
            [void]$sb.AppendLine(("| [{0}]({1}) | {2} | +{3} | {4} | {5} | {6} |" -f $repo.fullName, $repo.url, $repo.stars, $repo.deltaStars, $lang, $updated, $desc))
        }
        [void]$sb.AppendLine()
    }

    [void]$sb.AppendLine("## 信息来源")
    [void]$sb.AppendLine()
    [void]$sb.AppendLine("- GitHub Search API（repositories）")
    [void]$sb.AppendLine("- 说明：本报告由本地智能体自动生成，主要用于热点发现与方向判断。")
    [void]$sb.AppendLine()
    [void]$sb.AppendLine(("> 生成时间：{0}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")))

    return $sb.ToString()
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$configPath = Join-Path $scriptRoot "agent-config.json"

if (-not (Test-Path $configPath)) {
    throw "配置文件不存在: $configPath"
}

$config = Get-Content $configPath -Encoding UTF8 -Raw | ConvertFrom-Json
$curlPath = Get-CurlPath

$stateRoot = if ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA "keliangli-tech-news-agent"
} else {
    Join-Path $repoRoot "data\tech-news-agent"
}
$statePath = Join-Path $stateRoot "state.json"
$state = Load-State $statePath
$stateRepos = $state.repos

$runDateObj = [datetime]::ParseExact($RunDate, "yyyy-MM-dd", [System.Globalization.CultureInfo]::InvariantCulture)
$sinceDate = $runDateObj.AddDays(-$LookbackDays).ToString("yyyy-MM-dd")
$timestamp = "{0}T09:00:00+08:00" -f $RunDate

$topicResults = New-Object System.Collections.ArrayList
$allRepos = New-Object System.Collections.ArrayList
$latestRepoMap = @{}
$newRepoCount = 0

Set-Location $repoRoot

foreach ($topic in $config.topics) {
    $query = $topic.queryTemplate.Replace("{since}", $sinceDate)
    $encodedQuery = [uri]::EscapeDataString($query)
    $url = "https://api.github.com/search/repositories?q=$encodedQuery&sort=stars&order=desc&per_page=$PerTopic"
    $response = Invoke-GitHubApi -Url $url -CurlPath $curlPath

    if (-not $response.items -or $response.items.Count -eq 0) {
        $fallbackQuery = ($query -replace "pushed:>=[0-9\-]+", "").Trim()
        if (-not [string]::IsNullOrWhiteSpace($fallbackQuery) -and $fallbackQuery -ne $query) {
            $encodedFallback = [uri]::EscapeDataString($fallbackQuery)
            $fallbackUrl = "https://api.github.com/search/repositories?q=$encodedFallback&sort=stars&order=desc&per_page=$PerTopic"
            $response = Invoke-GitHubApi -Url $fallbackUrl -CurlPath $curlPath
        }
    }

    $topicRepos = New-Object System.Collections.ArrayList
    foreach ($item in $response.items) {
        $fullName = [string]$item.full_name
        $currentStars = [int]$item.stargazers_count
        $previousStars = 0

        if ($stateRepos.ContainsKey($fullName)) {
            $previousStars = [int]$stateRepos[$fullName].stars
        } else {
            $newRepoCount += 1
        }

        $repoObj = [pscustomobject]@{
            topic      = [string]$topic.name
            fullName   = $fullName
            url        = [string]$item.html_url
            stars      = $currentStars
            deltaStars = [Math]::Max(0, ($currentStars - $previousStars))
            language   = [string]$item.language
            updatedAt  = [string]$item.updated_at
            description = [string]$item.description
        }

        [void]$topicRepos.Add($repoObj)
        [void]$allRepos.Add($repoObj)
        $latestRepoMap[$fullName] = @{
            stars = $currentStars
        }
    }

    [void]$topicResults.Add([pscustomobject]@{
        name  = [string]$topic.name
        repos = $topicRepos
    })
}

if ($allRepos.Count -eq 0) {
    throw "未抓到任何热点仓库，请检查网络、API 配额或查询条件。"
}

$content = Build-PostContent -RunDate $RunDate -Timestamp $timestamp -Config $config -TopicResults $topicResults -AllRepos $allRepos -NewRepoCount $newRepoCount -SinceDate $sinceDate

$postDir = Join-Path $repoRoot "content\tech-news"
Ensure-Directory $postDir
$postPath = Join-Path $postDir ("tech-news-{0}.md" -f $RunDate)
$content | Set-Content -Encoding UTF8 -Path $postPath

Save-State -StatePath $statePath -RepoMap $latestRepoMap -NowIso (Get-Date).ToString("o")

Write-Host "技术新闻已生成: $postPath"
Write-Host ("收集仓库数: {0}, 新入榜: {1}" -f $allRepos.Count, $newRepoCount)

if (-not $SkipGitCommit) {
    & git add -- $postPath
    $changed = (& git status --porcelain -- $postPath)
    if (-not [string]::IsNullOrWhiteSpace($changed)) {
        & git commit -m ("chore: publish 技术新闻 {0}" -f $RunDate) -- $postPath | Out-Host
        if ($AutoPush) {
            & git push origin main | Out-Host
        }
    } else {
        Write-Host "文章内容无变化，未创建新提交。"
    }
}




