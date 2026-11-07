# spunup installer for Windows (PowerShell):  irm https://raw.githubusercontent.com/jthweny/spunup/main/install/install.ps1 | iex
# Installs everything inside %USERPROFILE%\.spunup. No admin rights needed. Uses your Node.js
# if it is new enough (22+); otherwise downloads a private copy that nothing else uses.
# Run in its own scope: under "irm | iex", "exit" would close the PowerShell window before the message can be
# read, and settings like $ErrorActionPreference would stay changed in it.
& {
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Windows PowerShell's progress bar makes the Node download crawl
$NodeVersion = "v26.13.0"   # current Node.js LTS (checked 2026-11-06)
$Pkg = if ($env:SPUNUP_PACKAGE) { $env:SPUNUP_PACKAGE } else { "spunup" }
$Root = if ($env:SPUNUP_HOME) { $env:SPUNUP_HOME } else { Join-Path $env:USERPROFILE ".spunup" }
$Bin = Join-Path $Root "bin"

function Fail($m) { Write-Host "spunup install failed: $m" -ForegroundColor Red; throw "spunup install failed" }

# 1. CPU
$arch = switch ($env:PROCESSOR_ARCHITECTURE) { "AMD64" { "x64" } "ARM64" { "arm64" } default { "" } }
if (-not $arch) { Fail "unsupported CPU type $($env:PROCESSOR_ARCHITECTURE)." }
Write-Host "spunup: installing for Windows ($arch)"
New-Item -ItemType Directory -Force -Path $Root, $Bin | Out-Null

# 2. Node.js 22 or newer
$node = $null
$have = Get-Command node -ErrorAction SilentlyContinue
if ($have) {
  $major = [int]((& node -p "process.versions.node.split('.')[0]") 2>$null)
  if ($major -ge 22) { $node = $have.Source; Write-Host "Using your Node.js $(& node -v)" }
}
if (-not $node) {
  $dir = Join-Path $Root "node"
  if (-not (Test-Path (Join-Path $dir "node.exe"))) {
    Write-Host "Downloading a private copy of Node.js $NodeVersion (only spunup uses it)..."
    $name = "node-$NodeVersion-win-$arch"
    $zip = Join-Path $Root "node.zip"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing "https://nodejs.org/dist/$NodeVersion/$name.zip" -OutFile $zip
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    Expand-Archive -Path $zip -DestinationPath $Root -Force
    Rename-Item (Join-Path $Root $name) $dir
    Remove-Item $zip
  }
  $node = Join-Path $dir "node.exe"
}
$nodeDir = Split-Path $node
$npm = Join-Path $nodeDir "npm.cmd"
if (-not (Test-Path $npm)) { $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source }
if (-not $npm) { Fail "npm was not found next to Node.js." }

# 3. Install the CLI into %USERPROFILE%\.spunup\app
Write-Host "Installing spunup..."
$env:PATH = "$nodeDir;$env:PATH"
& $npm install --global --prefix (Join-Path $Root "app") --no-fund --no-audit --no-update-notifier --loglevel=error $Pkg | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "npm could not install $Pkg." }
$cli = Join-Path $Root "app\node_modules\spunup\dist\cli.js"
if (-not (Test-Path $cli)) { Fail "the spunup files were not found after installing." }

# 4. The command. If "spunup" already belongs to something else, use "spunup-ai" instead.
$name = "spunup"
$existing = Get-Command spunup -ErrorAction SilentlyContinue
if ($existing -and ($existing.Source -ne (Join-Path $Bin "spunup.cmd"))) {
  $name = "spunup-ai"
  Write-Host "Note: a different program called ""spunup"" already exists ($($existing.Source))."
  Write-Host "      So this one is installed as ""spunup-ai""."
}
# Paths relative to the .cmd file (%~dp0) keep it plain ASCII, so a profile folder like C:\Users\José still
# works, and Node is found when the command runs: spunup's private copy, else the Node on PATH.
$lines = @('@echo off', 'set "SPUNUP_CLI=%~dp0..\app\node_modules\spunup\dist\cli.js"')
if ($env:SPUNUP_HOME) { $lines += 'set "SPUNUP_HOME=%~dp0.."' }
$lines += @('if exist "%~dp0..\node\node.exe" (', '  "%~dp0..\node\node.exe" "%SPUNUP_CLI%" %*', ') else (', '  node "%SPUNUP_CLI%" %*', ')')
Set-Content -Path (Join-Path $Bin "$name.cmd") -Encoding ASCII -Value ($lines -join "`r`n")

# 5. Make the command work in new terminals (user PATH, no admin needed).
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not ($userPath -split ";" | Where-Object { $_ -eq $Bin })) {
  [Environment]::SetEnvironmentVariable("Path", "$Bin;$userPath", "User")
}
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
  Write-Host "Tip: for Vast.ai and RunPod, turn on 'OpenSSH Client' in Settings > System > Optional features."
}
# "irm | iex" runs in this window, so the command works here straight away too.
if (-not ($env:PATH -split ";" | Where-Object { $_ -eq $Bin })) { $env:PATH = "$Bin;$env:PATH" }
Write-Host ""
Write-Host "Done. Type:  $name"
Write-Host "(Terminal windows that were already open need to be reopened first.)"
}
