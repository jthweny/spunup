#!/bin/sh
# spunup installer for macOS and Linux:  curl -fsSL https://raw.githubusercontent.com/jthweny/spunup/main/install/install.sh | sh
# Installs everything inside ~/.spunup. Never needs sudo. Uses your Node.js if it is new
# enough (22+); otherwise downloads a private copy that nothing else on your computer uses.
set -eu

NODE_VERSION="v26.13.0"          # current Node.js LTS (checked 2026-11-06)
PKG="${SPUNUP_PACKAGE:-spunup-cli}"  # npm package (or a tarball path/URL for testing)
ROOT="${SPUNUP_HOME:-$HOME/.spunup}"
BIN="$ROOT/bin"

say()  { printf '%s\n' "$*"; }
fail() { printf 'spunup install failed: %s\n' "$*" >&2; exit 1; }

fetch() { # fetch URL FILE
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
  else fail "needs curl or wget to download files."; fi
}

# 1. OS and CPU
case "$(uname -s)" in
  Darwin) OS=darwin ;;
  Linux)  OS=linux ;;
  *) fail "this script is for macOS and Linux. On Windows run in PowerShell:  irm https://raw.githubusercontent.com/jthweny/spunup/main/install/install.ps1 | iex" ;;
esac
case "$(uname -m)" in
  x86_64|amd64) ARCH=x64 ;;
  arm64|aarch64) ARCH=arm64 ;;
  *) fail "unsupported CPU type $(uname -m)." ;;
esac
if [ "$OS" = linux ] && ldd --version 2>&1 | grep -qi musl; then
  fail "this Linux uses musl (for example Alpine). Install Node.js 22 or newer with your package manager, then run: npm i -g spunup-cli"
fi
say "spunup: installing for $OS ($ARCH)"
mkdir -p "$ROOT" "$BIN"

# 2. Node.js 22 or newer
NODE=""
if command -v node >/dev/null 2>&1; then
  major=$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)
  if [ "$major" -ge 22 ] 2>/dev/null; then NODE="$(command -v node)"; say "Using your Node.js $(node -v)"; fi
fi
if [ -z "$NODE" ]; then
  dir="$ROOT/node"
  if [ ! -x "$dir/bin/node" ]; then
    say "Downloading a private copy of Node.js $NODE_VERSION (only spunup uses it)…"
    name="node-$NODE_VERSION-$OS-$ARCH"
    tmp="$ROOT/node.tar.gz"
    fetch "https://nodejs.org/dist/$NODE_VERSION/$name.tar.gz" "$tmp"
    rm -rf "$dir" "$ROOT/$name"
    tar -xzf "$tmp" -C "$ROOT"
    mv "$ROOT/$name" "$dir"
    rm -f "$tmp"
  fi
  NODE="$dir/bin/node"
fi
NODEDIR="$(dirname "$NODE")"
NPM="$NODEDIR/npm"
[ -x "$NPM" ] || NPM="$(command -v npm || true)"
[ -n "$NPM" ] || fail "npm was not found next to Node.js."

# 3. Install the spunup CLI into ~/.spunup/app (no sudo, no global npm folder)
say "Installing spunup…"
PATH="$NODEDIR:$PATH" "$NPM" install --global --prefix "$ROOT/app" --no-fund --no-audit --no-update-notifier --loglevel=error "$PKG" >/dev/null \
  || fail "npm could not install $PKG."
CLI="$ROOT/app/lib/node_modules/spunup-cli/dist/cli.js"
[ -f "$CLI" ] || fail "the spunup files were not found after installing."

# 4. The command. If "spunup" already belongs to something else, use "spunup-ai" instead.
NAME=spunup
existing="$(command -v spunup 2>/dev/null || true)"
if [ -n "$existing" ] && [ "$existing" != "$BIN/spunup" ] && [ "$(readlink "$existing" 2>/dev/null)" != "$BIN/spunup" ]; then
  NAME=spunup-ai
  say "Note: a different program called \"spunup\" already exists ($existing)."
  say "      So this one is installed as \"spunup-ai\"."
fi
# Node is looked up when the command runs, so removing or upgrading the Node it was installed with
# (nvm, Homebrew) doesn't break it: the one used now, else spunup's private copy, else any Node 22+.
HOMELINE=""
[ -n "${SPUNUP_HOME:-}" ] && HOMELINE="export SPUNUP_HOME=\"$ROOT\""
cat > "$BIN/$NAME" <<EOF
#!/bin/sh
$HOMELINE
for n in "$NODE" "$ROOT/node/bin/node"; do [ -x "\$n" ] && exec "\$n" "$CLI" "\$@"; done
if command -v node >/dev/null 2>&1 && [ "\$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)" -ge 22 ] 2>/dev/null; then
  exec node "$CLI" "\$@"
fi
echo "spunup: Node.js 22 or newer was not found. Run the spunup installer again." >&2
exit 1
EOF
chmod +x "$BIN/$NAME"

# 5. Make the command work in new terminals.
added=""
line="export PATH=\"$BIN:\$PATH\""
# Your login shell's file is created if missing (zsh on macOS does not read ~/.profile).
case "${SHELL:-}" in
  */zsh)  [ -f "$HOME/.zshrc" ] || : > "$HOME/.zshrc" ;;
  # Bash login shells on macOS read only the first of .bash_profile, .bash_login, .profile: a new
  # .bash_profile would hide an existing .profile, so it is only made when neither of those exists.
  */bash) [ -f "$HOME/.bashrc" ] || : > "$HOME/.bashrc"
          if [ "$OS" = darwin ] && [ ! -f "$HOME/.bash_profile" ] && [ ! -f "$HOME/.bash_login" ] && [ ! -f "$HOME/.profile" ]; then
            printf '[ -f ~/.bashrc ] && . ~/.bashrc\n' > "$HOME/.bash_profile"
          fi ;;
esac
for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.profile"; do
  if [ -f "$rc" ] || [ "$rc" = "$HOME/.profile" ]; then
    if ! grep -qs "$BIN" "$rc"; then printf '\n# spunup\n%s\n' "$line" >> "$rc"; added="$added $rc"; fi
  fi
done
if [ -d "$HOME/.config/fish" ]; then
  f="$HOME/.config/fish/conf.d/spunup.fish"; mkdir -p "$(dirname "$f")"
  [ -f "$f" ] || { echo "fish_add_path $BIN" > "$f"; added="$added $f"; }
fi

# This terminal can't pick up the new PATH, so also link the command into a folder of yours that is
# already on PATH (like ~/.local/bin): then it works here straight away. Never over someone else's file,
# and not for test installs (SPUNUP_HOME), which shouldn't replace the real command.
now=""
if [ -z "${SPUNUP_HOME:-}" ]; then
  for d in "$HOME/.local/bin" "$HOME/bin"; do
    case ":$PATH:" in *":$d:"*) ;; *) continue ;; esac
    [ -d "$d" ] && [ -w "$d" ] || continue
    if { [ ! -e "$d/$NAME" ] && [ ! -L "$d/$NAME" ]; } || [ "$(readlink "$d/$NAME" 2>/dev/null)" = "$BIN/$NAME" ]; then
      ln -sf "$BIN/$NAME" "$d/$NAME" && now=1 && break
    fi
  done
fi

say ""
say "Done. Type:  $NAME"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) if [ -z "$now" ]; then
       say "This terminal does not know about the new command yet. Open a new terminal window,"
       say "or paste this here first:  export PATH=\"$BIN:\$PATH\""
     fi ;;
esac
