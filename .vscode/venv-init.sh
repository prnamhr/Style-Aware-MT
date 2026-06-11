[ -f ~/.bashrc ] && source ~/.bashrc

# Re-enable VS Code shell integration (the blue/red command-status dots).
# A custom --init-file disables VS Code's automatic injection, so source it manually.
if [ "$TERM_PROGRAM" = "vscode" ] && command -v code >/dev/null 2>&1; then
  . "$(code --locate-shell-integration-path bash)"
fi

source "$(dirname "${BASH_SOURCE[0]}")/../.venv/bin/activate"
