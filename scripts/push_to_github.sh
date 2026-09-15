#!/usr/bin/env bash
# Push project to personal GitHub using github.com-personal SSH alias
# Usage: ./scripts/push_to_github.sh <repo_name_or_url>

set -e

REPO_TARGET="${1:-qwen2.5-coder-0.5b-nl2bash}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git branch -M main
fi

# Determine remote URL format
if [[ "$REPO_TARGET" == git@* ]] || [[ "$REPO_TARGET" == https://* ]]; then
    REMOTE_URL="$REPO_TARGET"
else
    # Default to TECHNOKART or personal user alias
    REMOTE_URL="git@github.com-personal:TECHNOKART/${REPO_TARGET}.git"
fi

echo "Setting remote origin to: $REMOTE_URL"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

echo "Staging files..."
git add .

echo "Committing..."
git commit -m "Initial commit: Qwen2.5-Coder NL2Bash fine-tuning with Unsloth and Colab CLI automation" || echo "Nothing new to commit"

echo "Pushing to $REMOTE_URL (branch: main)..."
git push -u origin main

echo "✅ Successfully pushed to GitHub via github.com-personal!"
