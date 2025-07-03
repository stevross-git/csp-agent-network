# File: scripts/setup-git-secrets.sh
#!/bin/bash
set -euo pipefail

echo "🔒 Setting up git-secrets..."

# Install git-secrets if not present
if ! command -v git-secrets &> /dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git-secrets
    else
        git clone https://github.com/awslabs/git-secrets.git
        cd git-secrets && sudo make install && cd .. && rm -rf git-secrets
    fi
fi

# Initialize git-secrets for the repo
git secrets --install
git secrets --register-aws

# Add custom patterns for Ultimate Agent
git secrets --add 'JWT_[A-Z_]*_SECRET'
git secrets --add 'AZURE_CLIENT_SECRET'
git secrets --add 'DB_PASSWORD'
git secrets --add 'REDIS_PASSWORD'
git secrets --add 'private_key'
git secrets --add 'BEGIN RSA PRIVATE KEY'
git secrets --add 'BEGIN OPENSSH PRIVATE KEY'
git secrets --add 'BEGIN DSA PRIVATE KEY'
git secrets --add 'BEGIN EC PRIVATE KEY'
git secrets --add 'BEGIN PGP PRIVATE KEY'
git secrets --add 'bearer [a-zA-Z0-9_\-\.=]+'
git secrets --add 'api[_\-]?key[_\-]?[a-zA-Z0-9_\-]+'

# Add allowed patterns (for documentation/examples)
git secrets --add --allowed 'example\.com'
git secrets --add --allowed 'localhost'
git secrets --add --allowed '\$\{[A-Z_]+\}'
git secrets --add --allowed 'your-.*-here'
git secrets --add --allowed '<.*>'

echo "✅ git-secrets configured!"
