#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 <source_env_dir> <target_dir>

Convert an OpenEnv environment from the monorepo to a standalone repository.

Arguments:
  source_env_dir   Path to existing environment (e.g., envs/echo_env)
  target_dir       Path for new standalone environment (e.g., ~/my_envs/echo_env_standalone)

Example:
  $0 envs/echo_env ~/my_envs/echo_env_standalone

The script will:
  1. Copy environment files to target directory
  2. Convert requirements.txt to pyproject.toml
  3. Add HuggingFace frontmatter to README
  4. Update Dockerfile for standalone builds
  5. Initialize git repository
  6. Generate uv.lock for dependencies

EOF
    exit 1
}

# Check arguments
if [ $# -ne 2 ]; then
    usage
fi

SOURCE_DIR="$1"
TARGET_DIR="$2"

# Validate source directory
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Check if it looks like an OpenEnv environment (either has openenv.yaml or server/ directory)
if [ ! -f "$SOURCE_DIR/openenv.yaml" ] && [ ! -d "$SOURCE_DIR/server" ]; then
    print_error "Not an OpenEnv environment (missing openenv.yaml and server/): $SOURCE_DIR"
    exit 1
fi

# Warn if it's a legacy environment
if [ ! -f "$SOURCE_DIR/openenv.yaml" ]; then
    print_warning "Legacy environment detected (no openenv.yaml) - will create one"
fi

# Extract environment name from source directory
ENV_NAME=$(basename "$SOURCE_DIR")
if [[ "$ENV_NAME" == *"_env" ]]; then
    BASE_NAME="${ENV_NAME%_env}"
else
    BASE_NAME="$ENV_NAME"
fi

# Convert to class name (capitalize first letter of each word)
CLASS_NAME=$(echo "$BASE_NAME" | sed -r 's/(^|_)([a-z])/\U\2/g')

print_header "OpenEnv Environment Conversion"
print_info "Source: $SOURCE_DIR"
print_info "Target: $TARGET_DIR"
print_info "Environment: $ENV_NAME"
print_info "Class Name: $CLASS_NAME"

# Confirm with user
echo ""
read -p "Continue with conversion? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Conversion cancelled"
    exit 0
fi

# Step 1: Copy environment files
print_header "Step 1: Copying Environment Files"

if [ -d "$TARGET_DIR" ]; then
    print_warning "Target directory already exists: $TARGET_DIR"
    read -p "Remove existing directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET_DIR"
        print_success "Removed existing directory"
    else
        print_error "Cannot proceed with existing directory"
        exit 1
    fi
fi

mkdir -p "$TARGET_DIR"
cp -r "$SOURCE_DIR"/* "$TARGET_DIR/"
print_success "Copied environment files to $TARGET_DIR"

cd "$TARGET_DIR"

# Step 1.5: Create openenv.yaml if missing (legacy environments)
if [ ! -f "openenv.yaml" ]; then
    print_header "Step 1.5: Creating openenv.yaml"
    print_info "Legacy environment detected - creating openenv.yaml"
    
    cat > openenv.yaml << EOF
name: ${ENV_NAME}
version: "0.1.0"
description: "${BASE_NAME^} environment for OpenEnv"
action: ${CLASS_NAME}Action
observation: ${CLASS_NAME}Observation
EOF
    
    print_success "Created openenv.yaml"
fi

# Step 2: Convert requirements.txt to pyproject.toml
print_header "Step 2: Setting Up pyproject.toml"

if [ -f "pyproject.toml" ]; then
    print_success "pyproject.toml already exists"
else
    print_info "Creating pyproject.toml"
    
    # Collect dependencies from requirements.txt if it exists
    DEPS=""
    if [ -f "server/requirements.txt" ]; then
        print_info "Converting server/requirements.txt to dependencies"
        while IFS= read -r line; do
            # Skip empty lines and comments
            if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
                DEPS="${DEPS}    \"${line}\",\n"
            fi
        done < "server/requirements.txt"
    fi
    
    # Always add openenv runtime
    DEPS="${DEPS}    \"openenv[core]>=0.2.0\","
    
    # Create pyproject.toml
    cat > pyproject.toml << EOF
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-${ENV_NAME}"
version = "0.1.0"
description = "${BASE_NAME^} Environment for OpenEnv"
requires-python = ">=3.10"
dependencies = [
$(echo -e "$DEPS")
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "ipykernel>=6.29.5",
]

[project.scripts]
server = "${ENV_NAME}.server.app:main"

[tool.setuptools]
packages = ["${ENV_NAME}"]

[tool.setuptools.package-data]
${ENV_NAME} = ["**/*.yaml", "**/*.yml"]
EOF
    
    print_success "Created pyproject.toml"
fi

# Step 3: Add HuggingFace frontmatter to README
print_header "Step 3: Updating README with HuggingFace Frontmatter"

if [ -f "README.md" ]; then
    # Check if frontmatter already exists
    if head -n 1 "README.md" | grep -q "^---$"; then
        print_success "README.md already has frontmatter"
    else
        print_info "Adding HuggingFace frontmatter to README.md"
        
        # Create temporary file with frontmatter
        cat > README.tmp << 'EOF'
---
title: __TITLE__ Environment Server
emoji: 🎮
colorFrom: '#00C9FF'
colorTo: '#1B2845'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

EOF
        
        # Replace placeholder with actual title
        TITLE="${BASE_NAME^}"
        sed -i "s/__TITLE__/${TITLE}/g" README.tmp
        
        # Append original README content
        cat "README.md" >> README.tmp
        mv README.tmp "README.md"
        
        print_success "Added HuggingFace frontmatter to README.md"
    fi
else
    print_warning "README.md not found - creating basic one"
    cat > README.md << EOF
---
title: ${BASE_NAME^} Environment Server
emoji: 🎮
colorFrom: '#00C9FF'
colorTo: '#1B2845'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# ${BASE_NAME^} Environment

An OpenEnv environment.

## Quick Start

\`\`\`python
from ${ENV_NAME} import ${CLASS_NAME}Action, ${CLASS_NAME}Env

# Create environment from Docker image
env = ${CLASS_NAME}Env.from_docker_image("${ENV_NAME}:latest")

# Reset
result = env.reset()

# Step
result = env.step(${CLASS_NAME}Action(...))

# Clean up
env.close()
\`\`\`

## Building

\`\`\`bash
openenv build
\`\`\`

## Deploying

\`\`\`bash
openenv push
\`\`\`
EOF
    print_success "Created README.md"
fi

# Step 4: Update Dockerfile
print_header "Step 4: Updating Dockerfile for Standalone Builds"

if [ -f "server/Dockerfile" ]; then
    # Check if Dockerfile already has standalone pattern
    if grep -q "pip install.*-e \." "server/Dockerfile"; then
        print_success "Dockerfile already configured for standalone builds"
    else
        print_info "Updating Dockerfile for standalone builds"
        
        # Create updated Dockerfile
        cat > server/Dockerfile.new << 'EOF'
# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app/env

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# Run the server
CMD ["python", "-m", "uvicorn", "__ENV_NAME__.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
        
        # Replace placeholder
        sed -i "s/__ENV_NAME__/${ENV_NAME}/g" server/Dockerfile.new
        
        # Backup original and replace
        mv server/Dockerfile server/Dockerfile.backup
        mv server/Dockerfile.new server/Dockerfile
        
        print_success "Updated Dockerfile (backup saved as server/Dockerfile.backup)"
    fi
else
    print_warning "server/Dockerfile not found"
fi

# Step 5: Ensure app.py has main() function
print_header "Step 5: Checking server/app.py"

if [ -f "server/app.py" ]; then
    if grep -q "def main()" "server/app.py"; then
        print_success "server/app.py has main() function"
    else
        print_warning "server/app.py missing main() function - adding it"
        cat >> server/app.py << 'EOF'


def main():
    """Main entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
EOF
        print_success "Added main() function to server/app.py"
    fi
else
    print_warning "server/app.py not found"
fi

# Step 6: Create .gitignore
print_header "Step 6: Creating .gitignore"

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Environment outputs
outputs/
logs/
*.log

# Build artifacts
*.backup
EOF

print_success "Created .gitignore"

# Step 7: Initialize git repository
print_header "Step 7: Initializing Git Repository"

if [ -d ".git" ]; then
    print_warning "Git repository already initialized"
else
    git init
    git add .
    git commit -m "Initial commit: Converted from OpenEnv monorepo

Environment: ${ENV_NAME}
Converted using convert_env.sh script"
    
    print_success "Initialized git repository"
fi

# Step 8: Generate uv.lock (if uv is available)
print_header "Step 8: Generating Dependency Lock File"

if command -v uv &> /dev/null; then
    print_info "Generating uv.lock..."
    if uv lock; then
        print_success "Generated uv.lock"
    else
        print_warning "Failed to generate uv.lock - you may need to fix pyproject.toml dependencies"
    fi
else
    print_warning "uv not found - skipping lock file generation"
    print_info "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# Final summary
print_header "Conversion Complete!"

echo ""
print_success "Environment converted successfully!"
echo ""
print_info "Next steps:"
echo ""
echo "  1. Review the converted files:"
echo "     cd $TARGET_DIR"
echo ""
echo "  2. Install dependencies:"
echo "     uv pip install -e ."
echo ""
echo "  3. Test locally:"
echo "     uv run server"
echo ""
echo "  4. Validate structure:"
echo "     openenv validate"
echo ""
echo "  5. Build Docker image:"
echo "     openenv build"
echo ""
echo "  6. Deploy to HuggingFace:"
echo "     openenv push"
echo ""
echo "  7. Or push to Docker registry:"
echo "     openenv push --registry docker.io/myusername"
echo ""
print_info "For detailed documentation, see CONVERT.md"
echo ""
