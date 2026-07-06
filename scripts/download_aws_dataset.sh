#!/bin/bash

set -e

# ─────────────────────────────────────────
# Colors
# ─────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_step() { echo -e "\n${CYAN}==>${NC} $1"; }
print_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

# ─────────────────────────────────────────
# Step 1: Install AWS CLI if not present
# ─────────────────────────────────────────
print_step "Checking AWS CLI..."

if command -v aws &>/dev/null; then
    print_ok "AWS CLI already installed: $(aws --version 2>&1)"
else
    print_step "Installing AWS CLI v2..."

    TMP_DIR=$(mktemp -d)
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$TMP_DIR/awscliv2.zip"
    unzip -q "$TMP_DIR/awscliv2.zip" -d "$TMP_DIR"

    if [ "$EUID" -eq 0 ]; then
        "$TMP_DIR/aws/install"
    else
        sudo "$TMP_DIR/aws/install"
    fi

    rm -rf "$TMP_DIR"
    print_ok "AWS CLI installed: $(aws --version 2>&1)"
fi

# ─────────────────────────────────────────
# Step 2: Login / configure credentials
# ─────────────────────────────────────────
print_step "AWS Authentication"

echo ""
echo "Choose authentication method:"
echo "  1) Enter AWS credentials manually (Access Key + Secret)"
echo "  2) Use existing profile / already configured"
echo "  3) Public bucket (no credentials needed)"
echo ""
read -rp "Your choice [1/2/3]: " AUTH_CHOICE

NO_SIGN=""

case "$AUTH_CHOICE" in
    1)
        echo ""
        read -rp "AWS Access Key ID:     " AWS_ACCESS_KEY_ID
        read -rsp "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
        echo ""
        read -rp "Default region (e.g. us-east-1): " AWS_REGION
        read -rp "Output format [json/text/table, default: json]: " AWS_OUTPUT
        AWS_OUTPUT="${AWS_OUTPUT:-json}"

        aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
        aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
        aws configure set default.region "$AWS_REGION"
        aws configure set default.output "$AWS_OUTPUT"

        print_ok "Credentials saved."

        # Verify
        print_step "Verifying credentials..."
        if aws sts get-caller-identity &>/dev/null; then
            ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
            USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
            print_ok "Logged in as: $USER_ARN (account: $ACCOUNT)"
        else
            print_err "Credential verification failed. Please check your keys."
            exit 1
        fi
        ;;
    2)
        print_step "Verifying existing credentials..."
        if aws sts get-caller-identity &>/dev/null; then
            ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
            USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
            print_ok "Using existing credentials: $USER_ARN (account: $ACCOUNT)"
        else
            print_err "No valid credentials found. Run 'aws configure' or choose option 1."
            exit 1
        fi
        ;;
    3)
        NO_SIGN="--no-sign-request"
        print_ok "Public bucket mode — no credentials required."
        ;;
    *)
        print_err "Invalid choice."
        exit 1
        ;;
esac

# ─────────────────────────────────────────
# Step 3: Ask for S3 path and local path
# ─────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo "          Download Configuration"
echo "────────────────────────────────────────"
echo ""

read -rp "S3 path (e.g. s3://my-bucket/datasets/mnist/): " S3_PATH

# Normalize: ensure it starts with s3://
if [[ "$S3_PATH" != s3://* ]]; then
    S3_PATH="s3://$S3_PATH"
fi

read -rp "Local destination path (e.g. ./data or /home/user/datasets): " LOCAL_PATH

# Expand ~ if used
LOCAL_PATH="${LOCAL_PATH/#\~/$HOME}"

# Create local dir if it doesn't exist
if [ ! -d "$LOCAL_PATH" ]; then
    echo ""
    read -rp "Directory '$LOCAL_PATH' does not exist. Create it? [Y/n]: " CREATE_DIR
    CREATE_DIR="${CREATE_DIR:-Y}"
    if [[ "$CREATE_DIR" =~ ^[Yy]$ ]]; then
        mkdir -p "$LOCAL_PATH"
        print_ok "Created directory: $LOCAL_PATH"
    else
        print_err "Destination directory does not exist. Aborting."
        exit 1
    fi
fi

# ─────────────────────────────────────────
# Step 4: Preview and confirm
# ─────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Summary"
echo "────────────────────────────────────────"
echo "  Source : $S3_PATH"
echo "  Destination : $LOCAL_PATH"
if [ -n "$NO_SIGN" ]; then
    echo "  Auth   : public (no sign)"
else
    echo "  Auth   : AWS credentials"
fi
echo ""
read -rp "Start download? [Y/n]: " CONFIRM
CONFIRM="${CONFIRM:-Y}"

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ─────────────────────────────────────────
# Step 5: Download
# ─────────────────────────────────────────
print_step "Downloading from $S3_PATH ..."
echo ""

aws s3 cp "$S3_PATH" "$LOCAL_PATH" \
    --recursive \
    $NO_SIGN \
    --no-progress

echo ""
print_ok "Download complete! Files saved to: $LOCAL_PATH"
