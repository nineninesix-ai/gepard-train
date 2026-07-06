#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─────────────────────────────────────────
# Colors
# ─────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

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
echo ""
read -rp "Your choice [1/2]: " AUTH_CHOICE

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
    *)
        print_err "Invalid choice."
        exit 1
        ;;
esac

# ─────────────────────────────────────────
# Step 3: Choose local dataset
# ─────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo "         Available local datasets"
echo "────────────────────────────────────────"

DATASETS=()
while IFS= read -r dir; do
    if [ -f "$dir/dataset_info.json" ]; then
        DATASETS+=("$dir")
    fi
done < <(find "$SCRIPT_DIR" -maxdepth 1 -mindepth 1 -type d | sort)

UPLOAD_ALL=false

if [ ${#DATASETS[@]} -eq 0 ]; then
    print_warn "No HF datasets found in $SCRIPT_DIR (no dataset_info.json). Enter path manually."
else
    for i in "${!DATASETS[@]}"; do
        NAME=$(basename "${DATASETS[$i]}")
        SIZE=$(du -sh "${DATASETS[$i]}" 2>/dev/null | cut -f1)
        echo "  $((i+1))) $NAME  [$SIZE]"
    done
    echo "  a) Upload ALL datasets"
    echo "  0) Enter path manually"
    echo ""
    read -rp "Choose dataset [0-${#DATASETS[@]}/a]: " DS_CHOICE

    if [[ "$DS_CHOICE" == "a" || "$DS_CHOICE" == "A" ]]; then
        UPLOAD_ALL=true
        print_ok "Mode: upload ALL ${#DATASETS[@]} datasets"
    elif [[ "$DS_CHOICE" =~ ^[1-9][0-9]*$ ]] && [ "$DS_CHOICE" -le "${#DATASETS[@]}" ]; then
        LOCAL_PATH="${DATASETS[$((DS_CHOICE-1))]}"
        print_ok "Selected: $LOCAL_PATH"
    fi
fi

if [ "$UPLOAD_ALL" = false ] && [ -z "$LOCAL_PATH" ]; then
    read -rp "Local dataset path: " LOCAL_PATH
    LOCAL_PATH="${LOCAL_PATH/#\~/$HOME}"
fi

if [ "$UPLOAD_ALL" = false ]; then
    if [ ! -d "$LOCAL_PATH" ]; then
        print_err "Directory '$LOCAL_PATH' does not exist."
        exit 1
    fi

    if [ ! -f "$LOCAL_PATH/dataset_info.json" ]; then
        print_warn "'$LOCAL_PATH' does not look like an HF dataset (no dataset_info.json)."
        read -rp "Continue anyway? [y/N]: " ANYWAY
        [[ "$ANYWAY" =~ ^[Yy]$ ]] || exit 0
    fi
fi

# ─────────────────────────────────────────
# Step 4: S3 destination
# ─────────────────────────────────────────
echo ""
read -rp "S3 destination (e.g. s3://my-bucket/datasets/): " S3_BASE

[[ "$S3_BASE" != s3://* ]] && S3_BASE="s3://$S3_BASE"
# Ensure trailing slash
[[ "$S3_BASE" != */ ]] && S3_BASE="$S3_BASE/"

# ─────────────────────────────────────────
# Step 5: Preview and confirm
# ─────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo " Summary"
echo "────────────────────────────────────────"

if [ "$UPLOAD_ALL" = true ]; then
    TOTAL_SIZE=$(du -sh "${DATASETS[@]}" 2>/dev/null | tail -1 | cut -f1)
    echo "  Mode        : Upload ALL datasets (${#DATASETS[@]} total)"
    echo "  Destination : ${S3_BASE}<dataset_name>"
    echo ""
    for dir in "${DATASETS[@]}"; do
        NAME=$(basename "$dir")
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "    • $NAME  [$SIZE]  →  ${S3_BASE}${NAME}"
    done
else
    DATASET_NAME=$(basename "$LOCAL_PATH")
    S3_PATH="${S3_BASE}${DATASET_NAME}"
    echo "  Source      : $LOCAL_PATH"
    echo "  Destination : $S3_PATH"
    echo "  Size        : $(du -sh "$LOCAL_PATH" | cut -f1)"
fi

echo ""
read -rp "Start upload? [Y/n]: " CONFIRM
CONFIRM="${CONFIRM:-Y}"

[[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# ─────────────────────────────────────────
# Step 6: Upload
# ─────────────────────────────────────────
if [ "$UPLOAD_ALL" = true ]; then
    TOTAL=${#DATASETS[@]}
    for i in "${!DATASETS[@]}"; do
        DIR="${DATASETS[$i]}"
        NAME=$(basename "$DIR")
        S3_PATH="${S3_BASE}${NAME}"
        echo ""
        print_step "[$((i+1))/$TOTAL] Uploading $NAME → $S3_PATH ..."
        aws s3 sync "$DIR" "$S3_PATH" --no-progress
        print_ok "Done: $S3_PATH"
    done
    echo ""
    print_ok "All $TOTAL datasets uploaded to ${S3_BASE}"
else
    print_step "Uploading $LOCAL_PATH → $S3_PATH ..."
    echo ""
    aws s3 sync "$LOCAL_PATH" "$S3_PATH" --no-progress
    echo ""
    print_ok "Upload complete! Dataset available at: $S3_PATH"
fi
