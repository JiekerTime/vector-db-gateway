#!/usr/bin/env bash
set -euo pipefail

REMOTE="${VECTOR_GATEWAY_REMOTE:-homeserver-ext}"
SSH_KEY="${VECTOR_GATEWAY_SSH_KEY:-}"
if [[ -z "$SSH_KEY" && "$REMOTE" == *@192.168.1.100 ]]; then
    SSH_KEY="$HOME/.ssh/ali"
fi
REMOTE_BUILD_DIR="/opt/docker/vector_db_gateway_git"
CONTAINER="vector-db-gateway"
IMAGE="vector-db-gateway:latest"
PORT=8526
NETWORK="docker_proxy"
DATA_DIR="/data/vector_db_gateway"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_MODE="${VECTOR_GATEWAY_GPU_MODE:-auto}"
WATCHTOWER_ENABLE="${VECTOR_GATEWAY_WATCHTOWER_ENABLE:-false}"

SSH_BASE=(ssh)
SCP_BASE=(scp)
if [[ -n "$SSH_KEY" ]]; then
    SSH_BASE+=(-i "$SSH_KEY")
    SCP_BASE+=(-i "$SSH_KEY")
    RSYNC_SSH="ssh -i $SSH_KEY"
else
    RSYNC_SSH="ssh"
fi
SSH_CMD=("${SSH_BASE[@]}" "$REMOTE")
SCP_CMD=("${SCP_BASE[@]}")

hotpatch() {
    echo "==> Hotpatch vector-db-gateway"
    changed=$( (git -C "$SCRIPT_DIR" diff --name-only HEAD 2>/dev/null; git -C "$SCRIPT_DIR" status --short | awk '{print $2}') | sort -u )
    if [[ -z "$changed" ]]; then
        echo "No changed files detected."
        exit 0
    fi

    for f in $changed; do
        [[ -f "$SCRIPT_DIR/$f" ]] || continue
        case "$f" in
            *.py|*.yaml|requirements.txt)
                parent="$(dirname "$f")"
                if [[ "$parent" != "." ]]; then
                    "${SSH_CMD[@]}" "docker exec $CONTAINER mkdir -p /app/$parent"
                fi
                "${SCP_CMD[@]}" "$SCRIPT_DIR/$f" "$REMOTE:/tmp/vg_patch_$(basename "$f")"
                "${SSH_CMD[@]}" "docker cp /tmp/vg_patch_$(basename "$f") $CONTAINER:/app/$f && rm /tmp/vg_patch_$(basename "$f")"
                ;;
        esac
    done
    "${SSH_CMD[@]}" "docker restart $CONTAINER && sleep 3 && docker logs --tail 10 $CONTAINER 2>&1"
}

full_deploy() {
    echo "==> Full deploy vector-db-gateway"
    "${SSH_CMD[@]}" "mkdir -p $REMOTE_BUILD_DIR $DATA_DIR/logs $DATA_DIR/cache $DATA_DIR/state"
    rsync -avz --delete \
        -e "$RSYNC_SSH" \
        --exclude '.git' \
        --exclude '.venv' \
        --exclude '.memory/' \
        --exclude 'AGENTS.md' \
        --exclude '__pycache__' \
        --exclude 'logs/' \
        --exclude '*.pyc' \
        "$SCRIPT_DIR/" "$REMOTE:$REMOTE_BUILD_DIR/"

    "${SSH_CMD[@]}" "cd $REMOTE_BUILD_DIR && docker build -t $IMAGE ."
    "${SSH_CMD[@]}" "docker stop $CONTAINER 2>/dev/null || true"
    "${SSH_CMD[@]}" "docker rm $CONTAINER 2>/dev/null || true"
    gpu_args=""
    if [[ "$GPU_MODE" != "off" ]]; then
        gpu_args="--gpus all"
    fi
    "${SSH_CMD[@]}" "
        docker run -d \
            --name $CONTAINER \
            --restart unless-stopped \
            --label com.centurylinklabs.watchtower.enable=$WATCHTOWER_ENABLE \
            $gpu_args \
            --network $NETWORK \
            -p $PORT:8526 \
            -v $DATA_DIR/logs:/app/logs \
            -v $DATA_DIR/state:/app/state \
            -v $DATA_DIR/cache:/root/.cache/huggingface \
            --memory=6g \
            --memory-swap=8g \
            $IMAGE
    "
}

case "${1:-full}" in
    hotpatch|hp) hotpatch ;;
    full) full_deploy ;;
    *) echo "Usage: $0 [full|hotpatch]"; exit 1 ;;
esac
