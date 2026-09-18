#!/usr/bin/env bash
# The GPU notebook VM. This is the thing that costs money -- read the numbers.
#
#   bash cloud/workbench.sh create    # make it (once)
#   bash cloud/workbench.sh start     # turn it on   -> billing starts
#   bash cloud/workbench.sh stop      # turn it off  -> billing stops
#   bash cloud/workbench.sh status    # is it running?
#   bash cloud/workbench.sh url       # open JupyterLab
#   bash cloud/workbench.sh delete    # remove it entirely
#
# COST, as of September 2026, us-west1 on-demand:
#   n1-standard-4 + 1x NVIDIA T4   ~$0.55/hour   ~$400/month if left running
#   g2-standard-4 (1x L4)          ~$0.70/hour   ~$510/month if left running
#   the 150 GB boot disk            ~$0.04/hour  charged even while STOPPED
#
# A stopped instance costs only its disk (~$15/month for 150 GB). A running one
# costs ~35x that. The whole cost story of this project is "did you stop it".

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID (see cloud/setup.sh output)}"
REGION="${REGION:-us-west1}"
ZONE="${ZONE:-${REGION}-b}"
INSTANCE="${INSTANCE:-firebrand-gpu}"
MACHINE="${MACHINE:-n1-standard-4}"
GPU="${GPU:-NVIDIA_TESLA_T4}"
DISK_GB="${DISK_GB:-150}"
IDLE_MIN="${IDLE_MIN:-30}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }

case "${1:-}" in

create)
  bold "Creating $INSTANCE ($MACHINE + 1x $GPU) in $ZONE"
  warn "Idle shutdown: ${IDLE_MIN} min. Google's default is 180 min; 30 is a"
  warn "better default for a student project, and it is the main thing standing"
  warn "between you and a forgotten \$400 month."
  echo
  # Idle shutdown watches KERNEL activity, not CPU. A cell that runs for hours
  # while printing nothing can still be judged idle -- which is why train.py
  # prints a line per epoch. Keep it that way.
  # The command name and --location/--machine-type/--metadata are confirmed
  # against Google's docs. The accelerator and disk flag SPELLINGS vary between
  # gcloud releases and I could not verify them against the live reference, so
  # if one is rejected, check the real list rather than guessing:
  #     gcloud workbench instances create --help
  # The console (Vertex AI -> Workbench -> Create) is the reliable fallback and
  # takes about the same time; the settings to pick are in the guide.
  if ! gcloud workbench instances create "$INSTANCE" \
    --project="$PROJECT_ID" \
    --location="$ZONE" \
    --machine-type="$MACHINE" \
    --accelerator-type="$GPU" \
    --accelerator-core-count=1 \
    --install-gpu-driver \
    --data-disk-size="$DISK_GB" \
    --metadata="idle-timeout-seconds=$((IDLE_MIN * 60))"; then
    warn ""
    warn "Creation failed. If the message named an unrecognised flag, list the"
    warn "real ones with:   gcloud workbench instances create --help"
    warn "Or create it in the console: Vertex AI -> Workbench -> Create, with"
    warn "  machine $MACHINE, 1x $GPU, ${DISK_GB} GB disk,"
    warn "  and metadata  idle-timeout-seconds=$((IDLE_MIN * 60))"
    exit 1
  fi
  echo
  bold "Created. It is RUNNING and billing now."
  echo "  bash cloud/workbench.sh url     # open it"
  echo "  bash cloud/workbench.sh stop    # when you walk away"
  ;;

start)
  gcloud workbench instances start "$INSTANCE" --location="$ZONE" --project="$PROJECT_ID"
  warn "Billing has started. Stop it when you are done."
  ;;

stop)
  gcloud workbench instances stop "$INSTANCE" --location="$ZONE" --project="$PROJECT_ID"
  bold "Stopped. You are now paying only for the disk (~\$15/month)."
  ;;

status)
  gcloud workbench instances describe "$INSTANCE" --location="$ZONE" \
    --project="$PROJECT_ID" --format='value(state,gceSetup.machineType)'
  ;;

url)
  URL=$(gcloud workbench instances describe "$INSTANCE" --location="$ZONE" \
        --project="$PROJECT_ID" --format='value(proxyUri)')
  bold "https://${URL#https://}"
  ;;

delete)
  warn "This destroys the instance AND its disk. Anything not in the bucket is gone."
  read -r -p "Type the instance name to confirm: " c
  [ "$c" = "$INSTANCE" ] || { echo "aborted"; exit 1; }
  gcloud workbench instances delete "$INSTANCE" --location="$ZONE" --project="$PROJECT_ID"
  ;;

*)
  sed -n '2,25p' "$0"
  exit 1
  ;;
esac
