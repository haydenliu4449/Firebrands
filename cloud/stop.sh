#!/usr/bin/env bash
# Stop everything that bills by the hour, in this project. Run it when in doubt.
#
#   bash cloud/stop.sh
#
# Safe: stopping preserves the disk and everything on it.

set -euo pipefail
PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"

echo "Workbench instances:"
gcloud workbench instances list --project="$PROJECT_ID" \
  --format='value(name,state,location)' 2>/dev/null | while read -r n s l; do
  echo "  $n [$s]"
  [ "$s" = "ACTIVE" ] && gcloud workbench instances stop "$n" --location="$l" \
      --project="$PROJECT_ID" && echo "    -> stopped"
done

echo "Compute Engine VMs:"
gcloud compute instances list --project="$PROJECT_ID" \
  --format='value(name,status,zone)' 2>/dev/null | while read -r n s z; do
  echo "  $n [$s]"
  [ "$s" = "RUNNING" ] && gcloud compute instances stop "$n" --zone="$z" \
      --project="$PROJECT_ID" --quiet && echo "    -> stopped"
done

echo
echo "Nothing should be billing by the hour now. Storage still costs"
echo "~\$0.02/GB/month, which for 10 GB is 20 cents."
