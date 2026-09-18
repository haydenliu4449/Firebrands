#!/usr/bin/env bash
# One-time Google Cloud setup for the firebrand project.
#
#   bash cloud/setup.sh
#
# Creates: a project (optional), a storage bucket, a budget alert, and the two
# APIs this project needs. Safe to re-run -- every step checks first.
#
# Run this on your laptop, not on a cloud VM.

set -euo pipefail

# gcloud prompts on stdin for things like "install the beta component?". Inside
# a script, with stderr redirected, that prompt is invisible and the script just
# appears to hang. Make gcloud fail instead of ask.
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

# ---------------------------------------------------------------- settings
PROJECT_ID="${PROJECT_ID:-firebrand-$(whoami | tr -cd 'a-z0-9')-$RANDOM}"
BUCKET="${BUCKET:-${PROJECT_ID}-data}"
REGION="${REGION:-us-west1}"        # Oregon: close to Berkeley, usually has GPUs
BUDGET_USD="${BUDGET_USD:-50}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- checks
if ! command -v gcloud >/dev/null; then
  warn "gcloud is not installed."
  echo "  https://cloud.google.com/sdk/docs/install"
  echo "  Windows: download the installer. macOS: brew install --cask google-cloud-sdk"
  exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q .; then
  bold "Logging you in (a browser window will open)"
  gcloud auth login
fi

bold "Account: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"

# ---------------------------------------------------------------- discover
# On a personal Google account you can create projects freely. On a managed
# university account you almost certainly cannot: berkeley.edu projects live
# under an organization, and `resourcemanager.projects.create` on the org root
# is reserved for IT. That is a policy, not a fault, and the fix is to USE a
# project rather than create one.
bold "Looking for projects you can already use"
VISIBLE=$(gcloud projects list --format='value(projectId)' 2>/dev/null || true)
if [ -n "$VISIBLE" ]; then
  echo "$VISIBLE" | sed 's/^/  /'
else
  echo "  (none visible)"
fi

# Explicit PROJECT_ID that already exists: just use it, never try to create.
if [ -n "${PROJECT_ID:-}" ] && gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
  bold "Using existing project $PROJECT_ID"
else
  if [ -z "${PROJECT_ID:-}" ]; then
    PROJECT_ID="firebrand-$(whoami | tr -cd 'a-z0-9')-$RANDOM"
  fi
  bold "Attempting to create project $PROJECT_ID"
  if ! gcloud projects create "$PROJECT_ID" --name="Firebrand tracking" 2>/tmp/fb_create_err; then
    ERR=$(cat /tmp/fb_create_err)
    echo "$ERR" | sed 's/^/  /'
    if echo "$ERR" | grep -qi 'resourcemanager.projects.create.*denied\|PERMISSION_DENIED'; then
      ORG=$(echo "$ERR" | grep -o 'organizations/[0-9]*' | head -1)
      warn ""
      warn "You do not have permission to create projects under ${ORG:-your organization}."
      warn "This is normal for a managed university account and is not something"
      warn "you can fix from here. You need an existing project instead."
      cat <<EOF

  WHAT TO DO

  1. If the lab already has a project, ask for its ID and re-run:

       PROJECT_ID=their-project-id bash cloud/setup.sh

  2. If not, send whoever administers the lab's Google Cloud this:

       Hi - I need a Google Cloud project for the Fire Research Lab
       firebrand tracking work. I need:
         * a project I have Editor on
         * a billing account linked to it
         * these APIs enabled: storage, notebooks, aiplatform, compute
         * a GPU quota of 1 for GPUS_ALL_REGIONS in us-west1
       Expected spend is under \$40/month (one T4 notebook VM used a few
       hours a week, plus ~10 GB of Cloud Storage).
       My account is $(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null).

  3. Check whether you can create inside a FOLDER even if not at the org root
     (some universities delegate per-department):

       gcloud resource-manager folders list --organization=${ORG#organizations/}
       gcloud projects create NEW_ID --folder=FOLDER_NUMBER

  4. Meanwhile, everything except training runs on your laptop for free --
     see steps 1 and 4-7 of the training runbook. You are not blocked.

EOF
      rm -f /tmp/fb_create_err
      exit 1
    fi
    rm -f /tmp/fb_create_err
    exit 1
  fi
fi
gcloud config set project "$PROJECT_ID" >/dev/null

# Billing check -- ADVISORY, not fatal.
#
# On a lab-managed project you very likely have Editor but not billing
# permissions, so this read fails even when billing is perfectly well set up by
# whoever owns the project. Treating "cannot read billing" as "no billing" would
# block you from a project that works fine. The real test is whether the API
# enable below succeeds.
#
# `gcloud billing` (GA) rather than `gcloud beta billing`: the beta form makes
# gcloud offer to install the beta component, which is a prompt, which is a hang.
BILLING=""
if BILLING_RAW=$(gcloud billing projects describe "$PROJECT_ID" \
                   --format='value(billingAccountName)' 2>&1); then
  BILLING="$BILLING_RAW"
fi

if [ -n "$BILLING" ]; then
  bold "Billing: $BILLING"
elif echo "${BILLING_RAW:-}" | grep -qi 'permission\|denied\|forbidden'; then
  warn "Cannot read billing on $PROJECT_ID (you lack billing permissions)."
  warn "That is normal on a shared project and is probably fine -- continuing."
else
  warn "No billing account appears to be linked to $PROJECT_ID."
  warn "If the next step fails, that is why. Ask whoever owns the project to link one."
fi

# ---------------------------------------------------------------- APIs
bold "Enabling APIs (takes a minute the first time)"
if ! API_ERR=$(gcloud services enable \
      storage.googleapis.com \
      notebooks.googleapis.com \
      aiplatform.googleapis.com \
      compute.googleapis.com 2>&1); then
  echo "$API_ERR" | sed 's/^/  /'
  warn ""
  if echo "$API_ERR" | grep -qi 'billing'; then
    warn "Billing is not enabled on $PROJECT_ID. Ask the project owner to link"
    warn "a billing account -- you cannot do this yourself without permissions."
  elif echo "$API_ERR" | grep -qi 'permission\|denied'; then
    warn "You do not have permission to enable APIs on $PROJECT_ID."
    warn "Ask the owner to enable: storage, notebooks, aiplatform, compute."
    warn "If they are already enabled, you can ignore this and keep going:"
    warn "  gcloud services list --enabled --project=$PROJECT_ID"
  fi
  exit 1
fi

# ---------------------------------------------------------------- bucket
if gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  bold "Bucket gs://$BUCKET already exists and you can see it"
else
  bold "Creating gs://$BUCKET in $REGION"
  # Uniform access: no per-object ACLs. Simpler, and far harder to accidentally
  # make world-readable -- which is how research data leaks.
  if ! BKT_ERR=$(gcloud storage buckets create "gs://$BUCKET" \
        --location="$REGION" \
        --uniform-bucket-level-access \
        --public-access-prevention 2>&1); then
    echo "$BKT_ERR" | sed 's/^/  /'
    warn ""
    if echo "$BKT_ERR" | grep -qi 'already own\|already exists\|HTTPError 409'; then
      warn "That bucket name is taken. Bucket names are unique across ALL of"
      warn "Google Cloud, not just this project -- and '$BUCKET' is an obvious one."
      warn "Pick a distinctive name and re-run:"
      warn "  PROJECT_ID=$PROJECT_ID BUCKET=${BUCKET}-$(whoami | tr -cd 'a-z0-9')-$RANDOM bash cloud/setup.sh"
    elif echo "$BKT_ERR" | grep -qi 'permission\|denied'; then
      warn "You cannot create buckets in $PROJECT_ID."
      warn "If the lab already has one, just use it -- you do not need your own:"
      warn "  gcloud storage ls"
      warn "  BUCKET=their-bucket-name bash cloud/setup.sh"
    fi
    exit 1
  fi
fi

# ---------------------------------------------------------------- budget
if [ -z "$BILLING" ]; then
  bold "Skipping budget alert (billing account not readable from this account)"
  echo "  Ask the project owner to set one, or create it in the console:"
  echo "  Billing -> Budgets & alerts"
else
  bold "Setting a \$${BUDGET_USD} budget alert"
  ACCOUNT_ID="${BILLING#billingAccounts/}"
  if gcloud billing budgets list --billing-account="$ACCOUNT_ID" \
       --format='value(displayName)' 2>/dev/null | grep -q "^firebrand-budget$"; then
    echo "  budget already exists"
  else
    gcloud billing budgets create \
      --billing-account="$ACCOUNT_ID" \
      --display-name="firebrand-budget" \
      --budget-amount="${BUDGET_USD}USD" \
      --threshold-rule=percent=0.5 \
      --threshold-rule=percent=0.9 \
      --threshold-rule=percent=1.0 \
      --filter-projects="projects/$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')" \
      2>/dev/null || warn "  could not create budget (needs billing.budgets.create)"
  fi
fi

warn ""
warn "A budget alert EMAILS you. It does not stop spending."
warn "The only hard stop is turning the VM off -- see cloud/stop.sh."

# ---------------------------------------------------------------- summary
cat <<EOF

$(bold "Done.")

  project   $PROJECT_ID
  bucket    gs://$BUCKET
  region    $REGION

Save these for the other scripts:

  export PROJECT_ID=$PROJECT_ID
  export BUCKET=$BUCKET
  export REGION=$REGION

Next:
  bash cloud/upload.sh  clips/               # push your footage up
  bash cloud/workbench.sh create             # start the GPU notebook VM
EOF
