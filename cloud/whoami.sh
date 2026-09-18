#!/usr/bin/env bash
# What can this Google Cloud account actually do?
#
#   bash cloud/whoami.sh
#
# Run this when something fails with PERMISSION_DENIED. On a managed
# university account the answer is usually "you can use things, but not
# create them", and this shows you exactly what you have been given.

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
dim()  { printf '\033[2m%s\033[0m\n' "$*"; }

bold "ACCOUNT"
gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | sed 's/^/  /' \
  || echo "  not logged in -- run: gcloud auth login"

bold ""
bold "APPLICATION-DEFAULT CREDENTIALS (what Python uses -- separate from the above)"
if [ -f "${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}/application_default_credentials.json" ] \
   || [ -f "$APPDATA/gcloud/application_default_credentials.json" ]; then
  echo "  present"
else
  echo "  MISSING -- run: gcloud auth application-default login"
fi

bold ""
bold "PROJECTS YOU CAN SEE"
P=$(gcloud projects list --format='table(projectId,name,projectNumber)' 2>&1)
if echo "$P" | grep -q 'projectId\|PROJECT_ID'; then echo "$P" | sed 's/^/  /'
else echo "  none -- you have not been added to any project yet"; fi

bold ""
bold "ORGANIZATION"
gcloud organizations list --format='table(displayName,ID)' 2>/dev/null | sed 's/^/  /' \
  || echo "  none visible"

bold ""
bold "FOLDERS (you may be able to create projects inside one even if not at the org root)"
ORG=$(gcloud organizations list --format='value(ID)' 2>/dev/null | head -1)
if [ -n "$ORG" ]; then
  gcloud resource-manager folders list --organization="$ORG" \
    --format='table(displayName,name)' 2>/dev/null | sed 's/^/  /' \
    || echo "  cannot list folders in organization $ORG"
else
  echo "  no organization visible"
fi

bold ""
bold "BILLING ACCOUNTS YOU CAN USE"
B=$(gcloud beta billing accounts list --format='table(name,displayName,open)' 2>&1)
if echo "$B" | grep -qi 'ACCOUNT_ID\|billingAccounts'; then echo "$B" | sed 's/^/  /'
else echo "  none -- someone else controls billing; you will need them to link it"; fi

bold ""
bold "CAN YOU CREATE PROJECTS?"
if gcloud projects create "fb-permcheck-$RANDOM$RANDOM" --format='value(projectId)' \
     --quiet >/dev/null 2>&1; then
  echo "  yes (a throwaway project was just created -- delete it from the console)"
else
  echo "  no. Use an existing project instead:"
  echo "     PROJECT_ID=the-project-id bash cloud/setup.sh"
fi

bold ""
dim "If you have a project but no billing, or no project at all, that is an"
dim "administrator's decision, not something to debug. cloud/setup.sh prints"
dim "the exact request to send them."
