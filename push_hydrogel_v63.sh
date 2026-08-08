#!/bin/bash
# push_hydrogel_v63.sh
# Run this when network to github.com is restored
# Local commits c7b17bb (v6.3) and d26950d (v6.2) are ready to push

set -e
cd "C:/Users/TS/WorkBuddy/HydroGelNet"

echo "[1/3] Verifying local commits..."
git log --oneline -3
echo ""

echo "[2/3] Attempting push (with SSL bypass + longer timeout)..."
GIT_SSL_NO_VERIFY=1 git -c http.lowSpeedLimit=1000 \
                      -c http.lowSpeedTime=999 \
                      -c http.postBuffer=524288000 \
                      push origin master:main 2>&1

echo ""
echo "[3/3] Verifying remote sync..."
git ls-remote origin master 2>&1 | head -3

echo ""
echo "Done. If push failed, check:"
echo "  - Network: ping github.com / curl -I https://github.com"
echo "  - Credentials: git config --list | grep -i credential"
echo "  - Manual: git push https://<TOKEN>@github.com/SHENTongfei/HydroGelNet.git master:main"
