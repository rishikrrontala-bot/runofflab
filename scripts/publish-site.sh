#!/bin/sh
# Publishes the static public build of RunoffLab to GitHub Pages: the built site only, in its own repository.
# The source code is not pushed. Without --publish this only builds and stages, so nothing goes public.
#   sh scripts/publish-site.sh            # build + stage, print what would be published
#   sh scripts/publish-site.sh --publish  # create the public repository, push the site, enable Pages
set -eu
cd "$(dirname "$0")/.."
REPO=${REPO:-runofflab}
OWNER=$(gh api user --jq .login)
STAGE=$(mktemp -d)/site

npm run build:public
# Superseded 256 m data files are not used by the app; they stay local.
rsync -a --exclude data/park.json --exclude data/park-expanded.json --exclude data/park-aerial.jpg dist-public/ "$STAGE/"
touch "$STAGE/.nojekyll"
cat > "$STAGE/README.md" <<EOF
# RunoffLab (public build)

Live app: https://$OWNER.github.io/$REPO/

This repository holds only the compiled static website for RunoffLab, a 2026 Congressional App Challenge (NJ-12)
entry by Rishik Rontala. It runs a hypothetical storm over public NJ LiDAR terrain at Mercer County Park, lets you
place rain gardens, and reruns the same storm. It is an experiment with stated assumptions, not a flood forecast.

Live trained-model requests are off in this version; the trained model's recorded answers are included and every
design is simulated in the browser. AI coding assistants wrote the code; the disclosure is part of the submission.
EOF
echo "Staged $(find "$STAGE" -type f | wc -l | tr -d ' ') files in $STAGE"

if [ "${1:-}" != "--publish" ]; then
  echo "Not published. Rerun with --publish to create https://github.com/$OWNER/$REPO (public) and https://$OWNER.github.io/$REPO/"
  exit 0
fi

cd "$STAGE"
git init -q -b main
git add -A
git -c user.name="Rishik Rontala" -c user.email="297979650+rishikrrontala-bot@users.noreply.github.com" \
  commit -q -m "Publish RunoffLab static build" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
gh repo create "$OWNER/$REPO" --public --description "RunoffLab: rain-garden runoff experiment on NJ LiDAR terrain (static build)" --source . --push
gh api -X POST "repos/$OWNER/$REPO/pages" -f "source[branch]=main" -f "source[path]=/" > /dev/null
echo "Published. GitHub Pages usually needs 1-2 minutes: https://$OWNER.github.io/$REPO/"
