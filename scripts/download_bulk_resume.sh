#!/usr/bin/env bash
# Resume the bulk scrape; if YouTube bot-blocks the plain run, retry that source with browser cookies (user approved 2026-11-11).
cd "$(dirname "$0")/.."
for n in bulk_story_kirtharrange bulk_mixed_brahvitawarmedia; do
  L=logs/download_${n}_resume.log
  uv run python scripts/download_media.py --only $n > $L 2>&1
  if grep -aq "Sign in to confirm" $L; then
    echo "$n: blocked without cookies, retrying with ${COOKIE_BROWSER:-chrome} cookies"
    YTDLP_COOKIES_BROWSER=${COOKIE_BROWSER:-chrome} uv run python scripts/download_media.py --only $n > ${L%.log}_cookies.log 2>&1
    echo "$n: still blocked on $(grep -ac 'Sign in to confirm' ${L%.log}_cookies.log) videos with cookies"
  else echo "$n: no block"; fi
  echo "$n: $(ls data/media/audio/$n/*.wav 2>/dev/null | wc -l) wavs"
done
grep -a "bulk_" logs/download_bulk_mixed_brahvitawarmedia_resume*.log | grep files | tail -8
