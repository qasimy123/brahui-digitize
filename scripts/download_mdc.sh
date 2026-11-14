#!/usr/bin/env bash
# Download Brahui datasets from Mozilla Data Collective. Needs MDC_API_KEY env var.
set -u
OUT="$(dirname "$0")/../data/raw/mdc"; mkdir -p "$OUT"
DS="common_voice_27_brh=cmu5qw5mb0016nq07rbpo3ax1 beco_literature=cmmus79780030ms07vyi4dsr5 beco_talar_magazine=cmlgx0s7j000jmg07pr4o635t beco_research_work=cmlgx1kdz000nmg0744z7bbxl"
for pair in $DS; do
  name=${pair%%=*}; id=${pair##*=}; mkdir -p "$OUT/$name"
  resp=$(curl -s -X POST "https://mozilladatacollective.com/api/datasets/$id/download" -H "Authorization: Bearer $MDC_API_KEY" -H "Content-Type: application/json")
  url=$(echo "$resp" | jq -r '.downloadUrl // empty')
  [ -z "$url" ] && { echo "FAIL $name: $resp" | head -c 300; echo; continue; }
  fname=$(echo "$resp" | jq -r '.fileName // .filename // empty'); [ -z "$fname" ] && fname=$(basename "${url%%\?*}")
  curl -sL -o "$OUT/$name/$fname" "$url" && echo "ok $name $fname $(du -h "$OUT/$name/$fname" | cut -f1)" || echo "FAIL curl $name"
done
echo done
