#!/usr/bin/env bash
# Download Brahui audio from every JESUS Film segment linked from jesusfilm.org/watch/jesus.html/brahui.html
# Output: data/raw/jesusfilm_brh/<slug>.m4a (AAC audio only, no video) + <slug>.m3u8 url list
set -u
OUT="$(dirname "$0")/../data/raw/jesusfilm_brh"; mkdir -p "$OUT"
UA="Mozilla/5.0"
MAIN="https://www.jesusfilm.org/watch/jesus.html/brahui.html"
urls=$( { echo "$MAIN"; curl -sL -A "$UA" "$MAIN" | grep -oE 'https://www\.jesusfilm\.org/watch/[^"\\]+/brahui\.html'; } | sort -u )
for u in $urls; do
  slug=$(echo "$u" | sed -E 's#.*/watch/([^/]+)\.html/.*#\1#')
  dest="$OUT/$slug.m4a"
  [ -s "$dest" ] && { echo "skip $slug"; continue; }
  m3u8=$(curl -sL -A "$UA" "$u" | grep -oE 'https://stream\.mux\.com/[A-Za-z0-9]+\.m3u8' | head -1)
  [ -z "$m3u8" ] && { echo "FAIL no stream: $slug"; continue; }
  echo "$u $m3u8" >> "$OUT/sources.txt"
  ffmpeg -nostdin -loglevel error -y -i "$m3u8" -vn -c:a copy "$dest" && echo "ok $slug" || echo "FAIL ffmpeg: $slug"
done
echo done

# Subtitles for the full film (same cut as the Brahui audio): English (Arclight language 529) and Urdu (407).
# The Urdu track is only linked from the Urdu watch page; file naming differs between the two.
SUB="$OUT/subtitles"; mkdir -p "$SUB"
curl -s -A "$UA" -o "$SUB/en.vtt" "https://api-media-core.jesusfilm.org/1_jf-0-0/editions/ot/subtitles/1_jf-0-0_ot_529.vtt"
curl -s -A "$UA" -o "$SUB/ur.vtt" "https://api-media-core.jesusfilm.org/1_jf-0-0/editions/ot/subtitles/407_ot_1_jf-0-0.vtt"
echo "subtitles: en $(grep -c -- '-->' "$SUB/en.vtt") cues, ur $(grep -c -- '-->' "$SUB/ur.vtt") cues"
