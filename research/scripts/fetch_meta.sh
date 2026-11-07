#!/bin/sh
# usage: fetch_meta.sh <candidates file> <outdir>
while read -r id; do
  [ -z "$id" ] && continue
  f=$(echo "$id" | tr '/' '__')
  curl -s "https://huggingface.co/api/models/$id?expand[]=safetensors&expand[]=gated&expand[]=config&expand[]=cardData&expand[]=downloads&expand[]=likes&expand[]=siblings&expand[]=createdAt&expand[]=tags" > "$2/models/$f.json" &
  curl -sL "https://huggingface.co/$id/resolve/main/config.json" > "$2/configs/$f.json" &
  wait
done < "$1"
