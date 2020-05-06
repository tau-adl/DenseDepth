#!/usr/bin/env bash

$(mkdir -p "Data/")
echo "Downloading Data"
fileid="1fdFu5NGXe4rTLYKD5wOqk9dl-eJOefXo"
filename="nyu_data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o "Data/"${filename}
$(rm cookie)
