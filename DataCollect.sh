#!/usr/bin/env bash

$(mkdir -p "Data/")
echo "Downloading Data"
file_link="http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz"
file_path="Data/nyudepthv2.tar.gz"
curl -L $file_link > $file_path
tar -xvf $file_path -C "Data/"
$(rm $file_path)
