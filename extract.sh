#!/bin/bash

for d in dataset/*
    do
        FILENAME=$(basename "$d")
        filename="${FILENAME%.*}"
        echo "$filename"
        cd data
        mkdir $filename
        cd $filename
        extract_mvs ../../$d -d -v
        cd ../..
        echo "DONE"
    done