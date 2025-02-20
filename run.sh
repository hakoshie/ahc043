#!/bin/bash

for i in $(seq -f "%04g" 0 30); do
    echo "Processing tools/in/$i.txt..."
    # pypy3 main.py < tools/in/"$i".txt >> simulation.txt
done

echo "All files processed!"
