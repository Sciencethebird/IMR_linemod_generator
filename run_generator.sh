#!/bin/sh

rm -r output/$1*
pip3 install opencv-python

python3 main_generator.py $1 --intrinsics 609.5134449988633 0.0 320.0 0.0 609.5134449988634 240.0 0.0 0.0 1.0
