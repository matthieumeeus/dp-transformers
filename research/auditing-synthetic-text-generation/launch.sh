#!/bin/bash

yq '.[]' experiment.yml  -r | parallel -j 16 --bar --halt now,fail=1
