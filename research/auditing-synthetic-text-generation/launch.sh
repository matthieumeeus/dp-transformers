#!/bin/bash

yq '.[]' experiment.yml  -r | parallel --bar --halt now,fail=1
