#/bin/bash

# Create a random python filename
filename=$(uuidgen)

# always remove the script file after running
trap "rm -f $filename.py" EXIT

# exit if any command fails
set -e

# convert the notebook to a python script. Take the file path from the first argument and convert into to a random file name
jupyter nbconvert --to script $1 --output $filename

# run the script
python $filename.py
