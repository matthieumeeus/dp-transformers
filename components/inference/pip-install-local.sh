#!/bin/bash

# Get the hostname
HOSTNAME=$(hostname)

# Define the lockfile name based on the hostname
LOCKFILE="/tmp/$HOSTNAME.lock"

# Function to clean up lockfile on script exit
cleanup() {
    rm -f "$LOCKFILE"
}

# Attempt to create a lockfile uniquely for the node
if ( set -o noclobber; echo "$$" > "$LOCKFILE") 2> /dev/null; then
    # Lockfile creation succeeded
    echo "Acquired lock, installing pip package on node: $HOSTNAME"
    
    # Ensure the lockfile is removed when the script exits or is interrupted
    trap cleanup EXIT
    
    python -m pip install $@
else
    # Lockfile already exists, wait for it to be released
    echo "Lockfile exists, waiting for pip installation to complete on node: $HOSTNAME"
    while [ -f "$LOCKFILE" ]; do
        sleep 1 # Wait for 1 second before checking again
    done
fi

echo "Pip installation completed on node: $HOSTNAME"
