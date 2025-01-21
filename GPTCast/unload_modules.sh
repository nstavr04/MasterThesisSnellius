# Made by me to unload all modules besides code server and 2023 since VSCode needs them

#!/bin/bash

# List of modules to keep
keep_modules=("2023" "code-server/4.93.1")

# Get the currently loaded modules
loaded_modules=$(module list 2>&1 | grep -oP '^\s+\K[^ ]+')

# Loop through loaded modules and unload those not in the keep list
for module in $loaded_modules; do
    if [[ ! " ${keep_modules[@]} " =~ " ${module} " ]]; then
        echo "Unloading module: $module"
        module unload "$module"
    fi
done

# Confirm remaining loaded modules
echo "Remaining loaded modules:"
module list
