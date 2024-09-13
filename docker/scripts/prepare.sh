#!/usr/bin/env bash

set -euo pipefail

# ANSI color codes
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RESET="\033[0m"

fn_print_color() {
    local color="$1"
    local message="$2"
    echo -e "${color}${message}${RESET}"
}

fn_check_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        fn_print_color "$YELLOW" "Warning: Directory $dir does not exist."
        return 1
    fi
    return 0
}

fn_check_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        fn_print_color "$YELLOW" "Warning: File $file does not exist."
        return 1
    fi
    return 0
}

fn_confirm_overwrite() {
    local dest="$1"
    read -r -p "$(fn_print_color "$YELLOW" "$dest already exists. Overwrite? (y/n): ")" choice
    [[ "${choice,,}" == "y" ]]
}

fn_copy_item() {
    local source="$1"
    local dest="$2"

    if [ -e "$dest" ] && ! fn_confirm_overwrite "$dest"; then
        fn_print_color "$YELLOW" "Skipping $source"
        return
    fi

    cp -r "$source" "$dest" && \
        fn_print_color "$GREEN" "Successfully copied $source to $dest" || \
        fn_print_color "$RED" "Failed to copy $source to $dest"
}

fn_create_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" && \
            fn_print_color "$GREEN" "Successfully created directory: $dir" || \
            fn_print_color "$RED" "Failed to create directory: $dir"
    else
        fn_print_color "$YELLOW" "Directory already exists: $dir"
    fi
}

# Main script from here ->
fn_print_color "$GREEN" "Starting environment setup for Docker container..."

source_paths=(
    "/ros2_ws/src/ftv/core/3d_models"
    "/ros2_ws/src/ftv/core/configuration.yaml"
    "/ros2_ws/src/ftv/core/cfg/default.json"
)

directories_to_create=(
    "./3d_models"
    "./cfg"
    "./tmp"
    "./task2/score"
)

# Create necessary directories
for dir in "${directories_to_create[@]}"; do
    fn_create_directory "$dir"
done

# Copy files and directories
for source in "${source_paths[@]}"; do
    if [ -d "$source" ]; then
        fn_check_directory "$source" && fn_copy_item "$source" "."
    elif [ -f "$source" ]; then
        fn_check_file "$source" && fn_copy_item "$source" "."
    else
        fn_print_color "$YELLOW" "Unknown item type: $source"
    fi
done

fn_print_color "$GREEN" "Environment setup complete."
