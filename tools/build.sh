#!/usr/bin/env bash
set -e

tools_path="$(cd "$(dirname "$0")" ; pwd -P)"
project_root="$(cd ${tools_path}/.. ; pwd -P)"

cd "${project_root}"
cmake --build build/
cd - &>/dev/null
