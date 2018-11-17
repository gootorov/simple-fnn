#!/usr/bin/env bash
set -e

tools_path="$(cd "$(dirname "$0")" ; pwd -P)"
project_root="$(cd ${tools_path}/.. ; pwd -P)"

cd "${tools_path}"
./build-release.sh
cd - &>/dev/null

cd "${project_root}"
./build/neural_net
cd - &>/dev/null
