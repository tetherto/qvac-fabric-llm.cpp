#!/usr/bin/env bash
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

readonly BYTES_PER_GIB=1073741824
readonly KIB_PER_GIB=1048576
readonly VULKAN_FIELDS_PER_DEVICE=4

load_config
setup_path

first_line() { "$@" 2>/dev/null | head -1; }

tool_version() {
  command -v "$1" > /dev/null || { echo ""; return; }
  first_line "$@"
}

os_version() {
  if [ "$OS_NAME" = "Darwin" ]; then echo "macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  else . /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-Linux}"; fi
}

cpu_model() {
  if [ "$OS_NAME" = "Darwin" ]; then sysctl -n machdep.cpu.brand_string
  else grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//'; fi
}

memory_gib() {
  if [ "$OS_NAME" = "Darwin" ]; then echo $(( $(sysctl -n hw.memsize) / BYTES_PER_GIB ))
  else echo $(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / KIB_PER_GIB )); fi
}

nvidia_field() {
  command -v nvidia-smi > /dev/null || { echo ""; return; }
  nvidia-smi -i "$(gpu_monitor_target)" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^ *//'
}

nvcc_binary() {
  if [ -x "$CUDA_HOME/bin/nvcc" ]; then echo "$CUDA_HOME/bin/nvcc"; else command -v nvcc; fi
}

nvcc_release() {
  local nvcc
  nvcc="$(nvcc_binary)"
  [ -n "$nvcc" ] || { echo ""; return; }
  "$nvcc" --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 | cut -d' ' -f2
}

vulkan_device_lines() {
  local first=$(( GPU_INDEX * VULKAN_FIELDS_PER_DEVICE + 1 )) last=$(( (GPU_INDEX + 1) * VULKAN_FIELDS_PER_DEVICE ))
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion' | sed -n "${first},${last}p"
}

vulkan_summary() {
  command -v vulkaninfo > /dev/null || { echo ""; return; }
  vulkan_device_lines | sed 's/^[[:space:]]*//' | tr -s ' ' | tr '\n' ';'
}

darwin_gpu_cores() {
  system_profiler SPDisplaysDataType 2>/dev/null | grep -m1 'Total Number of Cores' | cut -d: -f2 | sed 's/^ *//'
}

gpu_json() {
  if [ "$OS_NAME" = "Darwin" ]; then
    jq -n --arg name "$(cpu_model)" --arg cores "$(darwin_gpu_cores)" '{name:$name, gpu_cores:$cores, unified_memory:true}'
  else
    jq -n --arg name "$(nvidia_field name)" --arg memory "$(nvidia_field memory.total)" --arg driver "$(nvidia_field driver_version)" \
      --arg cuda "$(nvcc_release)" --arg vulkan "$(vulkan_summary)" --arg index "$(gpu_physical_index)" \
      '{name:$name, memory_mib:$memory, driver:$driver, cuda_toolkit:$cuda, vulkan:$vulkan, index:$index, unified_memory:false}'
  fi
}

compiler_version() {
  if [ "$OS_NAME" = "Darwin" ]; then tool_version clang --version; else tool_version gcc --version; fi
}

tools_json() {
  jq -n --arg cmake "$(tool_version cmake --version)" --arg ninja "$(tool_version ninja --version)" \
    --arg compiler "$(compiler_version)" --arg python "$(tool_version python3 --version)" --arg node "$(tool_version node --version)" \
    --arg git "$(tool_version git --version)" --arg nvcc "$(nvcc_release)" \
    '{cmake:$cmake, ninja:$ninja, compiler:$compiler, python:$python, node:$node, git:$git, nvcc:$nvcc}'
}

main() {
  jq -n --arg hostname "$(hostname -s)" --arg os "$OS_NAME" --arg os_version "$(os_version)" --arg kernel "$(uname -r)" \
    --arg arch "$(uname -m)" --arg cpu "$(cpu_model)" --argjson cpus "$(cpu_count)" --argjson memory "$(memory_gib)" \
    --argjson gpu "$(gpu_json)" --argjson tools "$(tools_json)" --arg collected "$(utc_now)" \
    '{hostname:$hostname, os:$os, os_version:$os_version, kernel:$kernel, arch:$arch, cpu:$cpu, cpu_count:$cpus,
      memory_gib:$memory, gpu:$gpu, tools:$tools, collected:$collected}'
}

main "$@"
