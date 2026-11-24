#!/usr/bin/env bash
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

load_config
setup_path
STORE="$MODELS_DIR/store"

repo_json() {
  local dir="$1" repo="$2" branch="$3" pin="$4" line
  line="$(repo_head_line "$dir" 2>/dev/null)"
  jq -n --arg repo "$(public_repo_url "$repo")" --arg branch "$branch" --arg pin "$pin" --arg sha "${line%% *}" --arg date "${line#* }" \
    '{repo:$repo, branch:$branch, pin:$pin, sha:$sha, commit_date:$date}'
}

family_file_names() {
  local family
  for family in $(split_list "$1"); do
    jq -r --arg f "$family" '.[$f].provision[] | (.registry // .huggingface.name // .fixture // .prepared // .sidecar.name // .derived.name) | split("/") | last' "$FAMILIES_JSON"
  done | sort -u
}

file_json() {
  local name="$1" path="$STORE/$1"
  [ -f "$path" ] || { jq -n --arg name "$name" '{name:$name, present:false}'; return; }
  jq -n --arg name "$name" --arg sha256 "$(cat "$path.sha256" 2>/dev/null)" --arg source "$(cat "$path.source" 2>/dev/null)" \
    --argjson bytes "$(wc -c < "$path" | tr -d ' ')" '{name:$name, present:true, bytes:$bytes, sha256:$sha256, source:$source}'
}

models_json() {
  local name
  for name in $(family_file_names "$1"); do file_json "$name"; done | jq -s '.'
}

asr_scorer_json() {
  jq '[to_entries[] | select(.key | startswith("_") | not) | .value.correctness.asr_model // empty] | unique | first // null' "$FAMILIES_JSON"
}

cuda_arch_in_use() {
  case ",$(resolve_backends)," in *",cuda,"*) echo "$CUDA_ARCH" ;; *) echo "" ;; esac
}

venv_packages() {
  [ -x "$MODELS_VENV/bin/python" ] || { echo ""; return; }
  "$MODELS_VENV/bin/python" -m pip freeze 2>/dev/null | tr '\n' ' '
}

registry_client_version() {
  jq -r '.version' "$REGISTRY_CLIENT_DIR/node_modules/@qvac/registry-client/package.json" 2>/dev/null
}

main() {
  local families="${1:-}"
  [ -n "$families" ] || families="$(all_families | tr '\n' ',')"
  jq -n --argjson engine "$(repo_json "$ENGINE_SRC" "$ENGINE_REPO" "$ENGINE_BRANCH" "${ENGINE_PIN:-}")" \
    --argjson ggml "$(repo_json "$GGML_SRC" "$GGML_REPO" "$GGML_BRANCH" "$(ggml_ref)")" \
    --argjson models "$(models_json "$families")" --argjson asr_scorer "$(asr_scorer_json)" \
    --arg cuda_arch "$(cuda_arch_in_use)" --arg registry_client "$(registry_client_version)" --arg venv "$(venv_packages)" \
    --arg families_sha256 "$( (sha256sum "$FAMILIES_JSON" 2>/dev/null || shasum -a 256 "$FAMILIES_JSON") | cut -d' ' -f1)" \
    '{engine:$engine, ggml:$ggml, build:{cuda_arch:$cuda_arch, ggml_native:"default (ON)"}, models:$models,
      tts_asr_scorer:$asr_scorer, registry_client:$registry_client, venv_packages:$venv, families_sha256:$families_sha256}'
}

main "$@"
