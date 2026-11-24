#!/usr/bin/env bash
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

readonly REGISTRY_CLIENT_PACKAGE="@qvac/registry-client"
readonly REGISTRY_PREFIX="qvac_models_compiled/ggml"
readonly HF_BASE="https://huggingface.co"
readonly CURL_RETRIES=20
readonly VENV_PACKAGES="gguf==0.18.0 numpy>=2.0,<3"

load_config
setup_path
REGISTRY_CLIENT_VERSION="${REGISTRY_CLIENT_VERSION:-0.7.0}"
STORE="$MODELS_DIR/store"
FAMILIES_ROOT="$MODELS_DIR/families"
VENV="$MODELS_VENV"
CLIENT_DIR="$REGISTRY_CLIENT_DIR"
FAILURES=0

sha256_of() {
  if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

record_file() {
  local name="$1" source="$2"
  echo "$source" > "$STORE/$name.source"
  sha256_of "$STORE/$name" > "$STORE/$name.sha256"
}

provisioned() {
  local name="$1" source="$2"
  [ -f "$STORE/$name" ] && [ "$(cat "$STORE/$name.source" 2>/dev/null)" = "$source" ]
}

registry_source() { echo "registry $REGISTRY_PREFIX/$1"; }

huggingface_source() { echo "huggingface $HF_BASE/$1/resolve/$2/$3"; }

fixture_source() { echo "fixture $1 @ $(repo_sha "$ENGINE_SRC")"; }

requantize_source() { echo "derived requantize $2 $3 from $1"; }

resample_source() { echo "derived resample $2 from $1 @ $(repo_sha "$ENGINE_SRC")"; }

discard_stale() {
  local name="$1"
  rm -f "${STORE:?}/${name:?}" "${STORE:?}/${name:?}.source" "${STORE:?}/${name:?}.sha256"
}

note_failure() {
  log "FAILED: $*"
  FAILURES=$(( FAILURES + 1 ))
}

registry_client_installed() {
  [ -d "$CLIENT_DIR/node_modules/$REGISTRY_CLIENT_PACKAGE" ] && [ "$(cat "$CLIENT_DIR/VERSION" 2>/dev/null)" = "$REGISTRY_CLIENT_VERSION" ]
}

ensure_registry_client() {
  registry_client_installed && return 0
  mkdir -p "$CLIENT_DIR"
  rm -f "${CLIENT_DIR:?}/VERSION"
  echo '{"name": "speech-bench-registry-client", "private": true}' > "$CLIENT_DIR/package.json"
  npm install --prefix "$CLIENT_DIR" --silent --no-audit --no-fund --ignore-scripts "$REGISTRY_CLIENT_PACKAGE@$REGISTRY_CLIENT_VERSION" &&
    [ -d "$CLIENT_DIR/node_modules/$REGISTRY_CLIENT_PACKAGE" ] && echo "$REGISTRY_CLIENT_VERSION" > "$CLIENT_DIR/VERSION"
}

ensure_venv() {
  [ -f "$VENV/READY" ] && return 0
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --upgrade pip && "$VENV/bin/pip" install -q $VENV_PACKAGES &&
    touch "$VENV/READY"
}

fetch_registry_keys() {
  [ $# -gt 0 ] || return 0
  ensure_registry_client || { note_failure "registry client install"; return 1; }
  REGISTRY_CLIENT="$CLIENT_DIR/node_modules/$REGISTRY_CLIENT_PACKAGE" node "$TOOLS_DIR/fetch-registry.js" "$STORE" "$@"
}

record_registry_keys() {
  local key
  for key in "$@"; do
    if [ -f "$STORE/$(basename "$key")" ]; then record_file "$(basename "$key")" "$(registry_source "$key")"
    else note_failure "registry $key"; fi
  done
}

missing_registry_keys() {
  local family key
  for family in $(split_list "$1"); do
    for key in $(jq -r --arg f "$family" '.[$f].provision[] | .registry // empty' "$FAMILIES_JSON"); do
      provisioned "$(basename "$key")" "$(registry_source "$key")" || { discard_stale "$(basename "$key")"; echo "$key"; }
    done
  done | sort -u
}

fetch_missing_registry_keys() {
  local keys
  keys="$(missing_registry_keys "$1")"
  [ -n "$keys" ] || return 0
  fetch_registry_keys $keys
  record_registry_keys $keys
}

verify_sha256() {
  local file="$1" expected="$2"
  [ -n "$expected" ] || { log "no sha256 pinned for $file"; return 1; }
  [ "$(sha256_of "$file")" = "$expected" ]
}

fetch_huggingface() {
  local repo="$1" revision="$2" name="$3" expected="$4" url="$HF_BASE/$1/resolve/$2/$3"
  provisioned "$name" "$(huggingface_source "$repo" "$revision" "$name")" && return 0
  discard_stale "$name"
  log "fetch $url"
  rm -f "${STORE:?}/${name:?}.part"
  curl -fsSL --retry "$CURL_RETRIES" --retry-all-errors -o "$STORE/$name.part" "$url" &&
    verify_sha256 "$STORE/$name.part" "$expected" && mv "$STORE/$name.part" "$STORE/$name" &&
    record_file "$name" "$(huggingface_source "$repo" "$revision" "$name")"
}

copy_fixture() {
  local path="$1" name
  name="$(basename "$path")"
  provisioned "$name" "$(fixture_source "$path")" && return 0
  cp "$ENGINE_SRC/$path" "$STORE/$name.part" && mv "$STORE/$name.part" "$STORE/$name" && record_file "$name" "$(fixture_source "$path")"
}

requantize() {
  local name="$1" source="$2" type="$3" filter="$4"
  provisioned "$name" "$(requantize_source "$source" "$type" "$filter")" && return 0
  discard_stale "$name"
  ensure_venv || return 1
  log "requantize $source -> $name ($type, $filter)"
  "$VENV/bin/python" "$ENGINE_SRC/tts/scripts/requantize-gguf.py" "$STORE/$source" "$STORE/$name.part" "$type" \
    --name-filter "$filter" > "$STORE/$name.log" 2>&1 &&
    mv "$STORE/$name.part" "$STORE/$name" && record_file "$name" "$(requantize_source "$source" "$type" "$filter")"
}

resample() {
  local name="$1" source="$2" rate="$3"
  provisioned "$name" "$(resample_source "$source" "$rate")" && return 0
  python3 "$TOOLS_DIR/prepare-derived.py" resample --source "$ENGINE_SRC/$source" --out "$STORE/$name.part" --rate "$rate" &&
    mv "$STORE/$name.part" "$STORE/$name" && record_file "$name" "$(resample_source "$source" "$rate")"
}

provision_derived() {
  local entry="$1" kind
  kind="$(echo "$entry" | jq -r '.kind')"
  case "$kind" in
    requantize) requantize "$(echo "$entry" | jq -r '.name')" "$(echo "$entry" | jq -r '.source')" \
                  "$(echo "$entry" | jq -r '.type')" "$(echo "$entry" | jq -r '.name_filter // ""')" ;;
    resample) resample "$(echo "$entry" | jq -r '.name')" "$(echo "$entry" | jq -r '.source')" "$(echo "$entry" | jq -r '.rate')" ;;
    *) return 1 ;;
  esac
}

sidecar_package() { echo "$STORE/${1%.mlmodelc}.mlpackage"; }

sidecar_args() { echo "$1" | jq -r '.args[]?'; }

run_sidecar_exporter() {
  local name="$1" exporter="$2" source="$3"; shift 3
  ( cd "$ENGINE_SRC" && "$COREML_PYTHON" "$exporter" --gguf "$STORE/$source" --out "$(sidecar_package "$name")" --compile-dir "$STORE" "$@" ) \
    > "$STORE/$name.export.log" 2>&1
}

export_sidecar() {
  local entry="$1" name exporter source
  name="$(echo "$entry" | jq -r '.name')"; exporter="$(echo "$entry" | jq -r '.exporter')"; source="$(echo "$entry" | jq -r '.source')"
  [ -d "$STORE/$name" ] && return 0
  if [ -z "$COREML_PYTHON" ]; then log "sidecar $name needs COREML_PYTHON in config.env"; return 1; fi
  [ -f "$STORE/$source" ] || { log "sidecar $name: source $source is not in the store"; return 1; }
  log "exporting Core ML sidecar $name from $source"
  run_sidecar_exporter "$name" "$exporter" "$source" $(sidecar_args "$entry") || return 1
  rm -rf "$(sidecar_package "$name")"
  [ -d "$STORE/$name" ]
}

entry_name() {
  echo "$1" | jq -r 'if .registry then (.registry | split("/") | last)
    elif .huggingface then .huggingface.name
    elif .fixture then (.fixture | split("/") | last)
    elif .prepared then .prepared
    elif .sidecar then .sidecar.name
    else .derived.name end'
}

link_name() {
  echo "$1" | jq -r --arg name "$(entry_name "$1")" '.as // $name'
}

provision_entry() {
  local entry="$1"
  if echo "$entry" | jq -e '.huggingface' > /dev/null; then
    fetch_huggingface "$(echo "$entry" | jq -r '.huggingface.repo')" "$(echo "$entry" | jq -r '.huggingface.revision')" \
      "$(echo "$entry" | jq -r '.huggingface.name')" "$(echo "$entry" | jq -r '.huggingface.sha256 // ""')"
  elif echo "$entry" | jq -e '.fixture' > /dev/null; then copy_fixture "$(echo "$entry" | jq -r '.fixture')"
  elif echo "$entry" | jq -e '.derived' > /dev/null; then provision_derived "$(echo "$entry" | jq -c '.derived')"
  elif echo "$entry" | jq -e '.sidecar' > /dev/null; then export_sidecar "$(echo "$entry" | jq -c '.sidecar')"
  else [ -f "$STORE/$(entry_name "$entry")" ]
  fi
}

link_into_family() {
  local family="$1" name="$2" link="$3"
  mkdir -p "$FAMILIES_ROOT/$family"
  ln -sfn "$STORE/$name" "$FAMILIES_ROOT/$family/$link"
}

provision_family() {
  local family="$1" entry name
  for entry in $(jq -c --arg f "$family" '.[$f].provision[]' "$FAMILIES_JSON" | tr ' ' '\001'); do
    entry="$(echo "$entry" | tr '\001' ' ')"
    name="$(entry_name "$entry")"
    if provision_entry "$entry"; then link_into_family "$family" "$name" "$(link_name "$entry")"; else note_failure "$family: $name"; fi
  done
}

provision_families() {
  local family
  for family in $(split_list "$1"); do provision_family "$family"; done
}

main() {
  local families="${1:-}"
  [ -n "$families" ] || families="$(all_families | tr '\n' ',')"
  mkdir -p "$STORE" "$FAMILIES_ROOT"
  fetch_missing_registry_keys "$families"
  provision_families "$families"
  log "models provisioned with $FAILURES failures"
  [ "$FAILURES" -eq 0 ]
}

main "$@"
