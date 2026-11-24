#!/bin/bash
# fresh-host.sh: give a benchmark binary a host identity the ANE has never seen,
# so a cold-load measurement really starts with no cached ANE plans.
#
# The ANE plan cache and the per-executable e5rt cache are keyed by the host's
# code-signing identity; a plain copy keeps the identifier, so a cold run
# needs a re-signed copy. For an ad-hoc signature the identifier and the file
# name are both set to qvac-bench-<uuid>, so either observation of the host key
# (code-signing identifier or executable name) sees a new host.
#
#   fresh-host.sh new <binary> [--dest DIR]   copy + re-sign; prints JSON
#                                              {"id","binary","identifier","cache"}
#   fresh-host.sh cleanup <id|binary>          delete the copy's directory and the
#                                              e5rt caches it created
#   fresh-host.sh cleanup-all                  every qvac-bench-* copy and cache
#   fresh-host.sh caches <id>                  list the cache directories of <id>
#
# Copies live under ${QVAC_BENCH_HOSTS:-$TMPDIR/qvac-bench-hosts}/<id>/<id>. A
# binary whose rpaths are relative to its own location (@loader_path,
# @executable_path) gets the original directory added as an absolute rpath
# before signing, so its dylibs still resolve from the copy.
set -euo pipefail

HOSTS_DIR="${QVAC_BENCH_HOSTS:-${TMPDIR:-/tmp}/qvac-bench-hosts}"
HOSTS_DIR="${HOSTS_DIR%/}"
CACHES_DIR="${QVAC_BENCH_CACHES:-$HOME/Library/Caches}"
PREFIX="qvac-bench-"

die() { echo "fresh-host.sh: $*" >&2; exit 1; }
usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 64; }

json_str() { # minimal JSON string escaping for paths and identifiers
  local s="${1//\\/\\\\}"; s="${s//\"/\\\"}"; printf '"%s"' "$s"
}

# The cache directories a host identity creates. The e5rt bundle cache lives
# under the identifier (ad-hoc copies: also the file name); Core ML may add a
# sibling com.apple.* directory under the same name.
caches_of() {
  local id="$1"
  [ -d "$CACHES_DIR/$id" ] && echo "$CACHES_DIR/$id"
  return 0
}

valid_id() { [[ "$1" =~ ^qvac-bench-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]]; }

cmd_new() {
  local src="" dest=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --dest) dest="$2"; shift 2 ;;
      -*) die "unknown option $1" ;;
      *) [ -z "$src" ] || die "one binary only"; src="$1"; shift ;;
    esac
  done
  [ -n "$src" ] || usage
  [ -f "$src" ] && [ -x "$src" ] || die "not an executable: $src"
  command -v codesign >/dev/null || die "codesign not found (macOS only)"

  local uuid id dir bin srcdir
  uuid=$(uuidgen | tr 'A-Z' 'a-z')
  id="$PREFIX$uuid"
  dir="${dest:-$HOSTS_DIR}/$id"
  mkdir -p "$dir"
  bin="$dir/$id"
  cp "$src" "$bin"
  chmod u+w "$bin"
  srcdir=$(cd "$(dirname "$src")" && pwd -P)

  # Relative rpaths would resolve against the copy's directory.
  if otool -l "$bin" | awk '/cmd LC_RPATH/ {getline; getline; print $2}' | grep -qE '^@(loader|executable)_path'; then
    install_name_tool -add_rpath "$srcdir" "$bin" 2>/dev/null || true
  fi

  codesign -s - -f -i "$id" "$bin" 2>/dev/null || die "codesign failed for $bin"
  local got
  got=$(codesign -dv "$bin" 2>&1 | awk -F= '/^Identifier=/ {print $2}')
  [ "$got" = "$id" ] || die "identifier is '$got', expected '$id'"

  printf '{"id":%s,"binary":%s,"identifier":%s,"cache":%s}\n' \
    "$(json_str "$id")" "$(json_str "$bin")" "$(json_str "$got")" "$(json_str "$CACHES_DIR/$id")"
}

resolve_id() { # <id|binary path> -> id
  local a="$1"
  if valid_id "$a"; then echo "$a"; return; fi
  local b; b=$(basename "$a")
  valid_id "$b" || die "not a fresh-host id or copy: $a"
  echo "$b"
}

remove_tree() { # refuses anything that is not a qvac-bench-* directory
  local p="$1"
  case "$(basename "$p")" in
    "$PREFIX"*) rm -rf -- "$p" ;;
    *) die "refusing to delete $p" ;;
  esac
}

cmd_cleanup() {
  [ $# -eq 1 ] || usage
  local id; id=$(resolve_id "$1")
  local removed=0 p paths=()
  paths+=("$HOSTS_DIR/$id")
  # One path per line: a path may contain spaces.
  while IFS= read -r p; do paths+=("$p"); done < <(caches_of "$id")
  for p in "${paths[@]}"; do
    [ -e "$p" ] || continue
    remove_tree "$p"; removed=$((removed + 1))
  done
  # A copy made with --dest: the binary's own directory.
  if [ "$1" != "$id" ] && [ -e "$1" ]; then
    local d; d=$(cd "$(dirname "$1")" && pwd -P)
    [ "$(basename "$d")" = "$id" ] && { remove_tree "$d"; removed=$((removed + 1)); }
  fi
  printf '{"id":%s,"removed":%d}\n' "$(json_str "$id")" "$removed"
}

cmd_cleanup_all() {
  local removed=0 p
  shopt -s nullglob
  for p in "$HOSTS_DIR"/"$PREFIX"* "$CACHES_DIR"/"$PREFIX"*; do
    remove_tree "$p"; removed=$((removed + 1))
  done
  printf '{"removed":%d}\n' "$removed"
}

cmd_caches() {
  [ $# -eq 1 ] || usage
  local id; id=$(resolve_id "$1")
  caches_of "$id"
}

[ $# -ge 1 ] || usage
sub="$1"; shift
case "$sub" in
  new) cmd_new "$@" ;;
  cleanup) cmd_cleanup "$@" ;;
  cleanup-all) cmd_cleanup_all "$@" ;;
  caches) cmd_caches "$@" ;;
  -h|--help|help) usage ;;
  *) die "unknown command $sub" ;;
esac
