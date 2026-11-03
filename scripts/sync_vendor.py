#!/usr/bin/env python3

import urllib.request
import os
import sys
import subprocess

HTTPLIB_VERSION = "refs/tags/v0.62.0"

# used by examples/gguf-hash, these repos have no release tag, so we pin a commit
XXHASH_COMMIT      = "9f465f1ea932d6ad9a26cd77496311ffa544cd68"
SHA1_COMMIT        = "e1e2536fcf6a8f9703be8c85d58724b408552287"
SHA256_COMMIT      = "5e637272c13f200872d55ff579f7e2ab6c3f252f"
ROTATE_BITS_COMMIT = "27e784942f67db44abf2115c6638e735b579acd1"

vendor = {
    "https://github.com/nlohmann/json/releases/latest/download/json.hpp":     "vendor/nlohmann/json.hpp",
    "https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp": "vendor/nlohmann/json_fwd.hpp",

    "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h": "vendor/stb/stb_image.h",

    # not using latest tag to avoid this issue: https://github.com/ggml-org/llama.cpp/pull/17179#discussion_r2515877926
    # "https://github.com/mackron/miniaudio/raw/refs/tags/0.11.24/miniaudio.h": "vendor/miniaudio/miniaudio.h",
    "https://github.com/mackron/miniaudio/raw/9634bedb5b5a2ca38c1ee7108a9358a4e233f14d/miniaudio.h": "vendor/miniaudio/miniaudio.h",

    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/httplib.h": "httplib.h",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/split.py":  "split.py",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/LICENSE":   "vendor/cpp-httplib/LICENSE",

    "https://raw.githubusercontent.com/sheredom/subprocess.h/0dccaa9aa176dd6d7ef8afeca3c18d6e80a32795/subprocess.h": "vendor/sheredom/subprocess.h",

    f"https://raw.githubusercontent.com/Cyan4973/xxHash/{XXHASH_COMMIT}/xxhash.c":      "vendor/hash/xxhash/xxhash.c",
    f"https://raw.githubusercontent.com/Cyan4973/xxHash/{XXHASH_COMMIT}/xxhash.h":      "vendor/hash/xxhash/xxhash.h",
    f"https://raw.githubusercontent.com/Cyan4973/xxHash/{XXHASH_COMMIT}/LICENSE":       "vendor/hash/xxhash/LICENSE",

    # clibs/sha1 ships no license file, the source header says public domain
    f"https://raw.githubusercontent.com/clibs/sha1/{SHA1_COMMIT}/sha1.c": "vendor/hash/sha1/sha1.c",
    f"https://raw.githubusercontent.com/clibs/sha1/{SHA1_COMMIT}/sha1.h": "vendor/hash/sha1/sha1.h",

    f"https://raw.githubusercontent.com/jb55/sha256.c/{SHA256_COMMIT}/sha256.c": "vendor/hash/sha256/sha256.c",
    f"https://raw.githubusercontent.com/jb55/sha256.c/{SHA256_COMMIT}/sha256.h": "vendor/hash/sha256/sha256.h",
    f"https://raw.githubusercontent.com/jb55/sha256.c/{SHA256_COMMIT}/LICENSE":  "vendor/hash/sha256/LICENSE",

    f"https://raw.githubusercontent.com/jb55/rotate-bits.h/{ROTATE_BITS_COMMIT}/rotate-bits.h": "vendor/hash/rotate-bits/rotate-bits.h",
    f"https://raw.githubusercontent.com/jb55/rotate-bits.h/{ROTATE_BITS_COMMIT}/LICENSE.md":   "vendor/hash/rotate-bits/LICENSE.md",
}

# local changes kept on top of the upstream sources
patches = {
    "vendor/hash/xxhash/xxhash.h": [(
        '#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) /* >= C11 */\n',
        '/* Windows SDK under 10.0.22000 is missing stdalign.h so we add a check\n'
        '   before allowing the windows compiler to use the C11 form.\n'
        '   Reference: https://github.com/Cyan4973/xxHash/issues/955 */\n'
        '#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) \\\n'
        '    && (defined(_MSC_VER) && (_MSC_VER >= 1000) || !defined(_MSC_VER)) /* >= C11 */\n'
    )],

    # sha1 exports a bare "SHA1" symbol, which clashes with the boringssl one at link time.
    # we compile it as C++ (see vendor/hash/CMakeLists.txt) and put it in a namespace.
    "vendor/hash/sha1/sha1.h": [
        (
            '#if defined(__cplusplus)\n'
            'extern "C" {\n'
            '#endif\n',

            'namespace vendor_hash {\n'
        ),
        (
            '#if defined(__cplusplus)\n'
            '}\n'
            '#endif\n',

            '} // namespace vendor_hash\n'
        ),
    ],

    "vendor/hash/sha1/sha1.c": [
        (
            '#include "sha1.h"\n',

            '#include "sha1.h"\n'
            '\n'
            'namespace vendor_hash {\n'
        ),
        (
            '    SHA1Final((unsigned char *)hash_out, &ctx);\n'
            '}\n',

            '    SHA1Final((unsigned char *)hash_out, &ctx);\n'
            '}\n'
            '\n'
            '} // namespace vendor_hash\n'
        ),
    ],

    # silence a maybe-uninitialized warning
    "vendor/hash/sha256/sha256.c": [(
        "  uint32_t W[16];\n",
        "  uint32_t W[16] = {0};\n"
    )],

    # QVAC-26299: bound the PNG zlib decode to the size IHDR declares, so a small
    # PNG with a huge deflated IDAT cannot allocate gigabytes before the size
    # check. Test: tests/test-stb-png-decode-bound.cpp.
    "vendor/stb/stb_image.h": [
        (
            '                                  no warranty implied; use at your own risk\n'
            '\n',

            '                                  no warranty implied; use at your own risk\n'
            '\n'
            '   QVAC local modification, re-applied by scripts/sync_vendor.py: QVAC-26299\n'
            '   bounds the PNG zlib decode to the size implied by IHDR (stbi__png_raw_len,\n'
            '   stbi__zlib_decode_exact_headerflag).\n'
            '\n'
        ),
        (
            '\n'
            'STBIDEF int stbi_zlib_decode_buffer(char *obuffer, int olen, char const *ibuffer, int ilen)\n',

            '\n'
            '#ifndef STBI_NO_PNG\n'
            '// QVAC-26299: fixed-size variant for PNG. The expandable decode above grows\n'
            '// the buffer up to UINT_MAX/2 whatever the image size, so a 1x1 PNG with a\n'
            '// few MB of deflated zeros allocated ~2 GiB. With exp=0 a stream that writes\n'
            '// past olen fails in stbi__zexpand with "output buffer limit".\n'
            'static char *stbi__zlib_decode_exact_headerflag(const char *buffer, int len, int olen, int *outlen, int parse_header)\n'
            '{\n'
            '   stbi__zbuf a;\n'
            '   char *p = (char *) stbi__malloc(olen);\n'
            '   if (p == NULL) return NULL;\n'
            '   a.zbuffer = (stbi_uc *) buffer;\n'
            '   a.zbuffer_end = (stbi_uc *) buffer + len;\n'
            '   if (stbi__do_zlib(&a, p, olen, 0, parse_header)) {\n'
            '      if (outlen) *outlen = (int) (a.zout - a.zout_start);\n'
            '      return a.zout_start;\n'
            '   } else {\n'
            '      STBI_FREE(a.zout_start);\n'
            '      return NULL;\n'
            '   }\n'
            '}\n'
            '#endif\n'
            '\n'
            'STBIDEF int stbi_zlib_decode_buffer(char *obuffer, int olen, char const *ibuffer, int ilen)\n'
        ),
        (
            '   if (!all_ok) return 0;\n'
            '\n'
            '   return 1;\n'
            '}\n',

            '   if (!all_ok) return 0;\n'
            '\n'
            '   return 1;\n'
            '}\n'
            '\n'
            '// QVAC-26299: exact length of the filtered scanlines the zlib stream must\n'
            '// produce, with the same per-pass geometry as stbi__create_png_image below\n'
            '// (empty Adam7 passes contribute nothing, not even filter bytes).\n'
            'static int stbi__png_raw_len(stbi__png *a, int depth, int interlaced, int *raw_len)\n'
            '{\n'
            '   static const int xorig[] = { 0,4,0,2,0,1,0 };\n'
            '   static const int yorig[] = { 0,0,4,0,2,0,1 };\n'
            '   static const int xspc[]  = { 8,8,4,4,2,2,1 };\n'
            '   static const int yspc[]  = { 8,8,8,4,4,2,2 };\n'
            '   int img_n = a->s->img_n;\n'
            '   int total = 0, p, x, y, row, len;\n'
            '   for (p=0; p < (interlaced ? 7 : 1); ++p) {\n'
            '      if (interlaced) {\n'
            '         x = (a->s->img_x - xorig[p] + xspc[p]-1) / xspc[p];\n'
            '         y = (a->s->img_y - yorig[p] + yspc[p]-1) / yspc[p];\n'
            '      } else {\n'
            '         x = a->s->img_x;\n'
            '         y = a->s->img_y;\n'
            '      }\n'
            '      if (x && y) {\n'
            '         if (!stbi__mad3sizes_valid(img_n, x, depth, 7)) return stbi__err("too large", "Corrupt PNG");\n'
            '         row = (((img_n * x * depth) + 7) >> 3) + 1;\n'
            '         if (!stbi__mul2sizes_valid(row, y)) return stbi__err("too large", "Corrupt PNG");\n'
            '         len = row * y;\n'
            '         if (!stbi__addsizes_valid(total, len)) return stbi__err("too large", "Corrupt PNG");\n'
            '         total += len;\n'
            '      }\n'
            '   }\n'
            '   *raw_len = total;\n'
            '   return 1;\n'
            '}\n'
        ),
        (
            "         case STBI__PNG_TYPE('I','E','N','D'): {\n"
            '            stbi__uint32 raw_len, bpl;\n'
            '            if (first) return stbi__err("first not IHDR", "Corrupt PNG");\n',

            "         case STBI__PNG_TYPE('I','E','N','D'): {\n"
            '            stbi__uint32 raw_len;\n'
            '            int exact_len;\n'
            '            if (first) return stbi__err("first not IHDR", "Corrupt PNG");\n'
        ),
        (
            '            if (z->idata == NULL) return stbi__err("no IDAT","Corrupt PNG");\n'
            '            // initial guess for decoded data size to avoid unnecessary reallocs\n'
            '            bpl = (s->img_x * z->depth + 7) / 8; // bytes per line, per component\n'
            '            raw_len = bpl * s->img_y * s->img_n /* pixels */ + s->img_y /* filter mode per row */;\n'
            '            z->expanded = (stbi_uc *) stbi_zlib_decode_malloc_guesssize_headerflag((char *) z->idata, ioff, raw_len, (int *) &raw_len, !is_iphone);\n'
            '            if (z->expanded == NULL) return 0; // zlib should set error\n',

            '            if (z->idata == NULL) return stbi__err("no IDAT","Corrupt PNG");\n'
            '            // QVAC-26299: decode into a buffer of exactly the declared size\n'
            '            // instead of a growable guess, so memory is bounded by IHDR and a\n'
            '            // stream that inflates past it is rejected ("output buffer limit").\n'
            '            // This also rejects the trailing-zeros files that issue #276 let\n'
            '            // through; refusing is intended here.\n'
            '            if (!stbi__png_raw_len(z, z->depth, interlace, &exact_len)) return 0;\n'
            '            z->expanded = (stbi_uc *) stbi__zlib_decode_exact_headerflag((char *) z->idata, ioff, exact_len, (int *) &raw_len, !is_iphone);\n'
            '            if (z->expanded == NULL) return 0; // zlib should set error\n'
        ),
    ],
}

# local changes too large for the replacements above, kept as diffs and applied with git apply
patch_files = [
    # backport of the fix for the stack overflow on deeply nested values (nlohmann/json#5387)
    # TODO: remove once nlohmann/json releases a version newer than 3.12.0
    "vendor/nlohmann/json-deep-nesting.patch",
]

for url, filename in vendor.items():
    print(f"downloading {url} to {filename}") # noqa: NP100
    urllib.request.urlretrieve(url, filename)

for filename, replacements in patches.items():
    print(f"patching {filename}") # noqa: NP100
    with open(filename, "r", encoding="utf-8", newline="") as f:
        content = f.read()
    for old, new in replacements:
        if content.count(old) != 1:
            print(f"Error: cannot apply patch on {filename}, upstream code has changed") # noqa: NP100
            sys.exit(1)
        content = content.replace(old, new)
    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write(content)

for patch_file in patch_files:
    print(f"applying {patch_file}") # noqa: NP100
    try:
        subprocess.check_call(["git", "apply", patch_file])
    except subprocess.CalledProcessError:
        print(f"Error: cannot apply {patch_file}, upstream code has changed") # noqa: NP100
        sys.exit(1)

print("Splitting httplib.h...") # noqa: NP100
try:
    subprocess.check_call([
        sys.executable, "split.py",
        "--extension", "cpp",
        "--out", "vendor/cpp-httplib"
    ])
except Exception as e:
    print(f"Error: {e}") # noqa: NP100
    sys.exit(1)
finally:
    os.remove("split.py")
    os.remove("httplib.h")
