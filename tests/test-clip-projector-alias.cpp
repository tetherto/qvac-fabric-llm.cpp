// QVAC-23075: the legacy projector alias table (clip_projector_type_from_alias in
// tools/mtmd/clip-impl.h). VisionPsy's published mmprojs declare clip.projector_type = "custom",
// a name any future model could also pick, so the alias only resolves when general.name matches
// too. That second half is the whole safety property, and it is one `&&` away from being lost.

#include "clip-impl.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

static void expect_alias(const char * proj_type, const char * model_name, projector_type expected) {
    const projector_type got = clip_projector_type_from_alias(proj_type, model_name);
    if (got != expected) {
        std::printf("FAIL alias('%s', '%s'): got %d, expected %d\n",
                    proj_type, model_name, (int) got, (int) expected);
        g_failures++;
        return;
    }
    std::printf("ok   alias('%s', '%s') -> %d\n", proj_type, model_name, (int) got);
}

int main() {
    // The one shipped pair. Both published VisionPsy mmprojs, base and Flash, carry these.
    expect_alias("custom", "VisionPsyNano", PROJECTOR_TYPE_VISIONPSY);

    // Same projector string, any other model: must stay unknown rather than being loaded as
    // VisionPsy, which is what makes "custom" safe to keep claiming.
    expect_alias("custom", "SomeOtherModel", PROJECTOR_TYPE_UNKNOWN);
    expect_alias("custom", "",              PROJECTOR_TYPE_UNKNOWN);
    expect_alias("custom", "visionpsynano", PROJECTOR_TYPE_UNKNOWN); // exact match, not folded

    // Right model name behind a different projector string is not the alias either.
    expect_alias("idefics3", "VisionPsyNano", PROJECTOR_TYPE_UNKNOWN);
    expect_alias("",         "VisionPsyNano", PROJECTOR_TYPE_UNKNOWN);

    // The canonical spelling resolves through the normal table, so an mmproj converted with
    // clip.projector_type = "visionpsy" needs no alias at all.
    if (clip_projector_type_from_string("visionpsy") != PROJECTOR_TYPE_VISIONPSY) {
        std::printf("FAIL clip_projector_type_from_string('visionpsy') does not resolve\n");
        g_failures++;
    } else {
        std::printf("ok   clip_projector_type_from_string('visionpsy')\n");
    }

    // And the alias string is NOT in the canonical table, so it can only ever be reached with
    // the model-name check applied.
    if (clip_projector_type_from_string("custom") != PROJECTOR_TYPE_UNKNOWN) {
        std::printf("FAIL clip_projector_type_from_string('custom') resolves without the name check\n");
        g_failures++;
    } else {
        std::printf("ok   clip_projector_type_from_string('custom') stays unknown\n");
    }

    if (g_failures) {
        std::printf("\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("\nprojector alias resolution is gated on general.name\n");
    return 0;
}
