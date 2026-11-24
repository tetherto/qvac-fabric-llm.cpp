// Included once per engine namespace by the generated probe_<ns>.cpp files of
// test-coreml-link (QVAC_COREML_NS is defined by the includer). Each probe
// drives its engine's copy of the runtime end to end with a fake LoadFn.

#include "qvac_coreml/manager.h"

#include <memory>
#include <string>

namespace QVAC_COREML_NS {

namespace {
LoadResult link_probe_load(const LoadRequest &) {
    LoadResult r;
    r.model = std::make_shared<int>(42);
    return r;
}
}  // namespace

const void * link_probe_identity() { return Manager::instance().get(); }

int link_probe() {
    std::shared_ptr<Manager> m = Manager::create();
    SlotSpec spec;
    spec.engine = "link";
    spec.stage = "probe";
    spec.units = ComputeUnits::cpu_only;
    spec.route = Route::prebuilt;
    spec.artifact_path = "/nonexistent/qvac-coreml-link-probe.mlmodelc";
    spec.load_fn = &link_probe_load;
    SlotPolicy policy;
    policy.wait = Wait::call;
    std::shared_ptr<Slot> slot = m->make_slot(spec, policy);
    Acquired a = slot->acquire(Demand::now);
    if (!a || *a.model->as<int>() != 42) return 1;
    Options o;
    o.compute_units = "cpu_and_ane";
    if (resolve_coreml(o, "QVAC_COREML_LINK_PROBE").compute_units != ComputeUnits::cpu_and_ane) return 2;
    if (to_json(slot->status()).find("\"state\":\"ready\"") == std::string::npos) return 3;
    return 0;
}

}  // namespace QVAC_COREML_NS
