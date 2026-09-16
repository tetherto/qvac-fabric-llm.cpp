#include "ggml-rpc.h"

#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-impl.h"
#include "transport.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <bcrypt.h>
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)


namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    int32_t use_count;
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_SYNCHRONIZE,
    RPC_CMD_SET_TENSOR_2D_HASH,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

// Protocol minor that introduced RPC_CMD_SET_TENSOR_2D_HASH. The client must not
// send it to a server that announced an older minor at HELLO: such a server
// answers "Unknown command" and closes the connection.
static constexpr uint8_t RPC_PROTO_MINOR_SET_TENSOR_2D_HASH = 1;

// Try a hash lookup first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
const size_t GRAPH_CACHE_MAX = 1024;

static constexpr int RPC_CLIENT_CONNECT_TIMEOUT_MS       = 5000;
static constexpr int RPC_CLIENT_IO_TIMEOUT_MS            = 300000;
static constexpr int RPC_COMM_CONNECT_TIMEOUT_MS         = 5000;
static constexpr int RPC_COMM_CONNECT_ATTEMPT_TIMEOUT_MS = 250;
static constexpr int RPC_COMM_CONNECT_RETRY_MS           = 50;
static constexpr int RPC_COMM_ACCEPT_TIMEOUT_MS          = RPC_COMM_CONNECT_TIMEOUT_MS + 1000;
static constexpr int RPC_COMM_HANDSHAKE_TIMEOUT_MS       = 1000;
static constexpr int RPC_COMM_IO_TIMEOUT_MS              = 30000;
static constexpr size_t RPC_COMM_FULL_DUPLEX_THRESHOLD   = 64 * 1024;
static constexpr size_t RPC_COMM_SESSION_ID_SIZE         = 16;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

struct rpc_msg_synchronize_req {
    uint32_t device;
};

struct rpc_msg_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

struct rpc_msg_set_tensor_2d_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
    uint64_t hash;
};

struct rpc_msg_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t round;
    uint32_t port;      // lower rank listens; higher rank connects
    char     host[64];  // lower rank's host
    uint8_t  session_id[RPC_COMM_SESSION_ID_SIZE];
    uint8_t  wire_bf16;
};

struct rpc_msg_comm_init_rsp {
    uint8_t ok;
};

struct rpc_msg_comm_peer_hello {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t session_id[RPC_COMM_SESSION_ID_SIZE];
};

struct rpc_msg_comm_allreduce_req {
    uint32_t   device;
    uint64_t   op_id;
    rpc_tensor tensor;
};

struct rpc_comm_frame_header {
    uint64_t op_id;
    uint32_t round;
};

struct rpc_msg_comm_free_req {
    uint32_t device;
};

#pragma pack(pop)

static rpc_msg_comm_peer_hello make_comm_peer_hello(const uint8_t * session_id) {
    rpc_msg_comm_peer_hello hello = {
        /*.major   =*/ RPC_PROTO_MAJOR_VERSION,
        /*.minor   =*/ RPC_PROTO_MINOR_VERSION,
        /*.patch   =*/ RPC_PROTO_PATCH_VERSION,
        /*.padding =*/ 0,
        /*.session_id =*/ {},
    };
    memcpy(hello.session_id, session_id, sizeof(hello.session_id));
    return hello;
}

static bool validate_comm_peer_hello(
        const rpc_msg_comm_peer_hello & hello, const uint8_t * session_id) {
    return hello.major == RPC_PROTO_MAJOR_VERSION &&
           hello.minor <= RPC_PROTO_MINOR_VERSION &&
           memcmp(hello.session_id, session_id, sizeof(hello.session_id)) == 0;
}

// RPC data structures

class rpc_command_queue;

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    // uids of graphs cached by the server for this device
    std::unordered_set<uint64_t> graph_uids;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::shared_ptr<rpc_command_queue> cmd_queue;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_command_queue> cmd_queue;
    void * base_ptr;
    uint64_t remote_ptr;
};

// RPC helper functions

static bool checked_mul_size(size_t a, size_t b, size_t & result) {
    if (a != 0 && b > SIZE_MAX / a) {
        return false;
    }
    result = a*b;
    return true;
}

static bool checked_add_size(size_t a, size_t b, size_t & result) {
    if (a > SIZE_MAX - b) {
        return false;
    }
    result = a + b;
    return true;
}

static bool checked_rpc_tensor_size(const rpc_tensor & tensor, size_t & tensor_size) {
    const ggml_type type = (ggml_type) tensor.type;
    const size_t block_size = ggml_blck_size(type);
    if (tensor.ne[0] % block_size != 0) {
        return false;
    }

    int64_t nelements = 1;
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor.ne[i] != 0 && nelements > INT64_MAX / tensor.ne[i]) {
            return false;
        }
        nelements *= tensor.ne[i];
    }

    size_t data_size;
    if (!checked_mul_size(ggml_type_size(type), tensor.ne[0] / block_size, data_size)) {
        return false;
    }
    for (uint32_t i = 1; i < GGML_MAX_DIMS; i++) {
        if (!checked_mul_size(data_size, tensor.ne[i], data_size)) {
            return false;
        }
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor.ne[i] == 0) {
            tensor_size = 0;
            return true;
        }
    }

    if (block_size == 1) {
        tensor_size = ggml_type_size(type);
        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            size_t extent;
            if (!checked_mul_size(tensor.ne[i] - 1, tensor.nb[i], extent) ||
                    !checked_add_size(tensor_size, extent, tensor_size)) {
                return false;
            }
        }
    } else {
        if (!checked_mul_size(tensor.ne[0] / block_size, tensor.nb[0], tensor_size)) {
            return false;
        }
        for (uint32_t i = 1; i < GGML_MAX_DIMS; i++) {
            size_t extent;
            if (!checked_mul_size(tensor.ne[i] - 1, tensor.nb[i], extent) ||
                    !checked_add_size(tensor_size, extent, tensor_size)) {
                return false;
            }
        }
    }
    return true;
}

static bool fill_secure_random(uint8_t * data, size_t size) {
#ifdef _WIN32
    if (size > UINT32_MAX) {
        return false;
    }
    return BCryptGenRandom(nullptr, data, (ULONG) size, BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
    std::ifstream source("/dev/urandom", std::ios::binary);
    source.read((char *) data, size);
    return source && source.gcount() == (std::streamsize) size;
#endif
}

static bool checked_span_size(size_t size, size_t n_copies, size_t stride) {
    return n_copies == 0 || stride == 0 || n_copies - 1 <= (SIZE_MAX - size) / stride;
}

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint64_t fnv_prime = 0x100000001b3ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    if (!sock->send_data(msg, msg_size)) {
        return false;
    }
    return sock->flush();
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
// On success `response` holds the version the server announced; the caller
// uses it to avoid commands newer than the server's minor.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock, rpc_msg_hello_rsp & response) {
    rpc_msg_hello_req request = {};
    response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
    return true;
}

struct rpc_completion {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    done    = false;
    bool                    success = false;
    std::vector<uint8_t>    response;

    void signal(bool value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!done) {
            success = value;
            done    = true;
            cv.notify_all();
        }
    }

    bool wait() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return done; });
        return success;
    }
};

struct rpc_pending_copy {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    done    = false;
    bool                    success = false;
    std::vector<uint8_t>    data;

    void signal(bool value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!done) {
            success = value;
            done    = true;
            cv.notify_all();
        }
    }

    bool wait_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [this] { return done; }) && success;
    }
};

struct rpc_pending_get_2d {
    void *               data;
    size_t               size;
    size_t               n_copies;
    size_t               stride;
    std::vector<uint8_t> packed;

    void unpack() {
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((char *) data + i * stride, packed.data() + i * size, size);
        }
    }
};

enum rpc_queue_cmd_type {
    RPC_QCMD_RPC,
    RPC_QCMD_CPY_GET,
};

struct rpc_queue_cmd {
    rpc_queue_cmd_type                  type     = RPC_QCMD_RPC;
    rpc_cmd                             command  = RPC_CMD_COUNT;
    bool                                response = false;
    std::vector<uint8_t>                data;
    void *                              output      = nullptr;
    size_t                              output_size = 0;
    std::shared_ptr<rpc_completion>     completion;
    std::shared_ptr<rpc_pending_copy>   pending_copy;
    std::shared_ptr<rpc_pending_get_2d> pending_get_2d;
};

class rpc_command_queue {
  public:
    static std::shared_ptr<rpc_command_queue> create(const std::string & endpoint) {
        std::string host;
        int         port;
        if (!parse_endpoint(endpoint, host, port)) {
            GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
            return nullptr;
        }
        if (!rpc_transport_init()) {
            return nullptr;
        }
        auto sock = socket_t::connect(host.c_str(), port, RPC_CLIENT_CONNECT_TIMEOUT_MS);
        rpc_msg_hello_rsp hello = {};
        if (sock == nullptr || !sock->set_timeout(RPC_CLIENT_IO_TIMEOUT_MS) || !negotiate_hello(sock, hello)) {
            return nullptr;
        }
        auto queue    = std::shared_ptr<rpc_command_queue>(new rpc_command_queue(endpoint, std::move(sock), hello.minor));
        queue->worker = std::thread(&rpc_command_queue::worker_loop, queue.get());
        LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
        return queue;
    }

    // Protocol minor the server announced at HELLO. negotiate_hello() already
    // rejected majors other than ours and minors newer than ours, so this is
    // in [0, RPC_PROTO_MINOR_VERSION]; gate commands added in a later minor on it.
    uint8_t server_minor() const { return server_proto_minor; }

    ~rpc_command_queue() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown = true;
        }
        cv.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
    }

    bool submit_rpc(rpc_cmd command, const void * input, size_t input_size) {
        // Own the payload before returning. Transport failures poison the queue
        // and are reported by the next synchronization, readback, or submission.
        rpc_queue_cmd queue_cmd;
        queue_cmd.command = command;
        if (input_size > 0) {
            queue_cmd.data.resize(input_size);
            memcpy(queue_cmd.data.data(), input, input_size);
        }
        return enqueue(std::move(queue_cmd));
    }

    void busy_spin_acquire() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            busy_spin_users++;
        }
        cv.notify_one();
    }

    void busy_spin_release() {
        std::lock_guard<std::mutex> lock(mutex);
        GGML_ASSERT(busy_spin_users > 0);
        busy_spin_users--;
    }

    std::shared_ptr<rpc_completion> submit_rpc_deferred(rpc_cmd      command,
                                                        const void * input,
                                                        size_t       input_size,
                                                        void *       output,
                                                        size_t       output_size) {
        auto          completion = std::make_shared<rpc_completion>();
        rpc_queue_cmd queue_cmd;
        queue_cmd.command  = command;
        queue_cmd.response = true;
        if (input_size > 0) {
            queue_cmd.data.resize(input_size);
            memcpy(queue_cmd.data.data(), input, input_size);
        }
        queue_cmd.output      = output;
        queue_cmd.output_size = output_size;
        queue_cmd.completion  = completion;
        enqueue(std::move(queue_cmd));
        return completion;
    }

    std::shared_ptr<rpc_completion> submit_rpc_deferred_owned(rpc_cmd      command,
                                                              const void * input,
                                                              size_t       input_size,
                                                              size_t       output_size) {
        auto completion = std::make_shared<rpc_completion>();
        completion->response.resize(output_size);
        rpc_queue_cmd queue_cmd;
        queue_cmd.command  = command;
        queue_cmd.response = true;
        if (input_size > 0) {
            queue_cmd.data.resize(input_size);
            memcpy(queue_cmd.data.data(), input, input_size);
        }
        queue_cmd.output      = completion->response.data();
        queue_cmd.output_size = output_size;
        queue_cmd.completion  = completion;
        enqueue(std::move(queue_cmd));
        return completion;
    }

    bool submit_rpc_sync(rpc_cmd command, const void * input, size_t input_size, void * output, size_t output_size) {
        return submit_rpc_deferred(command, input, input_size, output, output_size)->wait();
    }

    bool submit_set_tensor(const rpc_tensor & tensor, const void * data, size_t offset, size_t size) {
        const size_t         input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
        std::vector<uint8_t> input(input_size);
        memcpy(input.data(), &tensor, sizeof(tensor));
        memcpy(input.data() + sizeof(tensor), &offset, sizeof(offset));
        memcpy(input.data() + sizeof(tensor) + sizeof(offset), data, size);
        return submit_rpc(RPC_CMD_SET_TENSOR, input.data(), input.size());
    }

    bool submit_get_tensor(const rpc_msg_get_tensor_req & request, void * output, size_t output_size) {
        rpc_queue_cmd queue_cmd;
        queue_cmd.command  = RPC_CMD_GET_TENSOR;
        queue_cmd.response = true;
        queue_cmd.data.resize(sizeof(request));
        memcpy(queue_cmd.data.data(), &request, sizeof(request));
        queue_cmd.output      = output;
        queue_cmd.output_size = output_size;
        return enqueue(std::move(queue_cmd));
    }

    bool submit_get_tensor_2d(const rpc_msg_get_tensor_2d_req & request,
                              void *                            output,
                              size_t                            output_size,
                              size_t                            size,
                              size_t                            n_copies,
                              size_t                            stride) {
        rpc_queue_cmd queue_cmd;
        queue_cmd.command  = RPC_CMD_GET_TENSOR_2D;
        queue_cmd.response = true;
        queue_cmd.data.resize(sizeof(request));
        memcpy(queue_cmd.data.data(), &request, sizeof(request));
        queue_cmd.output_size = output_size;
        if (stride == size) {
            queue_cmd.output = output;
        } else {
            auto pending = std::make_shared<rpc_pending_get_2d>(
                rpc_pending_get_2d{ output, size, n_copies, stride, std::vector<uint8_t>(output_size) });
            queue_cmd.output         = pending->packed.data();
            queue_cmd.pending_get_2d = std::move(pending);
        }
        return enqueue(std::move(queue_cmd));
    }

    bool submit_cpy_get(const rpc_msg_get_tensor_req & request, const std::shared_ptr<rpc_pending_copy> & pending) {
        rpc_queue_cmd queue_cmd;
        queue_cmd.type    = RPC_QCMD_CPY_GET;
        queue_cmd.command = RPC_CMD_GET_TENSOR;
        queue_cmd.data.resize(sizeof(request));
        memcpy(queue_cmd.data.data(), &request, sizeof(request));
        queue_cmd.pending_copy = pending;
        return enqueue(std::move(queue_cmd));
    }

    bool submit_cpy_set(const rpc_tensor & tensor, const std::shared_ptr<rpc_pending_copy> & pending) {
        std::lock_guard<std::mutex> lock(submit_mutex);
        if (!pending->wait_for(std::chrono::milliseconds(RPC_CLIENT_IO_TIMEOUT_MS))) {
            return false;
        }
        const size_t input_size = sizeof(tensor) + sizeof(uint64_t) + pending->data.size();
        rpc_queue_cmd queue_cmd;
        queue_cmd.command = RPC_CMD_SET_TENSOR;
        queue_cmd.data.resize(input_size);
        memcpy(queue_cmd.data.data(), &tensor, sizeof(tensor));
        uint64_t offset = 0;
        memcpy(queue_cmd.data.data() + sizeof(tensor), &offset, sizeof(offset));
        memcpy(queue_cmd.data.data() + sizeof(tensor) + sizeof(offset), pending->data.data(), pending->data.size());
        return enqueue_locked(std::move(queue_cmd));
    }

    std::shared_ptr<rpc_completion> submit_synchronize(uint32_t device) {
        rpc_msg_synchronize_req request = { device };
        return submit_rpc_deferred(RPC_CMD_SYNCHRONIZE, &request, sizeof(request), nullptr, 0);
    }

    bool synchronize(uint32_t device) { return submit_synchronize(device)->wait(); }

    bool is_failed() {
        std::lock_guard<std::mutex> lock(mutex);
        return failed;
    }

    bool get_cached_alloc_size(const std::string & key, uint64_t & size) {
        std::lock_guard<std::mutex> lock(alloc_cache_mutex);
        auto                        it = alloc_cache.find(key);
        if (it == alloc_cache.end()) {
            return false;
        }
        size = it->second;
        return true;
    }

    void cache_alloc_size(std::string key, uint64_t size) {
        std::lock_guard<std::mutex> lock(alloc_cache_mutex);
        if (alloc_cache.size() >= 4096) {
            alloc_cache.clear();
        }
        alloc_cache.emplace(std::move(key), size);
    }

  private:
    rpc_command_queue(std::string endpoint, socket_ptr sock, uint8_t server_proto_minor) :
        endpoint(std::move(endpoint)), sock(std::move(sock)), server_proto_minor(server_proto_minor) {}

    static void signal_failed(rpc_queue_cmd & cmd) {
        if (cmd.pending_copy) {
            cmd.pending_copy->signal(false);
        }
        if (cmd.completion) {
            cmd.completion->signal(false);
        }
    }

    bool enqueue(rpc_queue_cmd && cmd) {
        std::lock_guard<std::mutex> lock(submit_mutex);
        return enqueue_locked(std::move(cmd));
    }

    bool enqueue_locked(rpc_queue_cmd && cmd) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (shutdown || failed) {
                signal_failed(cmd);
                return false;
            }
            commands.push_back(std::move(cmd));
        }
        cv.notify_one();
        return true;
    }

    bool execute(rpc_queue_cmd & cmd) {
        if (cmd.type == RPC_QCMD_CPY_GET) {
            bool status = send_rpc_cmd(sock, cmd.command, cmd.data.data(), cmd.data.size(),
                                       cmd.pending_copy->data.data(), cmd.pending_copy->data.size());
            cmd.pending_copy->signal(status);
            return status;
        }

        bool status;
        if (cmd.response) {
            status = send_rpc_cmd(sock, cmd.command, cmd.data.data(), cmd.data.size(), cmd.output, cmd.output_size);
            if (status && cmd.pending_get_2d) {
                cmd.pending_get_2d->unpack();
            }
        } else {
            status = send_rpc_cmd(sock, cmd.command, cmd.data.data(), cmd.data.size());
        }
        if (cmd.completion) {
            cmd.completion->signal(status);
        }
        return status;
    }

    void worker_loop() {
        while (true) {
            rpc_queue_cmd cmd;
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (busy_spin_users == 0) {
                    cv.wait(lock, [this] { return shutdown || !commands.empty() || busy_spin_users != 0; });
                }
                if (shutdown && commands.empty()) {
                    return;
                }
                if (commands.empty()) {
                    lock.unlock();
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
                    __asm__ volatile("yield" ::: "memory");
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__clang__) || defined(__GNUC__))
                    __asm__ volatile("pause" ::: "memory");
#else
                    std::this_thread::yield();
#endif
                    continue;
                }
                cmd = std::move(commands.front());
                commands.pop_front();
            }
            if (execute(cmd)) {
                continue;
            }

            std::deque<rpc_queue_cmd> pending;
            {
                std::lock_guard<std::mutex> lock(mutex);
                failed = true;
                pending.swap(commands);
            }
            signal_failed(cmd);
            for (auto & queued : pending) {
                signal_failed(queued);
            }
            GGML_LOG_ERROR("RPC command queue failed for endpoint %s\n", endpoint.c_str());
            return;
        }
    }

    std::string                               endpoint;
    socket_ptr                                sock;
    uint8_t                                   server_proto_minor = 0;
    std::thread                               worker;
    std::mutex                                mutex;
    std::condition_variable                   cv;
    std::deque<rpc_queue_cmd>                 commands;
    bool                                      shutdown = false;
    bool                                      failed   = false;
    unsigned                                  busy_spin_users = 0; // protected by mutex
    std::mutex                                submit_mutex;
    std::mutex                                alloc_cache_mutex;
    std::unordered_map<std::string, uint64_t> alloc_cache;
};

struct rpc_connection_attempt {
    std::condition_variable            cv;
    std::shared_ptr<rpc_command_queue> queue;
    bool                               keep_alive = false;
    bool                               done       = false;
};

struct rpc_connection_cache {
    std::mutex                                                                  mutex;
    std::unordered_map<std::string, std::weak_ptr<rpc_command_queue>>           queues;
    std::unordered_map<std::string, std::shared_ptr<rpc_command_queue>>         prefetched;
    std::unordered_map<std::string, std::shared_ptr<rpc_connection_attempt>>    connecting;
};

static rpc_connection_cache & get_connection_cache() {
    static rpc_connection_cache cache;
    return cache;
}

static std::shared_ptr<rpc_command_queue> get_cached_command_queue_locked(rpc_connection_cache & cache, const std::string & endpoint, bool keep_alive) {
    auto prefetch_it = cache.prefetched.find(endpoint);
    if (prefetch_it != cache.prefetched.end()) {
        if (!prefetch_it->second->is_failed()) {
            auto queue = prefetch_it->second;
            if (!keep_alive) {
                cache.prefetched.erase(prefetch_it);
            }
            return queue;
        }
        cache.prefetched.erase(prefetch_it);
    }

    auto it = cache.queues.find(endpoint);
    if (it != cache.queues.end()) {
        if (auto queue = it->second.lock()) {
            if (!queue->is_failed()) {
                if (keep_alive) {
                    cache.prefetched[endpoint] = queue;
                }
                return queue;
            }
        }
    }
    return nullptr;
}

static std::shared_ptr<rpc_command_queue> get_command_queue(const std::string & endpoint, bool keep_alive = false) {
    auto & cache = get_connection_cache();

    std::shared_ptr<rpc_connection_attempt> attempt;
    bool                                    is_connector = false;
    {
        std::unique_lock<std::mutex> lock(cache.mutex);
        if (auto queue = get_cached_command_queue_locked(cache, endpoint, keep_alive)) {
            return queue;
        }

        auto it = cache.connecting.find(endpoint);
        if (it == cache.connecting.end()) {
            attempt = std::make_shared<rpc_connection_attempt>();
            attempt->keep_alive = keep_alive;
            cache.connecting.emplace(endpoint, attempt);
            is_connector = true;
        } else {
            attempt = it->second;
            attempt->keep_alive = attempt->keep_alive || keep_alive;
        }
    }

    if (!is_connector) {
        std::unique_lock<std::mutex> lock(cache.mutex);
        attempt->cv.wait(lock, [&] { return attempt->done; });
        auto queue = attempt->queue;
        if (queue && !keep_alive) {
            cache.prefetched.erase(endpoint);
        }
        return queue;
    }

    auto queue = rpc_command_queue::create(endpoint);

    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (queue) {
            cache.queues[endpoint] = queue;
            if (attempt->keep_alive) {
                cache.prefetched[endpoint] = queue;
            }
        }
        attempt->queue = queue;
        attempt->done  = true;
        cache.connecting.erase(endpoint);
    }
    attempt->cv.notify_all();
    return queue;
}

static void release_prefetched_command_queue(const std::string & endpoint) {
    auto & cache = get_connection_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.prefetched.erase(endpoint);
}

static void normalize_alloc_cache_tensor(rpc_tensor & tensor) {
    tensor.id = tensor.id != 0;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        tensor.src[i] = tensor.src[i] != 0;
    }
    tensor.view_src = tensor.view_src != 0;
}

static std::string make_alloc_cache_key(rpc_msg_get_alloc_size_req request) {
    normalize_alloc_cache_tensor(request.tensor);
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        normalize_alloc_cache_tensor(request.srcs[i]);
    }
    return std::string((const char *) &request, sizeof(request));
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response,
                                                  sizeof(response));
    RPC_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor *                        tensor,
                                   const std::shared_ptr<rpc_command_queue> & cmd_queue = nullptr) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && tensor->data && ggml_backend_buffer_is_multi_buffer(buffer)) {
        buffer = ggml_backend_multi_buffer_get_buffer(buffer, tensor->data);
    }
    if (buffer && ggml_backend_buffer_is_rpc(buffer)) {
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        if (ctx != nullptr && (cmd_queue == nullptr || ctx->cmd_queue == cmd_queue)) {
            result.buffer = ctx->remote_ptr;
            result.data   = reinterpret_cast<uint64_t>(tensor->data);
        } else {
            result.buffer = 0;
            result.data   = 0;
        }
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.use_count = 0;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_memset_tensor_req request = {
        /* .tensor = */ serialize_tensor(tensor),
        /* .offset = */ offset,
        /* .size   = */ size,
        /* .value  = */ value,
    };
    bool status = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_MEMSET_TENSOR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (size > HASH_THRESHOLD) {
        rpc_msg_set_tensor_hash_req request;
        request.tensor = rpc_tensor;
        request.offset = offset;
        request.hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        bool status = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response,
                                                      sizeof(response));
        RPC_STATUS_ASSERT(status);
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
    }
    bool status = ctx->cmd_queue->submit_set_tensor(rpc_tensor, data, offset, size);
    RPC_STATUS_ASSERT(status);
}

static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    const size_t header_size = sizeof(rpc_tensor) + 4*sizeof(uint64_t);
    size_t data_size;
    size_t input_size;
    GGML_ASSERT(checked_mul_size(size, n_copies, data_size));
    GGML_ASSERT(checked_add_size(header_size, data_size, input_size));
    GGML_ASSERT(checked_span_size(size, n_copies, stride_tensor));
    GGML_ASSERT(checked_span_size(size, n_copies, stride_data));
    std::vector<uint8_t> input(input_size, 0);
    uint8_t * dest = input.data();
    memcpy(dest, &rpc_tensor, sizeof(rpc_tensor));
    dest += sizeof(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, sizeof(header));
    dest += sizeof(header);
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (const char *)data + i*stride_data, size);
    }
    // a 108.0 server does not know RPC_CMD_SET_TENSOR_2D_HASH; send it the plain transfer
    if (data_size > HASH_THRESHOLD && ctx->cmd_queue->server_minor() >= RPC_PROTO_MINOR_SET_TENSOR_2D_HASH) {
        rpc_msg_set_tensor_2d_hash_req request;
        request.tensor   = rpc_tensor;
        request.offset   = offset;
        request.size     = size;
        request.n_copies = n_copies;
        request.stride   = stride_tensor;
        request.hash     = fnv_hash(dest, data_size);
        rpc_msg_set_tensor_hash_rsp response;
        bool status = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_SET_TENSOR_2D_HASH, &request, sizeof(request), &response,
                                                      sizeof(response));
        RPC_STATUS_ASSERT(status);
        if (response.result) {
            return;
        }
    }
    bool status = ctx->cmd_queue->submit_rpc(RPC_CMD_SET_TENSOR_2D, input.data(), input.size());
    RPC_STATUS_ASSERT(status);
}

static void ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_2d_req request;
    request.tensor   = serialize_tensor(tensor);
    request.offset   = offset;
    request.size     = size;
    request.n_copies = n_copies;
    request.stride   = stride_tensor;
    size_t output_size;
    GGML_ASSERT(checked_mul_size(size, n_copies, output_size));
    GGML_ASSERT(checked_span_size(size, n_copies, stride_tensor));
    GGML_ASSERT(checked_span_size(size, n_copies, stride_data));
    if (stride_data == size) {
        bool status =
            ctx->cmd_queue->submit_rpc_sync(RPC_CMD_GET_TENSOR_2D, &request, sizeof(request), data, output_size);
        RPC_STATUS_ASSERT(status);
    } else {
        std::vector<uint8_t> packed(output_size);
        bool status = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_GET_TENSOR_2D, &request, sizeof(request), packed.data(),
                                                      packed.size());
        RPC_STATUS_ASSERT(status);
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((char *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status    = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    RPC_STATUS_ASSERT(status);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->cmd_queue != dst_ctx->cmd_queue) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response,
                                                      sizeof(response));
        RPC_STATUS_ASSERT(status);
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = ctx->cmd_queue->submit_rpc_sync(RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request;
    request.device = buft_ctx->device;
    request.size = size;
    rpc_msg_alloc_buffer_rsp response;
    auto                                   cmd_queue = get_command_queue(buft_ctx->endpoint);
    if (cmd_queue == nullptr) {
        return nullptr;
    }
    bool status =
        cmd_queue->submit_rpc_sync(RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
            buft, ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{ cmd_queue, nullptr, response.remote_ptr }, response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<rpc_command_queue> & cmd_queue, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool                      status =
        cmd_queue->submit_rpc_sync(RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<rpc_command_queue> & cmd_queue, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool                     status =
        cmd_queue->submit_rpc_sync(RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_backend_op_alloc_size_may_expand(tensor->op);

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        auto                                   cmd_queue = get_command_queue(buft_ctx->endpoint);
        RPC_STATUS_ASSERT(cmd_queue != nullptr);

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor, cmd_queue),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i], cmd_queue);
        }

        std::string                cache_key = make_alloc_cache_key(request);
        rpc_msg_get_alloc_size_rsp response;
        if (cmd_queue->get_cached_alloc_size(cache_key, response.alloc_size)) {
            return response.alloc_size;
        }
        bool status =
            cmd_queue->submit_rpc_sync(RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        cmd_queue->cache_alloc_size(std::move(cache_key), response.alloc_size);

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    if (rpc_ctx->cmd_queue != nullptr) {
        rpc_ctx->cmd_queue->synchronize(rpc_ctx->device);
    }
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    RPC_STATUS_ASSERT(rpc_ctx->cmd_queue != nullptr && rpc_ctx->cmd_queue->synchronize(rpc_ctx->device));
}

static void add_tensor(const ggml_cgraph * cgraph, ggml_tensor * tensor,
                       const std::shared_ptr<rpc_command_queue> & cmd_queue, std::vector<rpc_tensor> & tensors,
                       std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(cgraph, tensor->src[i], cmd_queue, tensors, visited);
    }
    add_tensor(cgraph, tensor->view_src, cmd_queue, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, cmd_queue);
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_count = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

static void serialize_graph(uint32_t                                   device,
                            const ggml_cgraph *                        cgraph,
                            const std::shared_ptr<rpc_command_queue> & cmd_queue,
                            std::vector<uint8_t> &                     output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph, cgraph->nodes[i], cmd_queue, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &cgraph->uid, sizeof(cgraph->uid));
    dest += sizeof(cgraph->uid);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & graph_uids = rpc_dev_ctx->graph_uids;
    bool reuse = cgraph->uid != 0 && graph_uids.count(cgraph->uid) > 0;
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        request.uid    = cgraph->uid;
        if (!rpc_ctx->cmd_queue->submit_rpc(RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request))) {
            return GGML_STATUS_FAILED;
        }
    } else {
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, rpc_ctx->cmd_queue, input);
        if (!rpc_ctx->cmd_queue->submit_rpc(RPC_CMD_GRAPH_COMPUTE, input.data(), input.size())) {
            return GGML_STATUS_FAILED;
        }
        if (cgraph->uid != 0) {
            if (graph_uids.size() >= GRAPH_CACHE_MAX) {
                graph_uids.clear();
            }
            graph_uids.insert(cgraph->uid);
        }
    }
    return GGML_STATUS_SUCCESS;
}

struct rpc_event_context {
    std::shared_ptr<rpc_command_queue> cmd_queue;
    std::shared_ptr<rpc_completion>    completion;
    bool                               recorded = false;
};

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * backend_ctx = (ggml_backend_rpc_context *) backend->context;
    rpc_event_context *        event_ctx   = (rpc_event_context *) event->context;
    event_ctx->cmd_queue                   = backend_ctx->cmd_queue;
    event_ctx->completion                  = backend_ctx->cmd_queue->submit_synchronize(backend_ctx->device);
    event_ctx->recorded                    = true;
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * backend_ctx = (ggml_backend_rpc_context *) backend->context;
    rpc_event_context *        event_ctx   = (rpc_event_context *) event->context;
    if (event_ctx->recorded && event_ctx->cmd_queue != backend_ctx->cmd_queue) {
        RPC_STATUS_ASSERT(event_ctx->completion != nullptr && event_ctx->completion->wait());
    }
}

static void ggml_backend_rpc_set_tensor_async(ggml_backend_t backend,
                                              ggml_tensor *  tensor,
                                              const void *   data,
                                              size_t         offset,
                                              size_t         size) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    RPC_STATUS_ASSERT(rpc_ctx->cmd_queue->submit_set_tensor(serialize_tensor(tensor), data, offset, size));
}

static void ggml_backend_rpc_get_tensor_async(ggml_backend_t      backend,
                                              const ggml_tensor * tensor,
                                              void *              data,
                                              size_t              offset,
                                              size_t              size) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    rpc_msg_get_tensor_req     request = {
        /* .tensor = */ serialize_tensor(tensor),
        /* .offset = */ offset,
        /* .size   = */ size,
    };
    RPC_STATUS_ASSERT(rpc_ctx->cmd_queue->submit_get_tensor(request, data, size));
}

static void ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static void ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    rpc_msg_get_tensor_2d_req  request = {
        /* .tensor   = */ serialize_tensor(tensor),
        /* .offset   = */ offset,
        /* .size     = */ size,
        /* .n_copies = */ n_copies,
        /* .stride   = */ stride_tensor,
    };
    RPC_STATUS_ASSERT(
        rpc_ctx->cmd_queue->submit_get_tensor_2d(request, data, size * n_copies, size, n_copies, stride_data));
}

static bool ggml_backend_rpc_cpy_tensor_async(ggml_backend_t      backend_src,
                                              ggml_backend_t      backend_dst,
                                              const ggml_tensor * src,
                                              ggml_tensor *       dst) {
    ggml_backend_rpc_context * dst_ctx = (ggml_backend_rpc_context *) backend_dst->context;
    const size_t               nbytes  = ggml_nbytes(src);

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_synchronize(backend_src);
        return dst_ctx->cmd_queue->submit_set_tensor(serialize_tensor(dst), src->data, 0, nbytes);
    }
    if (!ggml_backend_buffer_is_rpc(src->buffer) || !ggml_backend_buffer_is_rpc(dst->buffer)) {
        return false;
    }

    ggml_backend_rpc_buffer_context * src_ctx     = (ggml_backend_rpc_buffer_context *) src->buffer->context;
    ggml_backend_rpc_buffer_context * dst_buf_ctx = (ggml_backend_rpc_buffer_context *) dst->buffer->context;
    if (src_ctx->cmd_queue == dst_buf_ctx->cmd_queue) {
        rpc_msg_copy_tensor_req request = {
            /* .src = */ serialize_tensor(src),
            /* .dst = */ serialize_tensor(dst),
        };
        rpc_msg_copy_tensor_rsp response;
        bool status = dst_ctx->cmd_queue->submit_rpc_sync(RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response,
                                                          sizeof(response));
        return status && response.result;
    }

    auto pending = std::make_shared<rpc_pending_copy>();
    pending->data.resize(nbytes);
    rpc_msg_get_tensor_req request = {
        /* .tensor = */ serialize_tensor(src),
        /* .offset = */ 0,
        /* .size   = */ nbytes,
    };
    return src_ctx->cmd_queue->submit_cpy_get(request, pending) &&
           dst_buf_ctx->cmd_queue->submit_cpy_set(serialize_tensor(dst), pending);
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_rpc_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    struct buffer_type_owner {
        ggml_backend_rpc_buffer_type_context context;
        ggml_backend_buffer_type             buffer_type;
    };
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // Callers borrow stable pointers for the cache's lifetime. Reclaim the
    // metadata at shutdown without touching devices or making RPC calls.
    static std::unordered_map<std::string, std::unique_ptr<buffer_type_owner>> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return &it->second->buffer_type;
    }
    auto cmd_queue = get_command_queue(endpoint);
    if (cmd_queue == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t                                 alignment = get_alignment(cmd_queue, device);
    size_t                                 max_size  = get_max_size(cmd_queue, device);
    auto owner = std::make_unique<buffer_type_owner>();
    owner->context = {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    owner->buffer_type = {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ &owner->context
    };
    ggml_backend_buffer_type_t buft = &owner->buffer_type;
    buft_map.emplace(buft_name, std::move(owner));
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto        cmd_queue = get_command_queue(endpoint);
    if (cmd_queue == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context{
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ dev_name,
        /* .cmd_queue = */ cmd_queue,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<rpc_command_queue> & cmd_queue,
                              uint32_t                                   device,
                              size_t *                                   free,
                              size_t *                                   total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool                          status =
        cmd_queue->submit_rpc_sync(RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto cmd_queue = get_command_queue(endpoint);
    if (cmd_queue == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(cmd_queue, device, free, total);
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir, std::string bind_host)
        : backends(std::move(all_backends)), bind_host(std::move(bind_host)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool memset_tensor(const rpc_msg_memset_tensor_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_2d(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool set_tensor_2d_hash(const rpc_msg_set_tensor_2d_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool synchronize(const rpc_msg_synchronize_req & request);
    bool comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response);
    bool comm_allreduce(const rpc_msg_comm_allreduce_req & request);
    bool comm_free(const rpc_msg_comm_free_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    void sync_all_backends();
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    // One direct peer connection per recursive-doubling round.
    struct comm_state {
        std::vector<socket_ptr>  peers;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        bool                    wire_bf16 = true;
        uint64_t                next_op_id = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        ggml_backend_buffer_ptr send_host;
        ggml_backend_buffer_ptr recv_host;
        size_t                  host_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    bool comm_allreduce_round(const rpc_msg_comm_allreduce_req & request, comm_state & state, uint32_t round);

    std::vector<ggml_backend_t> backends;
    std::string bind_host;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    const uint64_t tensor_size = ggml_nbytes(tensor);
    if (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        GGML_LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        return false;
    }

    const uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    if (request.tensor.data < buffer_start) {
        GGML_LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        return false;
    }
    const uint64_t data_offset = request.tensor.data - buffer_start;
    if (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        GGML_LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        return false;
    }
    if (tensor->buffer->iface.memset_tensor == nullptr) {
        GGML_LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        return false;
    }

    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", value: %u\n",
            __func__, (void *) tensor->buffer, tensor->data, request.offset, request.size, request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    size_t tensor_size;
    if (!checked_rpc_tensor_size(*tensor, tensor_size)) {
        GGML_LOG_ERROR("[%s] invalid tensor dimensions or strides\n", __func__);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }
    const bool empty_tensor = tensor_size == 0;
    if (result->buffer && !empty_tensor && ggml_backend_buffer_is_multi_buffer(result->buffer)) {
        ggml_backend_buffer_t sub_buffer =
            ggml_backend_multi_buffer_get_buffer(result->buffer, reinterpret_cast<const void *>(tensor->data));
        if (sub_buffer == nullptr) {
            GGML_LOG_ERROR("[%s] tensor '%s' data 0x%" PRIx64 " is not in the multi-buffer\n",
                           __func__, tensor->name, tensor->data);
            return nullptr;
        }
        result->buffer = sub_buffer;
    }

    if (result->buffer && !empty_tensor) {
        // require that the tensor data does not go beyond the buffer end
        const uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        const uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        if (tensor_size > UINT64_MAX - tensor->data || buffer_size > UINT64_MAX - buffer_start) {
            GGML_LOG_ERROR("[%s] tensor or buffer range overflows\n", __func__);
            return nullptr;
        }
        const uint64_t tensor_end = tensor->data + tensor_size;
        const uint64_t buffer_end = buffer_start + buffer_size;
        if (tensor->data < buffer_start || tensor_end > buffer_end) {
            GGML_LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                           "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                           __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                           result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                           tensor->data, tensor_end, buffer_start, buffer_end);
            return nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    if (empty_tensor && result->buffer) {
        // Empty split views have no addressable data and can point outside the local shard.
        result->data = ggml_backend_buffer_get_base(result->buffer);
    }
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    if (cache_dir && size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

static bool validate_tensor_2d_region(const char * func, const rpc_tensor & in_tensor, const ggml_tensor * tensor,
                                      uint64_t offset, uint64_t size, uint64_t n_copies, uint64_t stride) {
    if (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
        return false;
    }
    const uint64_t span = (n_copies - 1)*stride + size;
    const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

    if (in_tensor.data < p0 || in_tensor.data > p1 || offset > p1 - in_tensor.data || span > p1 - in_tensor.data - offset) {
        GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                       func, in_tensor.data, offset, span, p0, p1);
        return false;
    }
    if (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
        GGML_LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                       func, offset, span, ggml_nbytes(tensor));
        return false;
    }
    return true;
}

bool rpc_server::set_tensor_2d(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    if (input.size() < sizeof(rpc_tensor) + 4*sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t header[4];
    memcpy(header, input.data() + sizeof(rpc_tensor), sizeof(header));
    const uint64_t offset   = header[0];
    const uint64_t size     = header[1];
    const uint64_t n_copies = header[2];
    const uint64_t stride   = header[3];

    const uint64_t data_size = input.size() - sizeof(rpc_tensor) - 4*sizeof(uint64_t);
    if (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        return false;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, offset, size, n_copies, stride);

    if (!validate_tensor_2d_region(__func__, *in_tensor, tensor, offset, size, n_copies, stride)) {
        return false;
    }

    const void * data = input.data() + sizeof(rpc_tensor) + 4*sizeof(uint64_t);
    if (cache_dir && data_size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t *) data, data_size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *) data, data_size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::set_tensor_2d_hash(
        const rpc_msg_set_tensor_2d_hash_req & request, rpc_msg_set_tensor_hash_rsp & response) {
    sync_all_backends();
    size_t data_size;
    if (request.n_copies == 0 || request.size == 0 ||
        request.size > SIZE_MAX || request.n_copies > SIZE_MAX ||
        !checked_mul_size((size_t) request.size, (size_t) request.n_copies, data_size)) {
        return false;
    }

    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file) || cached_file.size() != data_size) {
        response.result = 0;
        return true;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", n_copies: %" PRIu64 ", stride: %" PRIu64 ", hash: %" PRIx64 "\n",
            __func__, (void *) tensor->buffer, tensor->data, request.offset, request.size, request.n_copies,
            request.stride, request.hash);

    if (!validate_tensor_2d_region(__func__, request.tensor, tensor, request.offset, request.size, request.n_copies,
                                   request.stride)) {
        return false;
    }

    ggml_backend_tensor_set_2d(tensor, cached_file.data(), request.offset, request.size, request.n_copies,
                               request.stride, request.size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size, request.n_copies, request.stride);

    // sanitize tensor->data
    {
        if (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            return false;
        }
        if (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            return false;
        }
        const uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                           __func__, request.tensor.data, request.offset, span, p0, p1);
            return false;
        }
        if (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            GGML_LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                           __func__, request.offset, span, ggml_nbytes(tensor));
            return false;
        }
        if (request.size * request.n_copies > ggml_nbytes(tensor)) {
            GGML_LOG_ERROR("[%s] packed response size exceeds tensor size\n", __func__);
            return false;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint64_t uid;
    memcpy(&uid, src, sizeof(uid));
    src += sizeof(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    if (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
        if (graph->nodes[i] != nullptr) {
            const size_t hash_pos = ggml_hash_find_or_insert(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(nodes[i])->use_count;
        }
    }
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    auto it = stored_graphs[device].find(request.uid);
    if (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        GGML_LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        return false;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    return true;
}

bool rpc_server::synchronize(const rpc_msg_synchronize_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    ggml_backend_synchronize(backends[request.device]);
    return true;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
void rpc_server::sync_all_backends() {
    for (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link authenticates the peer session before using the same caps negotiation
// as the client HELLO, so it gets the same transport upgrades (e.g. RDMA).
bool rpc_server::comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response) {
    response.ok = 0;
    if (request.device >= backends.size() || request.world < 2 ||
            (request.world & (request.world - 1)) != 0 || request.rank >= request.world ||
            request.round >= 31 || (uint32_t(1) << request.round) >= request.world ||
            request.port == 0 || request.port > UINT16_MAX || request.wire_bf16 > 1) {
        return true;
    }
    comm_state & state = comm_states[request.device];
    if (request.round != state.peers.size() ||
            (request.round > 0 && (state.rank != request.rank || state.world != request.world ||
                                  state.wire_bf16 != (request.wire_bf16 != 0)))) {
        GGML_LOG_WARN("[%s] inconsistent communicator round for device %u\n", __func__, request.device);
        return true;
    }
    socket_ptr connection;
    const uint32_t peer_rank = request.rank ^ (uint32_t(1) << request.round);
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    if (request.rank < peer_rank) {
        socket_ptr srv = socket_t::create_server(bind_host.c_str(), request.port);
        if (srv == nullptr) {
            GGML_LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            return true;
        }
        const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(RPC_COMM_ACCEPT_TIMEOUT_MS);
        while (connection == nullptr) {
            const int remaining_ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now()).count();
            if (remaining_ms <= 0) {
                break;
            }
            socket_ptr peer = srv->accept(remaining_ms);
            if (peer == nullptr) {
                break;
            }
            rpc_msg_comm_peer_hello peer_hello = {};
            if (!peer->set_timeout(RPC_COMM_HANDSHAKE_TIMEOUT_MS) ||
                    !peer->recv_data(&peer_hello, sizeof(peer_hello)) ||
                    !validate_comm_peer_hello(peer_hello, request.session_id)) {
                GGML_LOG_WARN("[%s] rejected communicator peer\n", __func__);
                continue;
            }
            const rpc_msg_comm_peer_hello local_hello = make_comm_peer_hello(request.session_id);
            if (!peer->send_data(&local_hello, sizeof(local_hello)) ||
                    !peer->flush() ||
                    !peer->set_timeout(RPC_COMM_IO_TIMEOUT_MS)) {
                continue;
            }
            connection = std::move(peer);
        }
        if (connection == nullptr) {
            GGML_LOG_ERROR("[%s] timed out waiting for rank %u on comm port %u\n", __func__, peer_rank, request.port);
            return true;
        }
        if (!connection->recv_data(remote_caps, sizeof(remote_caps))) {
            return true;
        }
        connection->get_caps(local_caps);
        if (!connection->send_data(local_caps, sizeof(local_caps)) || !connection->flush()) {
            return true;
        }
        connection->update_caps(remote_caps);
    } else {
        const std::string host(request.host, strnlen(request.host, sizeof(request.host)));
        const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(RPC_COMM_CONNECT_TIMEOUT_MS);
        while (connection == nullptr) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                break;
            }
            const int remaining_ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - now).count();
            socket_ptr peer = socket_t::connect(host.c_str(), request.port,
                    std::min(remaining_ms, RPC_COMM_CONNECT_ATTEMPT_TIMEOUT_MS));
            if (peer != nullptr) {
                const rpc_msg_comm_peer_hello local_hello = make_comm_peer_hello(request.session_id);
                rpc_msg_comm_peer_hello remote_hello = {};
                if (peer->set_timeout(RPC_COMM_HANDSHAKE_TIMEOUT_MS) &&
                        peer->send_data(&local_hello, sizeof(local_hello)) &&
                        peer->flush() &&
                        peer->recv_data(&remote_hello, sizeof(remote_hello)) &&
                        validate_comm_peer_hello(remote_hello, request.session_id) &&
                        peer->set_timeout(RPC_COMM_IO_TIMEOUT_MS)) {
                    connection = std::move(peer);
                }
            }
            if (connection == nullptr) {
                const int retry_ms = std::min(RPC_COMM_CONNECT_RETRY_MS, (int)
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - std::chrono::steady_clock::now()).count());
                if (retry_ms > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(retry_ms));
                }
            }
        }
        if (connection == nullptr) {
            GGML_LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            return true;
        }
        connection->get_caps(local_caps);
        if (!connection->send_data(local_caps, sizeof(local_caps)) ||
            !connection->flush() ||
            !connection->recv_data(remote_caps, sizeof(remote_caps))) {
            return true;
        }
        connection->update_caps(remote_caps);
    }
    state.peers.push_back(std::move(connection));
    state.rank        = request.rank;
    state.world       = request.world;
    state.wire_bf16   = request.wire_bf16 != 0;
    GGML_LOG_INFO("[%s] device %u rank %u connected to rank %u in round %u\n",
                  __func__, request.device, request.rank, peer_rank, request.round);
    response.ok = 1;
    return true;
}

bool rpc_server::comm_allreduce(const rpc_msg_comm_allreduce_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    comm_state & state = comm_states[request.device];
    if (state.world < 2 || (uint64_t(1) << state.peers.size()) != state.world) {
        GGML_LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        return false;
    }
    if (request.op_id != state.next_op_id) {
        GGML_LOG_ERROR("[%s] unexpected operation id %" PRIu64 ", expected %" PRIu64 "\n",
                       __func__, request.op_id, state.next_op_id);
        return false;
    }
    for (uint32_t round = 0; round < state.peers.size(); round++) {
        if (!comm_allreduce_round(request, state, round)) {
            return false;
        }
    }
    state.next_op_id++;
    return true;
}

bool rpc_server::comm_allreduce_round(const rpc_msg_comm_allreduce_req & request, comm_state & state, uint32_t round) {
    const socket_ptr & peer = state.peers[round];
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    if (t_dst == nullptr || t_dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    const size_t  nbytes = ggml_nbytes(t_dst);
    const int64_t ne     = ggml_nelements(t_dst);
    if (nbytes == 0) {
        return true;
    }
    if (t_dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(t_dst) || t_dst->nb[0] != sizeof(float) || ne <= 0 ||
            (uint64_t) ne > SIZE_MAX / sizeof(float) || nbytes != (size_t) ne * sizeof(float)) {
        GGML_LOG_ERROR("[%s] all-reduce tensor must be contiguous F32\n", __func__);
        return false;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    const bool wire_bf16 = state.wire_bf16 && t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    size_t wire_bytes = nbytes;
    size_t need       = nbytes;
    size_t local_offset = 0;
    size_t peer_offset  = 0;
    if (wire_bf16) {
        size_t aligned_wire_end;
        if (!checked_mul_size((size_t) ne, 2, wire_bytes) ||
                !checked_add_size(wire_bytes, alignof(float) - 1, aligned_wire_end)) {
            GGML_LOG_ERROR("[%s] all-reduce scratch size overflows\n", __func__);
            return false;
        }
        local_offset = aligned_wire_end & ~(alignof(float) - 1);
        if (!checked_add_size(local_offset, nbytes, peer_offset) ||
                !checked_add_size(peer_offset, nbytes, need)) {
            GGML_LOG_ERROR("[%s] all-reduce scratch size overflows\n", __func__);
            return false;
        }
    }
    size_t frame_bytes;
    if (!checked_add_size(sizeof(rpc_comm_frame_header), wire_bytes, frame_bytes)) {
        GGML_LOG_ERROR("[%s] all-reduce frame size overflows\n", __func__);
        return false;
    }
    if (state.scratch_size < need) {
        ggml_backend_buffer_ptr scratch { ggml_backend_alloc_buffer(backend, need) };
        if (scratch == nullptr) {
            GGML_LOG_ERROR("[%s] failed to allocate %zu bytes of scratch space\n", __func__, need);
            return false;
        }
        ggml_backend_synchronize(backend);
        state.scratch = std::move(scratch);
        state.scratch_size = need;
    }
    char * scratch_base = (char *) ggml_backend_buffer_get_base(state.scratch.get());
    uint8_t * send_frame = nullptr;
    uint8_t * recv_frame = nullptr;
    static const bool use_pinned_comm = getenv("GGML_RPC_NO_PINNED_COMM") == nullptr;
    ggml_backend_buffer_type_t host_buft =
        use_pinned_comm ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend)) : nullptr;
    if (host_buft != nullptr && state.host_size < frame_bytes) {
        ggml_backend_synchronize(backend);
        state.send_host.reset(ggml_backend_buft_alloc_buffer(host_buft, frame_bytes));
        state.recv_host.reset(ggml_backend_buft_alloc_buffer(host_buft, frame_bytes));
        if (state.send_host != nullptr && state.recv_host != nullptr) {
            state.host_size = frame_bytes;
        } else {
            state.send_host.reset();
            state.recv_host.reset();
            state.host_size = 0;
        }
    }
    if (state.host_size >= frame_bytes) {
        send_frame = (uint8_t *) ggml_backend_buffer_get_base(state.send_host.get());
        recv_frame = (uint8_t *) ggml_backend_buffer_get_base(state.recv_host.get());
    }
    if (send_frame == nullptr || recv_frame == nullptr) {
        state.send_buf.resize(frame_bytes);
        state.recv_buf.resize(frame_bytes);
        send_frame = state.send_buf.data();
        recv_frame = state.recv_buf.data();
    }
    const rpc_comm_frame_header header = {request.op_id, round};
    memcpy(send_frame, &header, sizeof(header));
    uint8_t * send_data = send_frame + sizeof(header);
    uint8_t * recv_data = recv_frame + sizeof(header);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        return t;
    };
    auto new_cpy_node = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        return t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, false);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != nullptr ? 2 : 1;
        ggml_status status = ggml_backend_graph_compute_async(backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    ggml_tensor * t_wire_send = nullptr;
    ggml_tensor * t_wire_recv = nullptr;
    ggml_tensor * t_local     = t_dst;
    ggml_tensor * t_send      = t_dst;
    if (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_local     = new_scratch_tensor(GGML_TYPE_F32, local_offset);
        ggml_tensor * t_to_wire  = new_cpy_node(t_dst, t_wire_send);
        ggml_tensor * t_to_local = new_cpy_node(t_to_wire, t_local);
        compute_nodes(t_to_wire, t_to_local);
        t_send = t_wire_send;
    }
    ggml_backend_tensor_get_async(backend, t_send, send_data, 0, wire_bytes);
    ggml_backend_synchronize(backend);

    bool exchange_ok;
    if (wire_bytes >= RPC_COMM_FULL_DUPLEX_THRESHOLD) {
        exchange_ok = peer->exchange_data(send_frame, recv_frame, frame_bytes);
    } else if ((state.rank & (uint32_t(1) << round)) == 0) {
        exchange_ok = peer->send_data(send_frame, frame_bytes) &&
                      peer->flush() &&
                      peer->recv_data(recv_frame, frame_bytes);
    } else {
        exchange_ok = peer->recv_data(recv_frame, frame_bytes) &&
                      peer->send_data(send_frame, frame_bytes) &&
                      peer->flush();
    }
    if (!exchange_ok) {
        GGML_LOG_ERROR("[%s] peer exchange failed for operation %" PRIu64 " round %u\n", __func__, request.op_id, round);
        return false;
    }

    rpc_comm_frame_header peer_header;
    memcpy(&peer_header, recv_frame, sizeof(peer_header));
    if (peer_header.op_id != request.op_id || peer_header.round != round) {
        GGML_LOG_ERROR("[%s] unexpected peer operation %" PRIu64 " round %u, expected %" PRIu64 " round %u\n",
                       __func__, peer_header.op_id, peer_header.round, request.op_id, round);
        return false;
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, peer_offset);
    ggml_tensor * t_cast = nullptr;
    if (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, recv_data, 0, wire_bytes);
        t_cast = new_cpy_node(t_wire_recv, t_peer);
    } else {
        ggml_backend_tensor_set(t_peer, recv_data, 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_local;
    t_red->src[1] = t_cast != nullptr ? t_cast : t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    if (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } else {
        compute_nodes(t_red, nullptr);
    }
    return true;
}

bool rpc_server::comm_free(const rpc_msg_comm_free_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    ggml_backend_synchronize(backends[request.device]);
    comm_states[request.device] = comm_state();
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

rpc_server::~rpc_server() {
    sync_all_backends();
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             const std::string & bind_host, socket_ptr sock) {
    rpc_server server(backends, cache_dir, bind_host);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) - client needs upgrade to protocol v%d.x\n",
                       (size_t) hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_MEMSET_TENSOR: {
                rpc_msg_memset_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.memset_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_2d(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR_2D: {
                rpc_msg_get_tensor_2d_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor_2d(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_2D_HASH: {
                rpc_msg_set_tensor_2d_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_2d_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SYNCHRONIZE: {
                rpc_msg_synchronize_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.synchronize(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_INIT: {
                rpc_msg_comm_init_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_comm_init_rsp response;
                if (!server.comm_init(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_ALLREDUCE: {
                rpc_msg_comm_allreduce_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_allreduce(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_FREE: {
                rpc_msg_comm_free_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_free(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, host, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ true,
        /* .copy_stream           = */ false,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * dev_ctx   = (ggml_backend_rpc_device_context *) dev->context;
    auto                              cmd_queue = get_command_queue(dev_ctx->endpoint);
    if (cmd_queue == nullptr) {
        return nullptr;
    }
    return new ggml_backend_event{
        /* .device  = */ dev,
        /* .context = */ new rpc_event_context{ std::move(cmd_queue), nullptr, false },
    };
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    delete (rpc_event_context *) event->context;
    delete event;
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    rpc_event_context * event_ctx = (rpc_event_context *) event->context;
    if (event_ctx->recorded) {
        RPC_STATUS_ASSERT(event_ctx->completion != nullptr && event_ctx->completion->wait());
        event_ctx->completion.reset();
        event_ctx->recorded = false;
    }
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

// Recursive-doubling allreduce over direct server-to-server connections.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
struct ggml_backend_rpc_comm_shared_context {
    struct rank_info {
        std::string endpoint;
        uint32_t    device;
        std::shared_ptr<rpc_command_queue> cmd_queue;
    };
    std::vector<rank_info> ranks;
    std::mutex mutex;
    uint64_t next_op_id = 0;
    bool busy_spin = false;

    ~ggml_backend_rpc_comm_shared_context() {
        for (const auto & rank : ranks) {
            rpc_msg_comm_free_req request = {rank.device};
            rank.cmd_queue->submit_rpc(RPC_CMD_COMM_FREE, &request, sizeof(request));
            if (busy_spin) {
                rank.cmd_queue->busy_spin_release();
            }
        }
    }
};

struct ggml_backend_rpc_comm_context {
    std::shared_ptr<ggml_backend_rpc_comm_shared_context> shared;
};

static void ggml_backend_rpc_comm_free(void * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return;
    }
    delete comm_ctx;
}

static void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2 || n_backends > UINT32_MAX || (n_backends & (n_backends - 1)) != 0 ||
            std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        if (n_backends > 1 && (n_backends & (n_backends - 1)) != 0) {
            GGML_LOG_WARN("RPC all-reduce requires a power-of-two rank count, falling back to slow all-reduce\n");
        }
        return nullptr;
    }
    std::vector<ggml_backend_rpc_comm_shared_context::rank_info> ranks;
    ranks.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_rpc(backends[i])) {
            return nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        for (const auto & rank : ranks) {
            if (rank.endpoint == rpc_ctx->endpoint) {
                GGML_LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                return nullptr;
            }
        }
        auto cmd_queue = get_command_queue(rpc_ctx->endpoint);
        if (cmd_queue == nullptr) {
            return nullptr;
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, std::move(cmd_queue)});
    }

    const bool wire_bf16 = std::getenv("GGML_RPC_NO_WIRE_BF16") == nullptr;
    std::string key = wire_bf16 ? "bf16;" : "f32;";
    for (const auto & rank : ranks) {
        key += std::to_string(rank.endpoint.size()) + ":" + rank.endpoint + ":" +
               std::to_string(rank.device) + ";";
    }

    static std::mutex registry_mutex;
    static std::unordered_map<std::string, std::weak_ptr<ggml_backend_rpc_comm_shared_context>> registry;
    std::lock_guard<std::mutex> registry_lock(registry_mutex);

    auto it = registry.find(key);
    if (it != registry.end()) {
        if (auto shared = it->second.lock()) {
            GGML_LOG_INFO("%s: reusing butterfly communicator (%zu ranks)\n", __func__, n_backends);
            return new ggml_backend_rpc_comm_context{std::move(shared)};
        }
    }

    // Each lower rank listens on its own communication port. Reuse that port
    // across rounds; only the accepted peer sockets remain open between rounds.
    uint32_t port_base = 0;
    if (const char * env = std::getenv("GGML_RPC_COMM_PORT")) {
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(env, &end, 10);
        if (end == env || *end != '\0' || parsed == 0 || parsed > 65535 ||
                n_backends - 2 > 65535 - parsed) {
            GGML_LOG_WARN("%s: invalid GGML_RPC_COMM_PORT '%s' for %zu ranks\n", __func__, env, n_backends);
            return nullptr;
        }
        port_base = (uint32_t) parsed;
    }
    std::vector<std::string> hosts(n_backends);
    std::vector<uint32_t> ports(n_backends);
    // The highest rank always connects; no peer needs its host or listener port.
    for (size_t i = 0; i + 1 < n_backends; i++) {
        int rpc_port;
        if (!parse_endpoint(ranks[i].endpoint, hosts[i], rpc_port) || hosts[i].size() >= 64) {
            return nullptr;
        }
        if (port_base == 0 && (rpc_port <= 0 || rpc_port > 65535 - 1000)) {
            GGML_LOG_WARN("%s: RPC port %d + 1000 is out of range; set GGML_RPC_COMM_PORT\n", __func__, rpc_port);
            return nullptr;
        }
        ports[i] = port_base != 0 ? port_base + (uint32_t) i : (uint32_t) rpc_port + 1000;
    }

    std::vector<bool> initialized(n_backends, false);
    auto cleanup = [&]() {
        for (size_t i = 0; i < n_backends; i++) {
            if (initialized[i]) {
                rpc_msg_comm_free_req request = {ranks[i].device};
                ranks[i].cmd_queue->submit_rpc(RPC_CMD_COMM_FREE, &request, sizeof(request));
            }
        }
    };
    uint32_t round = 0;
    for (size_t mask = 1; mask < n_backends; mask <<= 1, round++) {
        // A separate secret per edge and round rejects stale or misrouted peers.
        std::vector<std::array<uint8_t, RPC_COMM_SESSION_ID_SIZE>> sessions(n_backends);
        for (size_t i = 0; i < n_backends; i++) {
            if ((i & mask) == 0 && !fill_secure_random(sessions[i].data(), sessions[i].size())) {
                GGML_LOG_WARN("%s: failed to generate communicator session id\n", __func__);
                cleanup();
                return nullptr;
            }
        }

        // Submit the entire round before waiting: lower ranks block in accept.
        std::vector<std::shared_ptr<rpc_completion>> completions;
        completions.reserve(n_backends);
        for (size_t i = 0; i < n_backends; i++) {
            const size_t listener = i & ~mask;
            rpc_msg_comm_init_req request = {};
            request.device    = ranks[i].device;
            request.rank      = (uint32_t) i;
            request.world     = (uint32_t) n_backends;
            request.round     = round;
            request.port      = ports[listener];
            request.wire_bf16 = wire_bf16;
            memcpy(request.session_id, sessions[listener].data(), sessions[listener].size());
            memcpy(request.host, hosts[listener].c_str(), hosts[listener].size());
            completions.push_back(ranks[i].cmd_queue->submit_rpc_deferred_owned(
                RPC_CMD_COMM_INIT, &request, sizeof(request), sizeof(rpc_msg_comm_init_rsp)));
        }
        bool ok = true;
        for (size_t i = n_backends; i-- > 0;) {
            const auto & completion = completions[i];
            rpc_msg_comm_init_rsp response = {};
            if (completion->wait() && completion->response.size() == sizeof(response)) {
                memcpy(&response, completion->response.data(), sizeof(response));
            }
            if (response.ok) {
                initialized[i] = true;
            } else {
                GGML_LOG_WARN("%s: rank %zu (%s) failed to initialize round %u\n",
                              __func__, i, ranks[i].endpoint.c_str(), round);
                ok = false;
            }
        }
        if (!ok) {
            cleanup();
            return nullptr;
        }
    }
    GGML_LOG_INFO("%s: butterfly communicator initialized (%zu ranks, %u rounds)\n",
                  __func__, n_backends, round);
    auto shared = std::make_shared<ggml_backend_rpc_comm_shared_context>();
    shared->ranks = std::move(ranks);
    // Acquire only after successful initialization, once per shared communicator.
    // Reused handles keep polling alive until the final handle is released.
    shared->busy_spin = std::getenv("GGML_RPC_NO_BUSY_SPIN") == nullptr;
    if (shared->busy_spin) {
        for (const auto & rank : shared->ranks) {
            rank.cmd_queue->busy_spin_acquire();
        }
    }
    registry[key] = shared;
    return new ggml_backend_rpc_comm_context{std::move(shared)};
}

static bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr || comm_ctx->shared == nullptr) {
        return false;
    }
    auto shared = comm_ctx->shared;
    std::lock_guard<std::mutex> lock(shared->mutex);
    const size_t n_ranks = shared->ranks.size();
    if (tensors == nullptr || tensors[0] == nullptr) {
        return false;
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    for (size_t i = 0; i < n_ranks; i++) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguous(tensors[i]) || tensors[i]->nb[0] != sizeof(float) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            return false;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    if (ne == 0) {
        return true;
    }
    const uint64_t op_id = shared->next_op_id;
    for (size_t i = 0; i < n_ranks; i++) {
        rpc_msg_comm_allreduce_req request = {};
        request.device = shared->ranks[i].device;
        request.op_id   = op_id;
        request.tensor = serialize_tensor(tensors[i]);
        if (!shared->ranks[i].cmd_queue->submit_rpc(RPC_CMD_COMM_ALLREDUCE, &request, sizeof(request))) {
            if (i == 0) {
                return false;
            }
            GGML_ABORT("RPC all-reduce dispatch failed");
        }
    }
    shared->next_op_id++;
    return true;
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_prefetch_connection") == 0) {
        return (void *)ggml_backend_rpc_prefetch_connection;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_rpc_comm_init;
    }
    if (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_rpc_comm_free;
    }
    if (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto cmd_queue = get_command_queue(endpoint);
    if (cmd_queue == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = cmd_queue->submit_rpc_sync(RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

// Establishes the connection without registering a backend. This lets callers
// warm many endpoints in parallel, then register them in deterministic order.
bool ggml_backend_rpc_prefetch_connection(const char * endpoint) {
    return get_command_queue(endpoint, true) != nullptr;
}

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    struct device_owner {
        ggml_backend_rpc_device_context context;
        ggml_backend_device             device;
    };
    struct reg_owner {
        ggml_backend_rpc_reg_context                context;
        std::vector<std::unique_ptr<device_owner>> devices;
        ggml_backend_reg                           reg;
    };
    // The API exposes borrowed registry/device pointers. Keep their addresses
    // stable while cached, and destroy all owned metadata at shutdown. These
    // destructors do not depend on the buffer-type or connection caches.
    static std::unordered_map<std::string, std::unique_ptr<reg_owner>> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = reg_map.find(endpoint);
    if (it != reg_map.end()) {
        release_prefetched_command_queue(endpoint);
        return &it->second->reg;
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    auto owner = std::make_unique<reg_owner>();
    owner->context.name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        auto device = std::make_unique<device_owner>();
        device->context = {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
            /* .graph_uids  = */    {},
        };

        device->device = {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ &device->context,
        };
        owner->context.devices.push_back(&device->device);
        owner->devices.push_back(std::move(device));
        dev_id++;
    }
    owner->reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ &owner->context
    };
    ggml_backend_reg_t reg = &owner->reg;
    reg_map.emplace(endpoint, std::move(owner));
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
