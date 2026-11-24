#include "pocket/frontend.h"
#include "json.hpp"
#include <fstream>
#include <cstdio>
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    try {
        tts_cpp::pocket::detail::Frontend f(argv[1]);
        std::ifstream in(argv[2]); nlohmann::json rows; in >> rows;
        int count = 0;
        for (const auto & r : rows) {
            const std::string text = r.at("text");
            auto ids = f.encode(text);
            if (ids != r.at("ids").get<std::vector<int>>()) throw std::runtime_error("token mismatch: "+text+"\nnative="+nlohmann::json(ids).dump()+"\npython="+r.at("ids").dump());
            if (f.decode(ids) != r.at("decoded").get<std::string>()) throw std::runtime_error("decode mismatch: "+text);
            const auto p = f.prepare(text);
            if (p.text != r.at("prepared").get<std::string>() || p.tail_frames != r.at("tail").get<int>()) throw std::runtime_error("prepare mismatch: "+text);
            const auto chunks = f.split(text, 50); const auto & expected = r.at("chunks");
            if (chunks.size() != expected.size()) throw std::runtime_error("chunk count mismatch: "+text);
            for (size_t i = 0; i < chunks.size(); ++i) if (chunks[i].text != expected[i].at("text").get<std::string>() || chunks[i].tail_frames != expected[i].at("tail").get<int>())
                throw std::runtime_error("chunk mismatch: "+text+"\nnative="+chunks[i].text+"\npython="+expected[i].at("text").get<std::string>());
            ++count;
        }
        std::printf("Pocket frontend matches %d upstream corpus entries\n", count); return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
