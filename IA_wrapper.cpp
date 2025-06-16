#include "IA_wrapper.h"
#include "ml_engine.h"
#include "ml_helpers.h"
#include "RL_agent.h"
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>

namespace ia {

// Variável de controle do reporter
static std::atomic<bool> g_stop_reporter(false);
static std::atomic<uint64_t> g_range_start(1);
static std::atomic<uint64_t> g_range_end(0xFFFFFFFFFFFFFFFFULL);
static std::atomic<uint64_t> g_stride(1);
static std::atomic<uint64_t> g_current(1);

float combined_key_score(const std::string &privkey_hex) {
    FeatureSet f = extract_features(privkey_hex);
    return MLEngine::ml_predict(f.to_vector());
}

void set_range_limits(uint64_t start, uint64_t end, uint64_t stride) {
    if (stride == 0) stride = 1;
    g_range_start.store(start);
    g_range_end.store(end);
    g_stride.store(stride);
    g_current.store(start);
}

Range next_range() {
    Range r;
    uint64_t cur = g_current.load();
    if (cur > g_range_end.load()) {
        r.from = r.to = 0;
        r.stride = g_stride.load();
        return r;
    }

    r.from = cur;
    uint64_t block = 0xFFFFFULL * g_stride.load();
    uint64_t tentative = cur + block - g_stride.load();
    uint64_t end = g_range_end.load();
    r.to = tentative > end ? end : tentative;
    r.stride = g_stride.load();
    r.min_score = 0.8f;
    g_current.store((r.to >= end) ? g_range_start.load() : r.to + g_stride.load());

    static auto last_log = std::chrono::steady_clock::now() - std::chrono::seconds(30);
    auto now = std::chrono::steady_clock::now();
    if (now - last_log >= std::chrono::seconds(30)) {
        std::cout << "[IA] Searching range 0x" << std::hex << r.from
                  << " - 0x" << r.to << " (stride " << std::dec << r.stride << ")" << std::endl;
        last_log = now;
    }

    return r;
}

void reward(const Range &, bool, const FeatureSet &) {
    // Aprendizado online ainda não implementado
}

void start_reporter() {
    std::thread([]() {
        while (!g_stop_reporter.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            float avg = MLEngine::ml_recent_score_avg();
            std::cout << "[IA] Média de score nos últimos 30s: " << avg << std::endl;
        }
    }).detach();
}

void stop_reporter() {
    g_stop_reporter.store(true);
}

void init(const std::string &model_path, const std::string &pos_data, const std::string &neg_data) {
    MLEngine::ml_init(model_path, pos_data);
    MLEngine::ml_load_training_data(pos_data, true);
    MLEngine::ml_load_training_data(neg_data, false);
}

} // namespace ia
