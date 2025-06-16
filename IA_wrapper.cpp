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

float combined_key_score(const std::string &privkey_hex) {
    FeatureSet f = extract_features(privkey_hex);
    return MLEngine::ml_predict(f.to_vector());
}

Range next_range() {
    static std::atomic<uint64_t> current(1);
    Range r;
    r.from = current;
    r.to = current + 0xFFFFF;
    r.stride = 1;
    r.min_score = 0.8f;
    current = r.to + 1;
    std::cout << "[IA] Procurando range 0x" << std::hex << r.from
              << " - 0x" << r.to << std::dec << std::endl;
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
