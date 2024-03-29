#pragma once

#include <memory>
#include <iostream>
#include "cutlass/numeric_types.h"
#include "src/fastertransformer/utils/ini.h"

namespace fastertransformer {

enum class FetchType {
    GPU_ONLY,
    FETCH_ON_DEMAND,
    PREFETCH
};

enum class QuantType {
    NO_QUANT,
    WEIGHT_ONLY,
    SMOOTH_QUANT
};

class GlobalConfig {
public:
    using quant_t = cutlass::fp4_t;

    using weight_t = float;

    using act_t = float;

    static GlobalConfig& instance()
    {
        static GlobalConfig instance;
        return instance;
    }

    void setDefault()
    {
        loadDefault();
    }

    void loadDefault()
    {
        mINI::INIFile file("/workspace/FasterTransformer/cpp_config.ini");
        mINI::INIStructure ini;
        file.read(ini);

        arena_size = std::stoul(ini["default"]["arena_size"]);

        encoder_fetcher_mode = static_cast<FetchType>(std::stoi(ini["default"]["encoder_fetcher_mode"]));
        decoder_fetcher_mode = static_cast<FetchType>(std::stoi(ini["default"]["decoder_fetcher_mode"]));

        profiling = std::stoi(ini["default"]["profiling"]);
        detailed_timing = std::stoi(ini["default"]["detailed_timing"]);

        offload_path = ini["default"]["offload_path"];
        disk_offload = std::stoi(ini["default"]["disk_offload"]);
        
        load_from_cpp = std::stoi(ini["default"]["load_from_cpp"]);

        use_cache = std::stoi(ini["default"]["use_cache"]);

        quant_mode = static_cast<QuantType>(std::stoi(ini["default"]["quant_mode"]));

        vocab_size = std::stoll(ini["default"]["vocab_size"]);

        fetch_all = std::stoi(ini["default"]["fetch_all"]);

        forced_num_experts = std::stoi(ini["default"]["forced_num_experts"]);

        cache_policy = ini["default"]["cache_policy"];
    }

    void print() const
    {
        // TODO: replace with FT_LOG
        std::cout << "arena_size: " << arena_size << std::endl
                  << "encoder_fetcher_mode: " << int(encoder_fetcher_mode) << std::endl
                  << "decoder_fetcher_mode: " << int(decoder_fetcher_mode) << std::endl
                  << "profiling: " << profiling << std::endl
                  << "detailed_timing: " << detailed_timing << std::endl
                  << "offload_path: " << offload_path << std::endl
                  << "disk_offload: " << disk_offload << std::endl
                  << "load_from_cpp: " << load_from_cpp << std::endl
                  << "use_cache: " << use_cache << std::endl
                  << "quant_mode: " << int(quant_mode) << std::endl
                  << "vocab_size: " << vocab_size << std::endl
                  << "fetch_all: " << fetch_all << std::endl
                  << "forced_num_experts: " << forced_num_experts << std::endl
                  << "cache_policy: " << cache_policy << std::endl;
    }


    size_t arena_size;

    FetchType encoder_fetcher_mode;
    FetchType decoder_fetcher_mode;

    bool profiling;
    bool detailed_timing;

    std::string offload_path;
    bool disk_offload;

    bool load_from_cpp;

    bool use_cache;

    QuantType quant_mode;

    int64_t vocab_size;  // workaround for missing vocab_size arg in encoder

    bool fetch_all;  // for SE-MoE

    int forced_num_experts;  // If 0, not force number of active experts

    std::string cache_policy;
private:
    GlobalConfig()
    { 
        setDefault(); 
        if (profiling) {
            print();
        }
    }
};

}