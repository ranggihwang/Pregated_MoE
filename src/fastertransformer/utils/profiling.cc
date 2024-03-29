#include "profiling.h"
#include "meter.h"
#include "src/fastertransformer/utils/logger.h"


namespace {

void clearEvents(std::vector<cudaEvent_t>& events)
{
    for (cudaEvent_t event : events)
    {
        cudaEventDestroy(event);
    }
    events.clear();
}

} // namespace

namespace fastertransformer {

const std::vector<std::string> Profiling::event_names_{
    "NMB_S",
    "NMB_E",
    "BLK_S",
    "BLK_E",
    "MEM_S",
    "MEM_E",
    "CMP_S",
    "CMP_E"
};

void Profiling::insert(cudaStream_t stream, EventType type)
{
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);
    events_[(int)type].push_back(event);
}

Profiling::~Profiling()
{
    reset();
}

void Profiling::reset()
{
    for (auto& events: events_) {
        clearEvents(events);
    }
    cache_hit_rate_.reset();
    memory_usages_.clear();
    max_num_active_experts_ = 0;
    average_num_active_experts_.reset();
}

void Profiling::report(bool detailed_timing) const
{
    float ms;
    AverageMeter<float> comp_lats, mem_lats, block_lats;

    std::cout << "Total events num:";
    for (int i = 0; i < NUM_EVENT_TYPE; i += 2) {
        FT_CHECK(events_[i].size() == events_[i + 1].size());
        std::cout << " (" << event_names_[i].substr(0, 3) << ")" << events_[i].size();
    }
    std::cout << std::endl;

    for (int i = 0; i < NUM_EVENT_TYPE; i += 2) {
        AverageMeter<float> avg_lats;
        for (int j = 0; j < events_[i].size(); j++) {
            cudaEventElapsedTime(&ms, events_[i][j], events_[i + 1][j]);
            avg_lats.update(ms);
        }
        std::cout << event_names_[i].substr(0, 3) << " AVG: " << avg_lats.getAvg() << " ms" << std::endl;
    }

    std::cout << "Average cache hit rate: " << cache_hit_rate_.getAvg() << std::endl;

    if (detailed_timing) {
        std::cout << "Timeline:" << std::endl;

        for (int i = 0; i < events_[0].size(); i++) {
            for (int j = 0; j < NUM_EVENT_TYPE; j++) {
                cudaEventElapsedTime(&ms, events_[0][0], events_[j][i]);
                std::cout << event_names_[j] << "#" << i << " " << ms << " ms" << std::endl;
            }
        }
    }

    std::cout << "MEM usage: ";
    for (size_t used : memory_usages_) {
        std::cout << used << " ";
    }
    std::cout << std::endl;

    std::cout << "Max active experts: " << max_num_active_experts_ << std::endl;

    std::cout << "Average active experts: " << average_num_active_experts_.getAvg() << std::endl;
}

void Profiling::recordMemoryUsage()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    memory_usages_.push_back(total - free);
}

} // namespace fastertransformer
