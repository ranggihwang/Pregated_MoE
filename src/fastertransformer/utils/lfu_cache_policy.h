/**
 * \file
 * \brief LFU cache policy implementation
 */
#ifndef LFU_CACHE_POLICY_H
#define LFU_CACHE_POLICY_H

#include "cache_policy.h"
#include <cstddef>
#include <iostream>
#include <map>
#include <unordered_map>

namespace caches
{
/**
 * \brief LFU (Least frequently used) cache policy
 * \details LFU policy in the case of replacement removes the least frequently used
 * element.
 *
 * Each access to an element in the cache increments internal counter (frequency) that
 * represents how many times that particular key has been accessed by someone. When a
 * replacement has to occur the LFU policy just takes a look onto keys' frequencies
 * and remove the least used one. E.g. cache of two elements where `A` has been accessed
 * 10 times and `B` – only 2. When you want to add a key `C` the LFU policy returns `B`
 * as a replacement candidate.
 * \tparam Key Type of a key a policy works with
 */
template <typename Key>
class LFUCachePolicy : public ICachePolicy<Key>
{
  public:
    using lfu_iterator = typename std::multimap<std::size_t, Key>::iterator;

    LFUCachePolicy() = default;
    ~LFUCachePolicy() override = default;

    void Insert(const Key &key) override
    {
        if (key.find("decoder") != std::string::npos 
            && cache_name.find("encoder") != std::string::npos) {
            for (const auto &n : lfu_storage) {
                Reset(n.first);
            }
        }
        constexpr std::size_t INIT_VAL = 1;
        // all new value initialized with the frequency 1
        lfu_storage[key] =
            frequency_storage.emplace_hint(frequency_storage.cend(), INIT_VAL, key);
        cache_name = key;
    }

    void InsertDummy(const Key &key) override
    {
        constexpr std::size_t INIT_VAL = 0;
        // all new value initialized with the frequency 1
        lfu_storage[key] =
            frequency_storage.emplace_hint(frequency_storage.cbegin(), INIT_VAL, key);
        cache_name = key;
    }

    void Touch(const Key &key) override
    {
        // get the previous frequency value of a key
        auto elem_for_update = lfu_storage[key];
        auto updated_elem = std::make_pair(elem_for_update->first + 1, elem_for_update->second);
        // update the previous value
        frequency_storage.erase(elem_for_update);
        lfu_storage[key] =
            frequency_storage.emplace_hint(frequency_storage.cend(), std::move(updated_elem));
    }

    void Erase(const Key &key) noexcept override
    {
        frequency_storage.erase(lfu_storage[key]);
        lfu_storage.erase(key);
    }

    const Key &ReplCandidate() const noexcept override
    {
        // at the beginning of the frequency_storage we have the
        // least frequency used value
        return frequency_storage.cbegin()->second;
    }

  private:
    std::multimap<std::size_t, Key> frequency_storage;
    std::unordered_map<Key, lfu_iterator> lfu_storage;

    // workaround: reset cache while switching from encoder to decoder
    std::string cache_name;

    void Reset(const Key &key)
    {
        // get the previous frequency value of a key
        auto elem_for_update = lfu_storage[key];
        auto updated_elem = std::make_pair(1, elem_for_update->second);
        // update the previous value
        frequency_storage.erase(elem_for_update);
        lfu_storage[key] =
            frequency_storage.emplace_hint(frequency_storage.cbegin(), std::move(updated_elem));
    }
};
} // namespace caches

#endif // LFU_CACHE_POLICY_H
