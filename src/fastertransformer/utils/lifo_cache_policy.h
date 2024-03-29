/**
 * \file
 * \brief LIFO cache policy implementation
 */
#ifndef LIFO_CACHE_POLICY_HPP
#define LIFO_CACHE_POLICY_HPP

#include "cache_policy.h"
#include <list>
#include <unordered_map>

namespace caches
{

/**
 * \brief FIFO (First in, first out) cache policy
 * \details FIFO policy in the case of replacement removes the first added element.
 *
 * That is, consider the following key adding sequence:
 * ```
 * A -> B -> C -> ...
 * ```
 * In the case a cache reaches its capacity, the FIFO replacement candidate policy
 * returns firstly added element `A`. To show that:
 * ```
 * # New key: X
 * Initial state: A -> B -> C -> ...
 * Replacement candidate: A
 * Final state: B -> C -> ... -> X -> ...
 * ```
 * An so on, the next candidate will be `B`, then `C`, etc.
 * \tparam Key Type of a key a policy works with
 */
template <typename Key>
class LIFOCachePolicy : public ICachePolicy<Key>
{
  public:
    using lifo_iterator = typename std::list<Key>::const_iterator;

    LIFOCachePolicy() = default;
    ~LIFOCachePolicy() = default;

    void Insert(const Key &key) override
    {
        lifo_queue.emplace_back(key);
        key_lookup[key] = --lifo_queue.end();
    }

    // handle request to the key-element in a cache
    void Touch(const Key &key) noexcept override
    {
        // nothing to do here in the LIFO strategy
        (void)key;
    }
    // handle element deletion from a cache
    void Erase(const Key &key) noexcept override
    {
        auto element = key_lookup[key];
        lifo_queue.erase(element);
        key_lookup.erase(key);
    }

    // return a key of a replacement candidate
    const Key &ReplCandidate() const noexcept override
    {
        const Key &front = lifo_queue.front();
        const Key &back = lifo_queue.back();
        // first evict useless caches
        if ((front.find("encoder") != std::string::npos)
            != (back.find("encoder") != std::string::npos)) {
            return lifo_queue.front();
        }
        if ((front.find("decoder") != std::string::npos)
            != (back.find("decoder") != std::string::npos)) {
            return lifo_queue.front();
        }
        // then lifo
        return lifo_queue.back();
    }

  private:
    std::list<Key> lifo_queue;
    std::unordered_map<Key, lifo_iterator> key_lookup;
};
} // namespace caches

#endif // LIFO_CACHE_POLICY_HPP
