/**
 * \file
 * \brief Generic cache implementation
 */
#ifndef CACHE_H
#define CACHE_H

#include "cache_policy.h"
#include "src/fastertransformer/utils/cuda_utils.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace caches
{

/**
 * \brief Fixed sized cache that can be used with different policy types (e.g. LRU, FIFO, LFU)
 * \tparam Key Type of a key (should be hashable)
 * \tparam Value Type of a value stored in the cache
 * \tparam Policy Type of a policy to be used with the cache
 * \tparam HashMap Type of a hashmap to use for cache operations. Should have `std::unordered_map` compatible interface
 */
template <typename Key, typename Value,
          typename HashMap = std::unordered_map<Key, Value>>
class fixed_sized_cache
{
  public:
    using iterator = typename HashMap::iterator;
    using const_iterator = typename HashMap::const_iterator;
    using operation_guard = typename std::lock_guard<std::mutex>;
    using on_erase_cb = typename std::function<void(const Key &key, const Value &value)>;

    /**
     * \brief Fixed sized cache constructor
     * \throw std::invalid_argument
     * \param[in] max_size Maximum size of the cache
     * \param[in] policy Cache policy to use
     * \param[in] on_erase on_erase_cb function to be called when cache's element get erased
     */
    explicit fixed_sized_cache(
        size_t max_size, std::shared_ptr<ICachePolicy<Key>> policy,
        on_erase_cb on_erase = [](const Key &, const Value &) {})
        : cache_policy{policy}, max_cache_size{max_size}, on_erase_callback{on_erase}
    {
        if (max_cache_size == 0)
        {
            throw std::invalid_argument{"Size of the cache should be non-zero"};
        }
    }

    ~fixed_sized_cache() noexcept
    {
        Clear();
    }

    /**
     * \brief Put element into the cache
     * \param[in] key Key value to use
     * \param[in] value Value to assign to the given key
     */
    void Put(const Key &key, const Value &value) noexcept
    {
        operation_guard lock{safe_op};
        auto elem_it = FindElem(key);

        if (elem_it == cache_items_map.end())
        {
            // add new element to the cache
            if (cache_items_map.size() + 1 > max_cache_size)
            {
                auto disp_candidate_key = cache_policy->ReplCandidate();

                Erase(disp_candidate_key);
            }

            Insert(key, value);
        }
        else
        {
            // update previous value
            Update(key, value);
        }
    }

    /**
     * \brief Put dummy element into the cache when cache is not full
     * \param[in] key Key value to use
     * \param[in] value Value to assign to the given key
     */
    void PutDummy(const Key &key, const Value &value) noexcept
    {
        operation_guard lock{safe_op};

        if (cache_items_map.size() == max_cache_size) {
            return;
        }

        cache_policy->InsertDummy(key);
        cache_items_map.emplace(std::make_pair(key, value));
    }

    /**
     * \brief Put element into the cache, assign and return the replaced value
     * \return If the first value is default_value, it means the cache is not filled. The second value means if the key is cached.
     * \param[in] key Key value to use
     * \param[in] value Value to assign to the given key
     */
    std::pair<Value, bool> GetOrPut(const Key &key, const Value &default_value) noexcept
    {
        operation_guard lock{safe_op};
        Value repl = default_value;
    
        const_iterator elem_it;
        bool cached;
        std::tie(elem_it, cached) = GetInternal(key);

        if (!cached)
        {
            // add new element to the cache
            if (cache_items_map.size() + 1 > max_cache_size)
            {
                auto disp_candidate_key = cache_policy->ReplCandidate();
                auto disp_elem_it = FindElem(disp_candidate_key);
                repl = disp_elem_it->second;

                Erase(disp_elem_it);
            }

            Insert(key, repl);
        }
        else
        {
            repl = elem_it->second;
        }
        return {repl, cached};
    }

    /**
     * \brief Try to get an element by the given key from the cache
     * \param[in] key Get element by key
     * \return Pair of iterator that points to the element and boolean value that shows
     * whether get operation has been successful or not. If pair's boolean value is false,
     * the element is not presented in the cache. If pair's boolean value is true,
     * returned iterator can be used to get access to the element
     */
    std::pair<const_iterator, bool> TryGet(const Key &key) const noexcept
    {
        operation_guard lock{safe_op};
        return GetInternal(key);
    }

    /**
     * \brief Get element from the cache if present
     * \warning This method will change in the future with an optional class capabilities
     * to avoid throwing exceptions
     * \throw std::range_error
     * \param[in] key Get element by key
     * \return Reference to the value stored by the specified key in the cache
     */
    const Value &Get(const Key &key) const
    {
        operation_guard lock{safe_op};
        auto elem = GetInternal(key);

        if (elem.second)
        {
            return elem.first->second;
        }
        else
        {
            throw std::range_error{"No such element in the cache"};
        }
    }

    /**
     * \brief Check whether the given key is presented in the cache
     * \param[in] key Element key to check
     * \retval true Element is presented in the case
     * \retval false Element is not presented in the case
     */
    bool Cached(const Key &key) const noexcept
    {
        operation_guard lock{safe_op};
        return FindElem(key) != cache_items_map.cend();
    }

    /**
     * \brief Get number of elements in cache
     * \return Number of elements currently stored in the cache
     */
    std::size_t Size() const
    {
        operation_guard lock{safe_op};

        return cache_items_map.size();
    }

    /**
     * Remove an element specified by key
     * \param[in] key Key parameter
     * \retval true if an element specified by key was found and deleted
     * \retval false if an element is not present in a cache
     */
    bool Remove(const Key &key)
    {
        operation_guard lock{safe_op};

        auto elem = FindElem(key);

        if (elem == cache_items_map.end())
        {
            return false;
        }

        Erase(elem);

        return true;
    }

  protected:
    void Clear()
    {
        operation_guard lock{safe_op};

        std::for_each(begin(), end(),
                      [&](const std::pair<const Key, Value> &el) { cache_policy->Erase(el.first); });
        cache_items_map.clear();
    }

    const_iterator begin() const noexcept
    {
        return cache_items_map.cbegin();
    }

    const_iterator end() const noexcept
    {
        return cache_items_map.cend();
    }

  protected:
    void Insert(const Key &key, const Value &value)
    {
        cache_policy->Insert(key);
        cache_items_map.emplace(std::make_pair(key, value));
    }

    void Erase(const_iterator elem)
    {
        cache_policy->Erase(elem->first);
        on_erase_callback(elem->first, elem->second);
        cache_items_map.erase(elem);
    }

    void Erase(const Key &key)
    {
        auto elem_it = FindElem(key);

        Erase(elem_it);
    }

    void Update(const Key &key, const Value &value)
    {
        cache_policy->Touch(key);
        cache_items_map[key] = value;
    }

    const_iterator FindElem(const Key &key) const
    {
        return cache_items_map.find(key);
    }

    std::pair<const_iterator, bool> GetInternal(const Key &key) const noexcept
    {
        auto elem_it = FindElem(key);

        if (elem_it != end())
        {
            cache_policy->Touch(key);
            return {elem_it, true};
        }

        return {elem_it, false};
    }

  private:
    HashMap cache_items_map;
    mutable std::shared_ptr<ICachePolicy<Key>> cache_policy;
    mutable std::mutex safe_op;
    std::size_t max_cache_size;
    on_erase_cb on_erase_callback;
};
} // namespace caches

#endif // CACHE_HPP
