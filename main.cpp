/*

    Copyright (C) 2026, Sergio Randriamihoatra

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE


#ifdef _WIN32
#include <windows.h>
#endif // _WIN32

#include <immintrin.h>
#include <thread>
#include <numeric>
#include <atomic>
#include <functional>
#include <shared_mutex>
#include <string>
#include <vector>
#include <iostream>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <ranges>
#include <span>
#include <array>
#include <cmath>
#include <future>
#include <condition_variable>
#include <queue>
#include <cstring>
#include <cstdlib>
#include <mutex> 
#include <cerrno> 

#if defined(_POSIX_VERSION) && _POSIX_VERSION >= 200809L
    
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
    
#endif // _POSIX_VERSION

// For __wc_internal_class implementation
#define __wc_lib_use_std_atomic

#if __cplusplus >= 202002L
#define __cpp_lib_use_likely
    // Can use C++20 [[likely]] for branch
    // prediction.
    
#endif // __cplusplus

#define WC_EXIT_CODE_FAILURE      -1 
#define WC_EXIT_CODE_SUCCESS       0 
#define WC_EXIT_CODE_ERROR         1 
#define WC_EXIT_CODE_UNKNOWN       2 

#define LIKELY(x) (__builtin_expect(!!(x), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))

#define __FORCE_INLINE __attribute__((always_inline)) inline

#if not defined(__wc_lib_thread_pool)
#define __wc_lib_thread_pool

#ifdef _WIN32

#include <tchar.h>
#include <cstdio>
#include <strsafe.h>

// For logging
void display_error(LPCTSTR lpszFunction)
// Routine Description:
// Retrieve and output the system error message for the last-error code
{
    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError();
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0,
        NULL);
    lpDisplayBuf =
        (LPVOID) LocalAlloc(LMEM_ZEROINIT,
                            (lstrlen((LPCTSTR) lpMsgBuf)
                             + lstrlen((LPCTSTR) lpszFunction)
                             + 40)  // account for format string
                            * sizeof(TCHAR));
    if (FAILED(StringCchPrintf((LPTSTR) lpDisplayBuf,
                               LocalSize(lpDisplayBuf) / sizeof(TCHAR),
                               TEXT("%s failed with error code %d as follows:\n%s"),
                               lpszFunction,
                               dw,
                               lpMsgBuf))) {
        printf("FATAL ERROR: Unable to output error code.\n");
    }
    _tprintf(TEXT("ERROR: %s\n"), (LPCTSTR) lpDisplayBuf);
    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
}

#endif // _WIN32 


namespace tp {
// Default CPU core number is N_CORE
inline constexpr int N_CORE = 4;

inline constexpr int BUFFER_SIZE = 128; 

// Cache line size for alignment
inline constexpr size_t CACHE_LINE_SIZE = 64;

// Small buffer optimization for task storage
template<size_t BufferSize = BUFFER_SIZE>
class task_wrapper {
  private:
    alignas(std::max_align_t) char buffer[BufferSize];
    
    void (*invoke_fn)(void*) = nullptr;
    void (*destroy_fn)(void*) = nullptr;
    void (*move_fn)(void*, void*) = nullptr;
    
  public:
    task_wrapper() = default;
    
    template<typename F> task_wrapper(F&& f) {
        using DecayF = std::decay_t<F>;
        
        static_assert(sizeof(DecayF) <= BufferSize, "Task too large for buffer");
        static_assert(alignof(DecayF) <= alignof(std::max_align_t), "Task alignment too large");
        
        new (buffer) DecayF(std::forward<F> (f));
        invoke_fn = [](void* p) {
            (*static_cast<DecayF*> (p))();
        };

        destroy_fn = [](void* p) {
            static_cast<DecayF *> (p)->~DecayF();
        };

        move_fn = [](void* src, void* dst) {
            new (dst) DecayF(std::move(*static_cast<DecayF*> (src)));
            static_cast<DecayF *> (src)->~DecayF();
        };
    }
    
    task_wrapper(task_wrapper&& other) noexcept {
        if (other.invoke_fn) 
        {
            other.move_fn(other.buffer, buffer);
            invoke_fn = other.invoke_fn;
            destroy_fn = other.destroy_fn;
            move_fn = other.move_fn;
            other.invoke_fn = nullptr;
            other.destroy_fn = nullptr;
            other.move_fn = nullptr;
        }
    }
    
    task_wrapper &operator=(task_wrapper&& other) noexcept {
        if (this != &other) 
        {
            if (destroy_fn) 
            {
                destroy_fn(buffer);
            }

            if (other.invoke_fn) 
            {
                other.move_fn(other.buffer, buffer);
                invoke_fn = other.invoke_fn;
                destroy_fn = other.destroy_fn;
                move_fn = other.move_fn;
                other.invoke_fn = nullptr;
                other.destroy_fn = nullptr;
                other.move_fn = nullptr;
            } else {
                invoke_fn = nullptr;
                destroy_fn = nullptr;
                move_fn = nullptr;
            }
        }
        
        return *this;
    }
    
    void operator()() {
        if (invoke_fn) {
            invoke_fn(buffer);
        }
    }
    
    explicit operator bool() const {
        return invoke_fn != nullptr;
    }
    
    ~task_wrapper() {
        if (destroy_fn) {
            destroy_fn(buffer);
        }
    }
    
    task_wrapper(const task_wrapper &) = delete;
    task_wrapper &operator= (const task_wrapper &) = delete;
};

// Per-thread queue with cache line alignment
struct alignas(CACHE_LINE_SIZE) aligned_task_queue {
    std::queue<task_wrapper<>> tasks;
    std::mutex mutex;
    
    aligned_task_queue(aligned_task_queue&& other) noexcept
        : tasks(std::move(other.tasks)) {
        // Note: mutex cannot be moved, so we just default construct it
    }
    
    aligned_task_queue &operator= (aligned_task_queue&& other) noexcept {
        if (this != &other) {
            tasks = std::move(other.tasks);
        }
        return *this;
    }
    
    aligned_task_queue() = default;
    aligned_task_queue(const aligned_task_queue &) = delete;
    aligned_task_queue &operator= (const aligned_task_queue &) = delete;
};

// Optimized thread pool with work stealing
template <size_t __n_core = N_CORE>
class __wc_thread_pool {
  public:
    __wc_thread_pool(const __wc_thread_pool<__n_core> &) = delete;
    __wc_thread_pool &operator= (const __wc_thread_pool<__n_core> &) = delete;
    
    static __wc_thread_pool<__n_core> *Instance() {
        std::call_once(__init_flag, []() {
            __instance.reset(new __wc_thread_pool<__n_core>());
        });

        return __instance.get();
    }
    
    // Submit a task and get a future for the result
    template<typename F, typename... Args>
    auto submit(F&& fn, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        // Use lambda instead of std::bind for better performance
        auto task = std::make_shared<std::packaged_task<return_type() >> (
            [fn = std::forward<F> (fn), ...args = std::forward<Args> (args)]() mutable -> return_type {
                return fn(std::move(args)...);
            }
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(__submit_mutex);
            if (__stop.load(std::memory_order_acquire)) {
                throw std::runtime_error("Cannot submit to stopped thread pool");
            }

            // Round-robin task distribution to reduce contention
            size_t target_queue = __next_queue.fetch_add(1, std::memory_order_relaxed) % __n_core;
            {
                std::unique_lock<std::mutex> queue_lock(__queues[target_queue].mutex);
                __queues[target_queue].tasks.emplace([task]() {
                    try {
                        (*task)();
                    }
                    catch (...) {
                        // Prevent thread death from exceptions
                    }
                });
            } __active_tasks.fetch_add(1, std::memory_order_release);
        } __tp_cv.notify_one();
        return res;
    }
    
    // Enqueue a void task
    void enqueue(std::function<void() > task) {
        {
            std::unique_lock<std::mutex> lock(__submit_mutex);
            if (__stop.load(std::memory_order_acquire)) {
                throw std::runtime_error("Cannot enqueue to stopped thread pool");
            }
            size_t target_queue = __next_queue.fetch_add(1, std::memory_order_relaxed) % __n_core;
            {
                std::unique_lock<std::mutex> queue_lock(__queues[target_queue].mutex);
                __queues[target_queue].tasks.emplace([task = std::move(task)]() {
                    try {
                        task();
                    } catch (...) {
                        // Prevent thread death from exceptions
                    }
                });
            }
            __active_tasks.fetch_add(1, std::memory_order_release);
        }
        __tp_cv.notify_one();
    }
    
    // Batch enqueue for better performance
    template<typename Iterator>
    void enqueue_batch(Iterator begin, Iterator end) {
        size_t count = 0;
        {
            std::unique_lock<std::mutex> lock(__submit_mutex);
            
            if (__stop.load(std::memory_order_acquire)) {
                throw std::runtime_error("Cannot enqueue to stopped thread pool");
            }

            for (auto it = begin; it != end; ++it) {
                size_t target_queue = __next_queue.fetch_add(1, std::memory_order_relaxed) % __n_core;
                {
                    std::unique_lock<std::mutex> queue_lock(__queues[target_queue].mutex);
                    __queues[target_queue].tasks.emplace([task = *it]() {
                        try {
                            task();
                        } catch (...) {
                            // Prevent thread death from exceptions
                        }
                    });
                }
                ++count;
            }
            __active_tasks.fetch_add(count, std::memory_order_release);
        }
        if (count == 1) {
            __tp_cv.notify_one();
        } else {
            __tp_cv.notify_all();
        }
    }
    
    // Graceful shutdown with timeout
    void shutdown(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        {
            std::unique_lock<std::mutex> lock(__submit_mutex);
            __stop.store(true, std::memory_order_release);
        }
        __tp_cv.notify_all();

        // If a thread do not respond anything after deadline seconds, 
        // detach it. 
        auto deadline = std::chrono::steady_clock::now() + timeout;
        
        for (auto& thread : __threads) {
            if (thread.joinable()) {
                auto remaining = deadline - std::chrono::steady_clock::now();
                
                if (remaining > std::chrono::milliseconds(0)) {
                    thread.join();
                } else {
                    // Timeout reached, detach remaining threads
                    thread.detach();
                }
            }
        }
    }
    
    ~__wc_thread_pool() {
        shutdown();
    }
    
    size_t thread_count() const {
        return __n_core;
    }
    
    size_t active_tasks() const {
        return __active_tasks.load(std::memory_order_acquire);
    }
    
    // Wait for all tasks to complete
    void wait_all() {
        while (__active_tasks.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }
    
  private:
    __wc_thread_pool()
        : __queues(__n_core) {
        __cpu_core = std::thread::hardware_concurrency();
        __threads.reserve(__n_core);
        for (size_t i = 0; i < __n_core; ++i) {
            __threads.emplace_back([this, i]() {
                worker_thread(i);
            });
#ifdef __linux__
            // Pin threads to CPU cores for better cache locality
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i % __cpu_core, &cpuset);
            pthread_setaffinity_np(__threads.back().native_handle(), sizeof(cpuset), &cpuset);
#endif
        }
    }
    
    // Worker thread with work stealing
    void worker_thread(size_t thread_id) {
        while (true) {
            task_wrapper<> task;
            bool found_task = false;
            // Try to get task from own queue first
            {
                std::unique_lock<std::mutex> lock(__queues[thread_id].mutex);
                if (!__queues[thread_id].tasks.empty()) {
                    task = std::move(__queues[thread_id].tasks.front());
                    __queues[thread_id].tasks.pop();
                    found_task = true;
                }
            }

            // If own queue is empty, try work stealing
            if (!found_task) {
                for (size_t i = 1; i <= __n_core; ++i) {
                    size_t steal_from = (thread_id + i) % __n_core;
                    std::unique_lock<std::mutex> lock(__queues[steal_from].mutex, std::try_to_lock);
                    
                    if (lock.owns_lock() && !__queues[steal_from].tasks.empty()) {
                        task = std::move(__queues[steal_from].tasks.front());
                        __queues[steal_from].tasks.pop();
                        found_task = true;
                        break;
                    }
                }
            }

            if (found_task) {
                task();
                __active_tasks.fetch_sub(1, std::memory_order_release);
            } else {
                // No task found, wait for notification
                std::unique_lock<std::mutex> lock(__wait_mutex);
                __tp_cv.wait(lock, [this]() {
                    return __stop.load(std::memory_order_acquire) ||
                           __active_tasks.load(std::memory_order_acquire) > 0;
                });
                
                // Check if we should stop
                if (__stop.load(std::memory_order_acquire)) {
                    // Drain remaining tasks before exiting
                    bool has_tasks = true;
                    
                    while (has_tasks) {
                        has_tasks = false;
                        std::unique_lock<std::mutex> lock(__queues[thread_id].mutex);
                        
                        if (!__queues[thread_id].tasks.empty()) {
                            task = std::move(__queues[thread_id].tasks.front());
                            __queues[thread_id].tasks.pop();
                            has_tasks = true;
                            lock.unlock();
                            task();
                            __active_tasks.fetch_sub(1, std::memory_order_release);
                        }
                    }
                    return;
                }
            }
        }
    }
    
    static std::unique_ptr<__wc_thread_pool<__n_core>> __instance;
    static std::once_flag __init_flag;
    
    unsigned int __cpu_core;
    std::vector<std::thread> __threads;
    
    // Per-thread queues for reduced contention
    std::vector<aligned_task_queue> __queues;
    
    // Separate mutexes to reduce lock contention
    std::mutex __submit_mutex;
    std::mutex __wait_mutex;
    
    std::condition_variable __tp_cv;
    std::atomic<bool> __stop{false};
    std::atomic<size_t> __active_tasks{0};
    std::atomic<size_t> __next_queue{0};
};

// Static member initialization
template <size_t __n_core>
std::unique_ptr<__wc_thread_pool<__n_core>> __wc_thread_pool<__n_core>::__instance;

template <size_t __n_core>
std::once_flag __wc_thread_pool<__n_core>::__init_flag;

// Convenience alias
using thread_pool = __wc_thread_pool<>;

} // namespace tp

#endif // __wc_lib_thread_pool 

#if not defined(__wc_argv_parse)
#define __wc_argv_parse

namespace detail {
inline constexpr int MAX_OPTIONS = 32;

// SIMD-optimized FNV-1a hash for longer strings
constexpr uint32_t hash_fnv1a(std::string_view __s) noexcept {
    uint32_t hash = 2166136261u;
    const char *ptr = __s.data();
    size_t len = __s.size();

    // Process 4 bytes at a time for better performance
    while (len >= 4) {
        hash ^= static_cast<uint8_t>(ptr[0]);
        hash *= 16777619u;
        hash ^= static_cast<uint8_t>(ptr[1]);
        hash *= 16777619u;
        hash ^= static_cast<uint8_t>(ptr[2]);
        hash *= 16777619u;
        hash ^= static_cast<uint8_t>(ptr[3]);
        hash *= 16777619u;
        ptr += 4;
        len -= 4;
    }
    
    // Handle remainder
    while (len--) {
        hash ^= static_cast<uint8_t>(*ptr++);
        hash *= 16777619u;
    }

    return hash;
}

// Branchless, unrolled fast atoi
constexpr int __fast_atoi(const char* __s) noexcept {
    int res = 0;
    int sign = 1;
    
    // Branchless sign handling
    sign = 1 - 2 * (*__s == '-');
    __s += (*__s == '-' || *__s == '+');
    
    // Unroll loop for common case (up to 8 digits)
    // Process 4 digits at once using SWAR (SIMD Within A Register)
    while (*__s >= '0' && *__s <= '9') {
        // Manual unrolling for 4 iterations
        res = res * 10 + (*__s++ - '0');
        if (!(*__s >= '0' && *__s <= '9')) {
            break;
        }

        res = res * 10 + (*__s++ - '0');
        if (!(*__s >= '0' && *__s <= '9')) {
            break;
        }

        res = res * 10 + (*__s++ - '0');
        if (!(*__s >= '0' && *__s <= '9')) {
            break;
        }

        res = res * 10 + (*__s++ - '0');
    }

    return res * sign;
}

// Bit-manipulation based integer width (no floating point!)
__FORCE_INLINE size_t __int_width(size_t __n) noexcept {
    if (__n == 0) {
        return 1;
    }

    // Use binary search on powers of 10
    // Faster than log10 for small numbers
    if (__n < 10) {
        return 1;
    }

    if (__n < 100) {
        return 2;
    }

    if (__n < 1000) {
        return 3;
    }

    if (__n < 10000) {
        return 4;
    }

    if (__n < 100000) {
        return 5;
    }

    if (__n < 1000000) {
        return 6;
    }

    if (__n < 10000000) {
        return 7;
    }

    if (__n < 100000000) {
        return 8;
    }

    if (__n < 1000000000) {
        return 9;
    }

    if (__n < 10000000000ULL) {
        return 10;
    }

    if (__n < 100000000000ULL) {
        return 11;
    }

    if (__n < 1000000000000ULL) {
        return 12;
    }

    if (__n < 10000000000000ULL) {
        return 13;
    }

    if (__n < 100000000000000ULL) {
        return 14;
    }

    if (__n < 1000000000000000ULL) {
        return 15;
    }

    if (__n < 10000000000000000ULL) {
        return 16;
    }

    if (__n < 100000000000000000ULL) {
        return 17;
    }

    if (__n < 1000000000000000000ULL) {
        return 18;
    }

    if (__n < 10000000000000000000ULL) {
        return 19;
    }

    return 20;
}

enum class OptionType : uint8_t { flag, val, is_multi_time_appear_true_v };

template <class __DataType>
struct __wc_option {
    char short_name;
    std::string_view long_name;
    OptionType __type;
    __DataType __default_value;
    
    constexpr __wc_option(char __s, std::string_view __l, OptionType __t, __DataType def = __DataType{})
        : short_name(__s), long_name(__l), __type(__t), __default_value(def) 
        {
            // Every option should be unique
            if (OptionType::is_multi_time_appear_true_v == __t) throw std::logic_error("An option was construct multiple times!"); 
        }
        
    constexpr ~__wc_option() = default;

    // For memory optimization, and to prevent any 
    // logic error also. 
    __wc_option(__wc_option&) = delete; 
    __wc_option(__wc_option&&) = delete;
    
    __wc_option(const __wc_option&) = delete; 
    __wc_option(const __wc_option&&) = delete; 
};

#if not defined __wc_argparser_class
#define __wc_argparser_class

// Compact option storage with better cache locality
template <size_t __max_opt = MAX_OPTIONS>
class __wc_argparser {
  public:
    template<typename _Tp>
    constexpr void add_opt(const __wc_option<_Tp> &__opt) {
        if (UNLIKELY(opt_count >= __max_opt))
            [[unlikely]]
            return;

        options[opt_count++] = 
        {
            __opt.short_name,
            static_cast<uint8_t>(__opt.long_name.size()),
            __opt.__type,
            0, // padding
            detail::hash_fnv1a(__opt.long_name),
            __opt.long_name.data()
        };
    }
    
    __wc_argparser() = default;
    ~__wc_argparser() = default;
    
    // Optimized parse with early exit and minimal branching
    void parse(int argc, const char** argv) noexcept {
        pcount = 0;
        
        for (int i = 1; i < argc; ++i) {
            const char *__arg = argv[i];
            
            // Early exit for non-options
            if (__arg[0] != '-') {
                continue;
            }

            // Branchless option type detection
            bool is_long = (__arg[1] == '-');
            const char *opt_start = __arg + 1 + is_long;
            
            if (*opt_start == '\0') {
                continue;
            }

            if (is_long) {
                __parse_long_options(opt_start, i, argc, argv);
            }

            else {
                __parse_short_options(*opt_start, i, argc, argv);
            }
        }
    }
    
    // Optimized lookup using SIMD-style comparison where possible
    [[nodiscard]]
    std::optional<const char *> get(char short_name) const noexcept {
        // Linear search is actually faster than hash for small arrays
        // due to better cache locality and branch prediction
        for (size_t i = 0; i < pcount; ++i) {
            if (options[parsed[i].__idx].short_name == short_name) {
                return parsed[i].__v;
            }
        }
        return std::nullopt;
    }
    
    [[nodiscard]]
    std::optional<const char *> get(std::string_view long_name) const noexcept {
        const uint32_t hash = detail::hash_fnv1a(long_name);
        
        // Use hash for long option lookup (worth it for string comparison)
        for (size_t i = 0; i < pcount; ++i) {
            const auto& opt = options[parsed[i].__idx];
            if (opt.hash == hash) { // Fast hash comparison first
                return parsed[i].__v;
            }
        }

        return std::nullopt;
    }
    
    [[nodiscard]]
    __FORCE_INLINE bool has(char __sn) const noexcept {
        return get(__sn).has_value();
    }
    
    [[nodiscard]]
    __FORCE_INLINE bool has(std::string_view __ln) const noexcept {
        return get(__ln).has_value();
    }
    
    [[nodiscard]]
    __FORCE_INLINE int get_int(char __sn, int __default_value = 0) const noexcept {
        auto val = get(__sn);
        return val ? detail::__fast_atoi(*val) : __default_value;
    }
    
  private:
    // Packed structure for better cache efficiency
    struct __parsed_option {
        uint8_t __idx;
        const char *__v;
    };
    
    // Cache-aligned option entry (pack strings together)
    struct alignas(32) option_entry {
        char short_name;
        uint8_t long_name_len;
        OptionType type;
        uint8_t _padding;
        uint32_t hash;
        const char *long_name;
    };
    
    // Use stack arrays for small, fixed-size data
    std::array<__parsed_option, __max_opt> parsed;
    uint8_t pcount = 0;
    
    std::array<option_entry, __max_opt> options;
    uint8_t opt_count = 0;
    
    void __parse_short_options(char opt, int &idx, int argc, const char** argv) noexcept {
        // Unrolled search for common cases (2-4 options)
        if (opt_count > 0 && options[0].short_name == opt) {
            __add_parsed_option(0, idx, argc, argv);
            return;
        }
        if (opt_count > 1 && options[1].short_name == opt) {
            __add_parsed_option(1, idx, argc, argv);
            return;
        }
        if (opt_count > 2 && options[2].short_name == opt) {
            __add_parsed_option(2, idx, argc, argv);
            return;
        }
        if (opt_count > 3 && options[3].short_name == opt) {
            __add_parsed_option(3, idx, argc, argv);
            return;
        }
        // Fallback for remaining options
        for (size_t i = 4; i < opt_count; ++i) {
            if (options[i].short_name == opt) {
                __add_parsed_option(i, idx, argc, argv);
                return;
            }
        }
    }
    
    void __parse_long_options(const char* opt, int &idx, int argc, const char** argv) noexcept {
        const uint32_t hash = detail::hash_fnv1a(opt);

        for (size_t i = 0; i < opt_count; ++i) {
            if (options[i].hash == hash) {
                __add_parsed_option(i, idx, argc, argv);
                return;
            }
        }
    }
    
    __FORCE_INLINE void __add_parsed_option(size_t i, int &idx, int argc, const char** argv) noexcept {
        if (pcount >= __max_opt) {
            return;
        }

        const char *value = nullptr;
        if (options[i].type != OptionType::flag && idx + 1 < argc) {
            value = argv[++idx];
        }

        parsed[pcount++] = { static_cast<uint8_t>(i), value };
    }
};

// Builder with move semantics optimization
template <size_t __max_opt = MAX_OPTIONS>
class __wc_argparser_builder {
  public:
    constexpr __wc_argparser_builder() = default;
    
    template <typename _Tn>
    constexpr __wc_argparser_builder &flag(char short_name, std::string_view long_name) {
        parser.add_opt(__wc_option<bool>(short_name, long_name, OptionType::flag));
        return *this;
    }
    
    template <typename _Tn>
    constexpr __wc_argparser_builder &option(char short_name, std::string_view long_name) {
        parser.add_opt(__wc_option<_Tn>(short_name, long_name, OptionType::val));
        return *this;
    }
    
    constexpr __wc_argparser<__max_opt> build() {
        return std::move(parser);
    }
    
  private:
    __wc_argparser<__max_opt> parser;
};

#endif // __wc_argparser_class 
}

#endif // __wc_argv_parse 

#if not defined(__wc_mapped_fs)
#define __wc_mapped_fs

// For file management (eg. reading)
// Map the file content to memory address
// with munmap() and mmap().
namespace fs {

using namespace detail;

// __wc_mapped_file mode.
// The constructor will depend on the
// specified mode.
enum class __wc_mapped_file_mode { BytesOnly, NeedMmap };

struct __wc_mapped_file {
    void *data_ {};
    size_t size_ {};
    
#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
    HANDLE hMapFile_ = INVALID_HANDLE_VALUE;
#else
    int __fd = -1;
#endif
    
    std::string filename_;
    
    // After redesigning,
    // Should include the result in the file structure
    // (For better formatting)
    size_t __w_cnt {};
    size_t __l_cnt {};
    size_t __b_cnt {};
    size_t __c_cnt {};
    
    __wc_mapped_file_mode mode_;
    bool is_stdin_ = false;
    
    // Constructor for stdin
    explicit __wc_mapped_file()
        : filename_(""), is_stdin_(true), mode_(__wc_mapped_file_mode::NeedMmap) {
        // Read all stdin into memory
        std::vector<char> buffer;
        constexpr size_t chunk_size = 65536; // 64KB chunks
        char temp_buf[chunk_size];

        while (std::cin.read(temp_buf, chunk_size) || std::cin.gcount() > 0) {
            buffer.insert(buffer.end(), temp_buf, temp_buf + std::cin.gcount());
        }

        size_ = buffer.size();
        if (size_ > 0) {
            // Allocate memory and copy data
            data_ = malloc(size_);
            if (data_) {
                std::memcpy(data_, buffer.data(), size_);
            } else {
                size_ = 0;
            }
        }
    }
    
    explicit __wc_mapped_file(std::string_view __fn, __wc_mapped_file_mode __mode)
        :  filename_(__fn), is_stdin_(false) {
        if (LIKELY(__mode == __wc_mapped_file_mode::NeedMmap))
            [[likely]] {
            mode_ = __mode;  // SET THE MODE!
#ifdef _WIN32
            hFile_ = CreateFileA(filename_.c_str(),
                                 GENERIC_READ,
                                 FILE_SHARE_READ,
                                 nullptr,
                                 OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL,
                                 nullptr);
                                 
            if (hFile_ == INVALID_HANDLE_VALUE) {
                display_error(TEXT("CreateFileA"));
                return;
            }
            
            LARGE_INTEGER fileSize;
            
            if (!GetFileSizeEx(hFile_, &fileSize)) {
                display_error(TEXT("GetFileSizeEx"));
                CloseHandle(hFile_);
                hFile_ = INVALID_HANDLE_VALUE;
                return;
            }
            
            size_ = static_cast<size_t> (fileSize.QuadPart);
            
            if (size_ == 0) {
                CloseHandle(hFile_);
                hFile_ = INVALID_HANDLE_VALUE;
                return;
            }
            
            hMapFile_ = CreateFileMappingA(hFile_,
                                           nullptr,
                                           PAGE_READONLY,
                                           0,
                                           0,
                                           nullptr);
                                           
            if (hMapFile_ == nullptr) {
                display_error(TEXT("CreateFileMappingA"));
                CloseHandle(hFile_);
                hFile_ = INVALID_HANDLE_VALUE;
                return;
            }
            
            data_ = MapViewOfFile(hMapFile_,
                                  FILE_MAP_READ,
                                  0,
                                  0,
                                  0);
                                  
            if (data_ == nullptr) {
                display_error(TEXT("MapViewOfFile"));
                CloseHandle(hMapFile_);
                CloseHandle(hFile_);
                hMapFile_ = INVALID_HANDLE_VALUE;
                hFile_ = INVALID_HANDLE_VALUE;
                return;
            }
            
#else
            // stat and fstat are linux specific system calls
            struct stat __sb;
            
            if (UNLIKELY((__fd = open(filename_.c_str(), O_RDONLY)) == -1))
                [[unlikely]] {
                
                std::cerr << "Cannot open " << filename_ << ":"; 
                std::perror("open()"); 
                std::exit(WC_EXIT_CODE_FAILURE); 
            }
            
            if (UNLIKELY(fstat(__fd, &__sb) == -1))
                [[unlikely]] {
                close(__fd);
                __fd = -1;
                std::exit(WC_EXIT_CODE_FAILURE); 
            }
            
            if (UNLIKELY((size_ = __sb.st_size) == 0))
                [[unlikely]] {
                close(__fd);
                __fd = -1;
                std::exit(WC_EXIT_CODE_FAILURE); 
            }
            
            
            if (UNLIKELY((data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, __fd, 0)) == MAP_FAILED))
                [[unlikely]] {
                std::cerr << "ERROR: mmap failed! errno=" << errno << " (" << strerror(errno) << ")" << std::endl;
                data_ = nullptr;
                close(__fd);
                __fd = -1;
                return;
            }
            
            madvise(data_, size_, MADV_SEQUENTIAL | MADV_WILLNEED);
#ifdef MADV_HUGEPAGE
            // Give directions to the kernel about the address
            // range beginning at data (void*)
            madvise(data_, size_, MADV_HUGEPAGE);
#endif // MADV_HUGEPAGE 
#endif // _WIN32
        }
        // Do some optimization tricks with -c option
        // Use statx instead of fstat or stat.
        else {
#ifdef _WIN32
            hFile_ = CreateFileA(filename_.c_str(),
                                 GENERIC_READ,
                                 FILE_SHARE_READ,
                                 nullptr,
                                 OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL,
                                 nullptr);
            if (hFile_ == INVALID_HANDLE_VALUE) {
                display_error(TEXT("CreateFileA"));
                return;
            }

            LARGE_INTEGER fileSize;
            if (!GetFileSizeEx(hFile_, &fileSize)) {
                display_error(TEXT("GetFileSizeEx"));
                CloseHandle(hFile_);
                hFile_ = INVALID_HANDLE_VALUE;
                return;
            }

            size_ = static_cast<size_t> (fileSize.QuadPart);
            if (size_ == 0) {
                CloseHandle(hFile_);
                hFile_ = INVALID_HANDLE_VALUE;
                return;
            }
#else
            struct statx __sb;
            if (UNLIKELY(statx(AT_FDCWD, filename_.c_str(), 0, STATX_SIZE, &__sb) == -1))
                [[unlikely]] {
                std::cerr << "Statx error" << std::endl;
                return;
            }

            if (UNLIKELY((size_ = __sb.stx_size) == 0))
                [[unlikely]] {
                std::cerr << "Size error" << std::endl;
                return;
            }

#endif // _WIN32
            mode_ = __mode;
        }
    }
    
    
    // No copy constructor (Object should be unique)
    __wc_mapped_file(const __wc_mapped_file &) = delete;
    __wc_mapped_file &operator= (const __wc_mapped_file &) = delete;
    
    // Move constructor
    __wc_mapped_file(__wc_mapped_file&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
#if defined(__linux__)
        , __fd(other.__fd)
#elif defined(_WIN32)
        , hFile_(other.hFile_)
        , hMapFile_(other.hMapFile_)
#endif
        , filename_(std::move(other.filename_))
        , __w_cnt(other.__w_cnt)
        , __l_cnt(other.__l_cnt)
        , __b_cnt(other.__b_cnt)
        , __c_cnt(other.__c_cnt)
        , mode_(other.mode_)
        , is_stdin_(other.is_stdin_) {
        other.data_ = nullptr;
        other.size_ = 0;
#if defined(__linux__)
        other.__fd = -1;
#elif defined(_WIN32)
        other.hFile_ = INVALID_HANDLE_VALUE;
        other.hMapFile_ = INVALID_HANDLE_VALUE;
#endif
    }
    
    // Move assignment
    __wc_mapped_file &operator= (__wc_mapped_file&& other) noexcept {
        if (this != &other) {
            // Clean up existing resources
            if (valid()) {
                if (is_stdin_ && data_ != nullptr) {
                    // Free malloc'd memory for stdin
                    free(data_);
                }
#ifdef __linux__
                else if (!is_stdin_ && data_ != nullptr) {
                    munmap(data_, size_);
                }

                if (__fd != -1) {
                    close(__fd);
                }
#elif defined(_WIN32)
                else if (!is_stdin_ && data_ != nullptr) {
                    UnmapViewOfFile(data_);
                }

                if (hMapFile_ != INVALID_HANDLE_VALUE) {
                    CloseHandle(hMapFile_);
                }

                if (hFile_ != INVALID_HANDLE_VALUE) {
                    CloseHandle(hFile_);
                }
#endif
            }
            // Move from other
            data_ = other.data_;
            size_ = other.size_;
#ifdef __linux__
            __fd = other.__fd;
#elif defined(_WIN32)
            hFile_ = other.hFile_;
            hMapFile_ = other.hMapFile_;
#endif
            filename_ = std::move(other.filename_);
            __w_cnt = other.__w_cnt;
            __l_cnt = other.__l_cnt;
            __b_cnt = other.__b_cnt;
            __c_cnt = other.__c_cnt;
            mode_ = other.mode_;
            is_stdin_ = other.is_stdin_;
            // Nullify other
            other.data_ = nullptr;
            other.size_ = 0;
#ifdef __linux__
            other.__fd = -1;
#elif defined(_WIN32)
            other.hFile_ = INVALID_HANDLE_VALUE;
            other.hMapFile_ = INVALID_HANDLE_VALUE;
#endif
        }

        return *this;
    }
    
    virtual ~__wc_mapped_file() {
        if (valid()) {
            if (is_stdin_ && data_ != nullptr) {
                // Free malloc'd memory for stdin
                free(data_);
            }
#ifdef _WIN32
            else {
                if (data_ != nullptr) {
                    UnmapViewOfFile(data_);
                }

                if (hMapFile_ != INVALID_HANDLE_VALUE) {
                    CloseHandle(hMapFile_);
                }

                if (hFile_ != INVALID_HANDLE_VALUE) {
                    CloseHandle(hFile_);
                }
            }
#else
            else {
                if (data_ != nullptr) {
                    munmap(data_, size_);
                }

                if (__fd != -1) {
                    close(__fd);
                }
            }
#endif // _WIN32
        }
    }
    
    [[nodiscard]] __FORCE_INLINE std::span<const char> as_span() const noexcept {
        // If BytesOnly mode, data_ is nullptr, so return empty span
        if (mode_ == __wc_mapped_file_mode::BytesOnly || data_ == nullptr) {
            return {};
        }

        return { static_cast<const char *> (data_), size_ };
    }
    
    [[nodiscard]] __FORCE_INLINE bool valid() const noexcept {
        if (mode_ == __wc_mapped_file_mode::NeedMmap) {
            return data_ != nullptr;
        } else {
            return size_ > 0;
        }
    }
    
    [[nodiscard]] __FORCE_INLINE size_t size() const noexcept {
        return size_;
    }
    
    [[nodiscard]] __FORCE_INLINE std::string filename() const noexcept {
        return filename_;
    }
    
    // Setter
    __FORCE_INLINE void setWordCnt(const size_t w_cnt) noexcept {
        __w_cnt = w_cnt;
    }
    
    __FORCE_INLINE void setLineCnt(const size_t l_cnt) noexcept {
        __l_cnt = l_cnt;
    }
    
    __FORCE_INLINE void setCharCnt(const size_t c_cnt) noexcept {
        __c_cnt = c_cnt;
    }
    
    __FORCE_INLINE void setBytesCnt(const size_t b_cnt) noexcept {
        __b_cnt = b_cnt;
    }
    
    //Getter
    [[nodiscard]] size_t getWordCnt() const noexcept {
        return __w_cnt;
    }
    
    [[nodiscard]] size_t getLineCnt() const noexcept {
        return __l_cnt;
    }
    
    [[nodiscard]] size_t getCharCnt() const noexcept {
        return __c_cnt;
    }
    
    [[nodiscard]] size_t getBytesCnt() const noexcept {
        return __b_cnt;
    }
};
}

#endif // __wc_mapped_fs

#if not defined(__wc_internal_class_already_decl)
#define __wc_internal_class_already_decl

namespace wc_class {
using namespace detail;
using namespace std::literals;

// CPU macro for betteer performance
// Mainly for x86-64
inline constexpr size_t CACHE_LINE_SIZE = 64;
inline constexpr size_t L1_CHUNK = 32 * 1024;
inline constexpr size_t L2_CHUNK = 256 * 1024;
inline constexpr size_t L3_CHUNK = 8 * 1024 * 1024;

// Parallel processing threshold
inline constexpr size_t PARALLEL_THRESHOLD = 512 * 1024;

// The size of a SIMD register
#ifdef __AVX512F__
inline constexpr size_t SIMD_WIDTH {64};

#elif defined(__AVX2__)
inline constexpr size_t SIMD_WIDTH {32};

#elif defined(__SSE2__)
inline constexpr size_t SIMD_WIDTH {16};

#else
inline constexpr size_t SIMD_WIDTH {8};
#endif //__AVX512F__

template <class BitChar, class Translation>
class __wc_internal_class {

  public:
    static __wc_internal_class<BitChar, Translation> *Instance() {
#ifdef __wc_lib_use_std_call_once
#define __wc_lib_use_res_flag
        std::once_flag flag = wc_flag;
        std::call_once(flag, []() {
#if defined(__wc_lib_has_default_value)
            instance = new __wc_internal_class();
#else
            instance = new __wc_internal_class("", std::identity {});
#endif // __wc_lib_has_default_value
        });

        return instance;
#elif defined(__wc_lib_use_std_atomic)
        // Destroy the private member __wc_internal_class* instance;
        // Later, can cause multiple compilation issues if not
        // handled correctly.
        //
        // Needs C++17 or later for std::optional or std::variant
#if defined(__cplusplus) && __cplusplus >= 201703L
#if defined(__wc_lib_private_instance)
        try {
            if (__instance != std::nullopt) {
                __instance->reset();
            };
        }
        catch (std::bad_optional_access &__opt_access) {
            std::cout << __opt_access.what() << '\n';
        };
#endif // __wc_lib_private_instance
        static std::atomic<__wc_internal_class<BitChar, Translation> *> instance;
#else
#error "Compile with C++17 or later"
#endif
        // Here load() return a pointer to a __wc_internal_class.
        auto *mem = instance.load(std::memory_order_acquire);
        
        if (mem == nullptr) {
            std::lock_guard<std::shared_mutex> __guard(lock);
            mem = instance.load(std::memory_order_relaxed);
            if (mem == nullptr) {
                mem = new __wc_internal_class<BitChar, Translation>();
                instance.store(mem, std::memory_order_release);
            }
        }

        return mem;

// The simplest implementation
#elif defined(___wc_lib_use_meyer)
        static __wc_internal_class<BitChar, Translation> instance;
        return instance;
#endif // __wc_lib_use_std_call_once
    }
    
    // Initializing private members argc and argv.
    // Normally, argv is a filename.
    void init(int ___argc, const char **___argv) {
        this->argc = ___argc;
        
        auto views = std::ranges::subrange(___argv, ___argv + ___argc) | std::views::transform([](const char *s) {
            return std::string(s);
        })

#if __cplusplus >= 202302L
        // std::ranges::to was added in C++23, version below C++23 will make you compilator
        // complain about it. It's better to add a safe alternatives.
        | std::ranges::to<std::vector>();
        this->argv = views;
#else
        // Do not remove this semicolon!
        // It is an important one
        ;
        std::vector<std::string> args(views.begin(), views.end());
        this->argv = args;
#endif // __cplusplus
    }
    
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline, hot))
#elif defined(_MSC_VER)
    __forceinline
#endif // __GNUC__
    
#if defined(_MSVC)
    #pragma region __WC_CHAR_IMPL
#endif // _MSVC
    // Maybe great if we use std::variant<>
    size_t __wc_char_0(Translation translation = std::identity{}, size_t f_idx = 0) {
#ifdef __linux__
        // Just some fstat tricks for optimizations
        try {
            std::uintmax_t ____sz = std::filesystem::file_size(this->argv[1]);
            return ____sz;
        }
        catch (std::filesystem::filesystem_error &e) {
            std::cerr << e.what() << std::endl;
        }
        return size_t (-1);
#elif defined(_WIN32)
        // For future perspective.
        // Not fully implemented in this version.
        // Windows API (and programming) is slightly different,
        // we have to use GetFileSizeEx()
        HANDLE hFile = CreateFileA(mapped_file[f_idx].filename().c_str(),
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   NULL,
                                   OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL,
                                   NULL);
        if (hFile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER __sz;
            if (GetFileSizeEx(hFile, &__sz)) {
                CloseHandle(hFile);
                return static_cast<size_t> (__sz.QuadPart);
            }
            CloseHandle(hFile);
        }
#else
        // POSIX standard API
        // If read is not available with the OS API.
        size_t __c_count {};
        std::ifstream file(this->argv[1]);
        for (const char &c : std::string(std::istreambuf_iterator<char> (file),
                                         std::istreambuf_iterator<char>())) {
            __c_count++;
        }
        return __c_count;
#endif // _PLATFORM_SPECIFIC
    }
    
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline, hot))
#elif defined(_MSVC)
    __force_inline
#endif // ___GNUC__
    // Very simple and blazingly fast trick
    // Just fstat() system call.
    // Works with -c
    size_t __wc_char_1(Translation translation = std::identity{}, size_t f_idx = 0) {
        if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
            return 0;
        }
        return mapped_file[f_idx].size();
    }
    
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline, hot))
#elif defined(_MSVC)
    __force_inline
#endif // ___GNUC__
    // Another test
    // This one use multithreading.
    size_t __wc_char_2(Translation translation = std::identity{}) {
        // Should be architecture dependent.
        // Have to check the CPU caracteristics, because too many
        // threads can be worse than single thread sometimes (on some architecure)
        constexpr int num_threads = 12;
        
        std::ifstream file(this->argv[1], std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        std::vector<char> buffer(size);
        
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), size);
        
        size_t chunk_size = size / num_threads;
        std::vector<std::thread> threads;
        std::vector<size_t> counts(num_threads, 0);
        
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
            
            threads.emplace_back([&buffer, start, end, &counts, i, translation]() {
                //counts[i] = end - start;
                // Or with transformation:
                counts[i] = std::count_if(buffer.begin() + start, buffer.begin() + end,
                                        [&](char c) { return translation(c); });
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        return std::accumulate(counts.begin(), counts.end(), size_t (0));
    }
    
#if defined(_MSVC)
    #pragma endregion __WC_CHAR_IMPL
#endif // _MSVC
    
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline, hot))
#elif defined(_MSVC)
    __force_inline
#endif // ___GNUC__
    
#if defined(_MSVC)
    #pragma region __WC_LINE_IMPL
#endif // _MSVC
    // Count line.
    // Linear implementation, not really the good one.
    size_t __wc_line_0(Translation translation = std::identity{}) {
        size_t __l_count {};
        std::ifstream file(this->argv[1], std::ios::binary || std::ios::ate);
        std::stringstream buffer;

        buffer << file.rdbuf();
        for (const char &c : std::string(std::istreambuf_iterator<char> (file),
                                         std::istreambuf_iterator<char>())) {
            if (c == '\n') {
                __l_count++;
            }
        }
        return __l_count;
    }
    
#ifdef __AVX512F__
    [[gnu::target("avx512f")]]
    // Using AVX512
    // Slightly better (Use inline assembly)
    __FORCE_INLINE
    size_t __wc_line_1(Translation translation = std::identity{}, size_t f_idx = 0) noexcept {
        if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
            return 0;
        }
        size_t __l_count {};
        auto __data = mapped_file[f_idx].as_span();
        const char *ptr = __data.data();
        const char *end = ptr + __data.size();
        const __m512i newline = _mm512_set1_epi8('\n');
        // Process 64 bytes at a time
        // If a newline is found on the n-th bits,
        // it is switched to one, else every non newline bits is
        // set to 0. That's what the function _mm512_cmpeq_epi8_mask() does.
        while (LIKELY(ptr + 64 <= end)) {
            __builtin_prefetch(ptr + 128, 0, 0);
            __m512i __chk = _mm512_loadu_si512(reinterpret_cast<const __m512i*> (ptr));
            __mmask64 __m = _mm512_cmpeq_epi8_mask(__chk, newline);
            // Count the number of set bits (population count)
            // for data type unsigned long long
            __l_count += __builtin_popcountll(__m);
            ptr += 64;
        }
        // Handle remainder (Because
        // every input is not forcefully with 0 mod 64.
        while (LIKELY(ptr < end)) {
            __l_count += (*ptr++ == '\n');
        }
        return __l_count;
    }
    
#elif defined(__AVX2__)
    [[gnu::target("avx2")]]
    __FORCE_INLINE
    size_t __wc_line_1(Translation translation = std::identity{}, size_t f_idx = 0) noexcept {
        if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
            std::cerr << "Mapped file is empty" << std::endl;
            return 0;
        }
        size_t __l_count {};
        auto __data = mapped_file[f_idx].as_span();
        const char *ptr = __data.data();
        const char *end = ptr + __data.size();
        const __m256i newline = _mm256_set1_epi8('\n');
        size_t simd_iterations = 0;
        size_t simd_newlines = 0;
        // Same logic as with __AVX512F__
        // Just different instruction set
        // More high level and understandable
        while (LIKELY(ptr + 32 <= end)) {
            // __builtin_prefetch ( ptr + 128, 0, 3 );  // TEMPORARILY DISABLED
            __m256i __chk = _mm256_loadu_si256(reinterpret_cast<const __m256i*> (ptr));
            __m256i __cmp = _mm256_cmpeq_epi8(__chk, newline);
            int __m = _mm256_movemask_epi8(__cmp);
            int count = std::popcount(static_cast<uint32_t> (__m));
            __l_count += count;
            simd_newlines += count;
            simd_iterations++;
            ptr += 32;
        }

        size_t rem = 0;
        while (LIKELY(ptr < end)) {
            rem += (*ptr++ == '\n');
        }

        __l_count += rem;
        return __l_count;
    }
    
#elif defined(__SSE2__)
    //  SSE is the newest instruction set for
    //  x86-64 CPU.
    [[gnu::target("sse2")]]
    __FORCE_INLINE
    size_t __wc_line_1(Translation translation = std::identity{}, size_t f_idx = 0) noexcept {
        if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
            return 0;
        }

        size_t __l_count {};
        auto __data = mapped_file[f_idx].as_span();
        const char *ptr = __data.data();
        const char *end = ptr + __data.size();
        const __m128i newline = _mm_set1_epi8('\n');

        while (LIKELY(ptr + 16 <= end)) {
            __builtin_prefetch(ptr + 64, 0, 3);
            __m128i __chk = _mm_loadu_si128(reinterpret_cast<const __m128i*> (ptr));
            __m128i __cmp = _mm_cmpeq_epi8(__chk, newline);
            int mask = _mm_movemask_epi8(__cmp);
            __l_count += std::popcount(static_cast<uint32_t> (mask));
            ptr += 16;
        }

        while (LIKELY(ptr < end)) {
            __l_count += (*ptr++ == '\n');
        }

        return __l_count;
    }
    
#else // Fallback to a scalar implementation 
    __FORCE_INLINE size_t __wc_line_1(Translation translation = std::identity {}, size_t f_idx = 0)
    noexcept {
        size_t __l_count {};
        auto __data = mapped_file[f_idx].as_span();
        const char *ptr = __data.data();
        const char *end = ptr + __data.size();

        // Process 32 bytes at a time (Ah the mf)
        while (LIKELY(ptr + 32 <= end)) {
            __l_count += (ptr[0] == '\n') + (ptr[1] == '\n') +
                         (ptr[2] == '\n') + (ptr[3] == '\n') +
                         (ptr[4] == '\n') + (ptr[5] == '\n') +
                         (ptr[6] == '\n') + (ptr[7] == '\n') +
                         (ptr[8] == '\n') + (ptr[9] == '\n') +
                         (ptr[10] == '\n') + (ptr[11] == '\n') +
                         (ptr[12] == '\n') + (ptr[13] == '\n') +
                         (ptr[14] == '\n') + (ptr[15] == '\n') +
                         (ptr[16] == '\n') + (ptr[17] == '\n') +
                         (ptr[18] == '\n') + (ptr[19] == '\n') +
                         (ptr[20] == '\n') + (ptr[21] == '\n') +
                         (ptr[22] == '\n') + (ptr[23] == '\n') +
                         (ptr[24] == '\n') + (ptr[25] == '\n') +
                         (ptr[26] == '\n') + (ptr[27] == '\n') +
                         (ptr[28] == '\n') + (ptr[29] == '\n') +
                         (ptr[30] == '\n') + (ptr[31] == '\n');
            ptr += 32;
        }

        // Handle remainder
        while (LIKELY(ptr < end)) {
            __l_count += (*ptr++ == '\n');
        }

        return __l_count;
    }
    
#endif // __AVX512F__
    
#if defined(_MSVC)
    #pragma endregion  __WC_LINE_IMPL
#endif // _MSVC
    
    
#if defined(_MSVC)
    #pragma region __WC_WORD_IMPL
#endif // _MSVC 
    
    // Dummy function, will break on very large file
    // theorically.
    size_t __wc_word_0(Translation translation = std::identity{}, size_t f_idx = 0) {
        size_t __w_count {}, pos = 0;
        auto __data = mapped_file[f_idx].as_span();
        auto __str = std::string_view(__data.data());

        while (LIKELY(true))
            [[likely]] {
            if ((pos = __str.find_first_not_of(" \r\n\t", pos + 1)) == __str.npos) {
                break;
            }
            
            __w_count++;
            
            if ((pos = __str.find_first_of(" \r\n\t", pos + 1)) == __str.npos) {
                break;
            }
        }
        return __w_count;
    }
    
#if defined(_MSVC)
    #pragma region __WC_CHAR_M_IMPL
#endif
    
#ifdef __AVX512F__
    [[gnu::target("avx512f")]]
    __FORCE_INLINE
    size_t __wc_char_m(Translation translation = std::identity{}, size_t f_idx = 0) noexcept {
        if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
            return 0;
        }

        auto __data = mapped_file[f_idx].as_span();
        const char *ptr = __data.data();
        const char *end = ptr + __data.size();
        size_t char_count = 0;

        // UTF-8 continuation bytes: 10xxxxxx (top 2 bits = 10)
        const __m512i continuation_mask = _mm512_set1_epi8(0xC0); // 11000000
        const __m512i continuation_pattern = _mm512_set1_epi8(0x80); // 10000000

        while (LIKELY(ptr + 64 <= end)) {
            __builtin_prefetch(ptr + 128, 0, 0);
            __m512i chunk = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
            // Mask to get top 2 bits of each byte
            __m512i masked = _mm512_and_si512(chunk, continuation_mask);
            // Compare with continuation pattern (10xxxxxx)
            __mmask64 is_continuation = _mm512_cmpeq_epi8_mask(masked, continuation_pattern);
            // Count non-continuation bytes (these are character starts)
            char_count += 64 - __builtin_popcountll(is_continuation);
            ptr += 64;
        }

        // Handle remainder
        while (LIKELY(ptr < end)) {
            char_count += ((*ptr & 0xC0) != 0x80);
            ptr++;
        }

        return char_count;
    }
    
    
#elif defined(__AVX2__)
    [[gnu::target("avx2")]]
    __FORCE_INLINE
    size_t __wc_char_m(Translation translation = std::identity{}, size_t f_idx = 0) noexcept {
        if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
            return 0;
        }

        auto __data = mapped_file[f_idx].as_span();
        const char *ptr = __data.data();
        const char *end = ptr + __data.size();
        size_t char_count = 0;
        // UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF)
        // We count all bytes that are NOT continuation bytes
        const __m256i continuation_mask = _mm256_set1_epi8(0xC0); // 11000000
        const __m256i continuation_pattern = _mm256_set1_epi8(0x80); // 10000000

        while (LIKELY(ptr + 32 <= end)) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            // Get top 2 bits: (byte & 11000000)
            __m256i masked = _mm256_and_si256(chunk, continuation_mask);
            // Compare with 10000000 (continuation bytes)
            __m256i is_continuation = _mm256_cmpeq_epi8(masked, continuation_pattern);
            // Get mask of continuation bytes
            int continuation_bits = _mm256_movemask_epi8(is_continuation);
            // Count non-continuation bytes (characters)
            // 32 total bytes - number of continuation bytes
            char_count += 32 - __builtin_popcount(static_cast<uint32_t>(continuation_bits));
            ptr += 32;
        }
        // Handle remainder
        while (LIKELY(ptr < end)) {
            // A byte is a character start if top 2 bits are NOT 10
            char_count += ((*ptr & 0xC0) != 0x80);
            ptr++;
        }

        return char_count;
    }
    
#elif defined(__SSE2__)
    [[gnu::target("sse2")]]
    __FORCE_INLINE
    size_t __wc_char_m(Translation translation = std::identity{}, size_t f_idx = 0) noexcept {
        if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
            return 0;
        }

        auto __data = mapped_file[f_idx].as_span();
        const char *ptr = __data.data();
        const char *end = ptr + __data.size();
        size_t char_count = 0;
        const __m128i continuation_mask = _mm_set1_epi8(0xC0);
        const __m128i continuation_pattern = _mm_set1_epi8(0x80);

        while (LIKELY(ptr + 16 <= end)) {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
            __m128i masked = _mm_and_si128(chunk, continuation_mask);
            __m128i is_continuation = _mm_cmpeq_epi8(masked, continuation_pattern);
            int continuation_bits = _mm_movemask_epi8(is_continuation);
            char_count += 16 - __builtin_popcount(static_cast<uint32_t>(continuation_bits));
            ptr += 16;
        }

        while (LIKELY(ptr < end)) {
            char_count += ((*ptr & 0xC0) != 0x80);
            ptr++;
        }

        return char_count;
    }
    
#else
    // Scalar fallback
    __FORCE_INLINE
    size_t __wc_char_m(Translation translation = std::identity{}, size_t f_idx = 0) noexcept {
        if (mapped_file.empty() || !mapped_file[f_idx].valid()) {
            return 0;
        }

        auto __data = mapped_file[f_idx].as_span();
        const char *ptr = __data.data();
        const char *end = ptr + __data.size();
        size_t char_count = 0;

        // Process 32 bytes at a time
        while (LIKELY(ptr + 32 <= end)) {
            char_count +=
                ((ptr[0] & 0xC0) != 0x80) + ((ptr[1] & 0xC0) != 0x80) +
                ((ptr[2] & 0xC0) != 0x80) + ((ptr[3] & 0xC0) != 0x80) +
                ((ptr[4] & 0xC0) != 0x80) + ((ptr[5] & 0xC0) != 0x80) +
                ((ptr[6] & 0xC0) != 0x80) + ((ptr[7] & 0xC0) != 0x80) +
                ((ptr[8] & 0xC0) != 0x80) + ((ptr[9] & 0xC0) != 0x80) +
                ((ptr[10] & 0xC0) != 0x80) + ((ptr[11] & 0xC0) != 0x80) +
                ((ptr[12] & 0xC0) != 0x80) + ((ptr[13] & 0xC0) != 0x80) +
                ((ptr[14] & 0xC0) != 0x80) + ((ptr[15] & 0xC0) != 0x80) +
                ((ptr[16] & 0xC0) != 0x80) + ((ptr[17] & 0xC0) != 0x80) +
                ((ptr[18] & 0xC0) != 0x80) + ((ptr[19] & 0xC0) != 0x80) +
                ((ptr[20] & 0xC0) != 0x80) + ((ptr[21] & 0xC0) != 0x80) +
                ((ptr[22] & 0xC0) != 0x80) + ((ptr[23] & 0xC0) != 0x80) +
                ((ptr[24] & 0xC0) != 0x80) + ((ptr[25] & 0xC0) != 0x80) +
                ((ptr[26] & 0xC0) != 0x80) + ((ptr[27] & 0xC0) != 0x80) +
                ((ptr[28] & 0xC0) != 0x80) + ((ptr[29] & 0xC0) != 0x80) +
                ((ptr[30] & 0xC0) != 0x80) + ((ptr[31] & 0xC0) != 0x80);
            ptr += 32;
        }

        while (LIKELY(ptr < end)) {
            char_count += ((*ptr++ & 0xC0) != 0x80);
        }

        return char_count;
    }
    
#endif // __AVX512F__
    
#if defined(_MSVC)
    #pragma endregion __WC_CHAR_M_IMPL
#endif
    
    
    
    [[nodiscard]] __FORCE_INLINE size_t getTotalWord() const noexcept {
        return total_word;
    }
    
    [[nodiscard]] __FORCE_INLINE size_t getTotalLine() const noexcept {
        return total_line;
    }
    
    [[nodiscard]] __FORCE_INLINE size_t getTotalChar() const noexcept {
        return total_char;
    }
    
    [[nodiscard]] __FORCE_INLINE size_t getTotalBytes() const noexcept {
        return total_bytes;
    }
    
    // Last call
    __FORCE_INLINE void printTotal() const noexcept {
        for (const auto &file : mapped_file) {
            if (count_line) {
                std::cout << std::setw(__max_line_width) << file.getLineCnt() << ' ';
            }
            if (count_word) {
                std::cout << std::setw(__max_word_width) << file.getWordCnt() << ' ';
            }
            if (count_char) {
                std::cout << std::setw(__max_char_width) << file.getCharCnt() << ' ';
            }
            if (count_bytes) {
                std::cout << std::setw(__max_bytes_width) << file.getBytesCnt() << ' ';
            }
            std::cout << file.filename() << std::endl;
        }
        if (count_line) {
            std::cout << std::setw(__max_line_width) << total_line << ' ';
        }
        if (count_word) {
            std::cout << std::setw(__max_word_width) << total_word << ' ';
        }
        if (count_char) {
            std::cout << std::setw(__max_char_width) << total_char << ' ';
        }
        if (count_bytes) {
            std::cout << std::setw(__max_bytes_width) << total_bytes << ' ';
        }
        std::cout << "total" << std::endl;
    }
    
#if defined(_MSVC)
    #pragma endregion __WC_WORD_IMPL
#endif // _MSVC 
    // Global wrapper for every command line options
    void wc(Translation __local_transform = std::identity{}) {
        size_t var {};
        __parse_argv();

        for (int i = 0; i < mapped_file.size(); ++i) {
            if (count_line) {
                var = __wc_line_1(__local_transform, i);
                mapped_file[i].setLineCnt(var);
                __max_line_width = std::max(__max_line_width, detail::__int_width(var));
                total_line += var;
            }
            if (count_word) {
                var = __wc_word_0(__local_transform, i);
                mapped_file[i].setWordCnt(var);
                __max_word_width = std::max(__max_word_width, detail::__int_width(var));
                total_word += var;
            }
            if (count_bytes) {
                var = __wc_char_1(__local_transform, i);
                mapped_file[i].setBytesCnt(var);
                __max_bytes_width = std::max(__max_bytes_width, detail::__int_width(var));
                total_bytes += var;
            }
            if (count_char) {
                var = __wc_char_m(__local_transform, i);
                mapped_file[i].setCharCnt(var);
                __max_char_width = std::max(__max_bytes_width, detail::__int_width(var));
                total_char += var;
            }
        }
    }
    
    void wc_parallel_0(Translation __local_transform = std::identity{}) {
        size_t var{};
        __parse_argv();
        auto* pool = tp::__wc_thread_pool<>::Instance();
        std::vector<std::future<void>> futures;

        for (size_t i = 0; i < mapped_file.size(); ++i) {
            auto future = pool->submit([this, i, __local_transform]() {
                size_t var{};
                if (count_line) {
                    var = __wc_line_1(__local_transform, i);
                    mapped_file[i].setLineCnt(var);
                }
                if (count_word) {
                    var = __wc_word_0(__local_transform, i);
                    mapped_file[i].setWordCnt(var);
                }
                if (count_bytes) {
                    var = __wc_char_1(__local_transform, i);
                    mapped_file[i].setBytesCnt(var);
                }
            });

            futures.push_back(std::move(future));
        }

        for (auto& future : futures) {
            future.get();
        }

        for (const auto& file : mapped_file) {
            if (count_line) {
                auto line_cnt = file.getLineCnt();
                __max_line_width = std::max(__max_line_width, detail::__int_width(line_cnt));
                total_line += line_cnt;
            }
            if (count_word) {
                auto word_cnt = file.getWordCnt();
                __max_word_width = std::max(__max_word_width, detail::__int_width(word_cnt));
                total_word += word_cnt;
            }
            if (count_bytes) {
                auto bytes_cnt = file.getBytesCnt();
                __max_bytes_width = std::max(__max_bytes_width, detail::__int_width(bytes_cnt));
                total_bytes += bytes_cnt;
            }
        }
    }
    
    
    void wc_parallel_operations(Translation __local_transform = std::identity{}) {
        __parse_argv();
        auto* pool = tp::__wc_thread_pool<>::Instance();
        std::vector<std::future<void>> futures;

        if (count_line) {
            for (size_t i = 0; i < mapped_file.size(); ++i) {
                futures.push_back(pool->submit([this, i, __local_transform]() {
                    auto var = __wc_line_1(__local_transform, i);
                    mapped_file[i].setLineCnt(var);
                }));
            }
        }
        if (count_word) {
            for (size_t i = 0; i < mapped_file.size(); ++i) {
                futures.push_back(pool->submit([this, i, __local_transform]() {
                    auto var = __wc_word_0(__local_transform, i);
                    mapped_file[i].setWordCnt(var);
                }));
            }
        }
        if (count_bytes) {
            for (size_t i = 0; i < mapped_file.size(); ++i) {
                futures.push_back(pool->submit([this, i, __local_transform]() {
                    auto var = __wc_char_1(__local_transform, i);
                    mapped_file[i].setBytesCnt(var);
                }));
            }
        }

        for (auto& future : futures) {
            future.get();
        }

        // Calculate totals (sequential)
        for (const auto& file : mapped_file) {
            if (count_line) {
                auto line_cnt = file.getLineCnt();
                __max_line_width = std::max(__max_line_width, detail::__int_width(line_cnt));
                total_line += line_cnt;
            }
            if (count_word) {
                auto word_cnt = file.getWordCnt();
                __max_word_width = std::max(__max_word_width, detail::__int_width(word_cnt));
                total_word += word_cnt;
            }
            if (count_bytes) {
                auto bytes_cnt = file.getBytesCnt();
                __max_bytes_width = std::max(__max_bytes_width, detail::__int_width(bytes_cnt));
                total_bytes += bytes_cnt;
            }
        }
    }
    
    void wc_parallel_hybrid(Translation __local_transform = std::identity{}) {
        __parse_argv();
        auto* pool = tp::__wc_thread_pool<>::Instance();
        const size_t num_files = mapped_file.size();
        const size_t num_threads = pool->thread_count();

        // Handle single file case - no parallelization needed at file level
        if (num_files == 1) {
            size_t var{};
            if (count_line) {
                var = __wc_line_1(__local_transform, 0);
                mapped_file[0].setLineCnt(var);
                total_line = var;
                __max_line_width = detail::__int_width(var);
            }
            if (count_word) {
                var = __wc_word_0(__local_transform, 0);
                mapped_file[0].setWordCnt(var);
                total_word = var;
                __max_word_width = detail::__int_width(var);
            }
            if (count_bytes) {
                var = __wc_char_1(__local_transform, 0);
                mapped_file[0].setBytesCnt(var);
                total_bytes = var;
                __max_bytes_width = detail::__int_width(var);
            }
            if (count_char) {
                var = __wc_char_m(__local_transform, 0);
                mapped_file[0].setCharCnt(var);
                total_char = var;
                __max_char_width = detail::__int_width(var);
            }
            return;
        }

        // Per-thread accumulators to minimize atomic contention
        struct alignas(64) ThreadLocalAccumulator {
            size_t total_line = 0;
            size_t total_word = 0;
            size_t total_bytes = 0;
            size_t total_char = 0;
            size_t max_line_width = 0;
            size_t max_word_width = 0;
            size_t max_bytes_width = 0;
            size_t max_char_width = 0;
        };

        std::vector<ThreadLocalAccumulator> accumulators(num_threads);
        std::vector<std::future<void>> futures;
        futures.reserve(num_files);
        // Determine optimal chunk size for work distribution
        const size_t min_chunk_size = 1;
        const size_t max_chunk_size = 10;
        const size_t chunk_size = std::clamp(
                                      num_files / (num_threads * 4),
                                      min_chunk_size,
                                      max_chunk_size
                                  );
        // Process files in chunks
        for (size_t chunk_start = 0; chunk_start < num_files; chunk_start += chunk_size) {
            size_t chunk_end = std::min(chunk_start + chunk_size, num_files);
            size_t thread_idx = (chunk_start / chunk_size) % num_threads;
            futures.push_back(pool->submit([ &, chunk_start, chunk_end, thread_idx, __local_transform]() {
                auto& acc = accumulators[thread_idx];
                for (size_t i = chunk_start; i < chunk_end; ++i) {
                    size_t var{};
                    if (count_line) {
                        var = __wc_line_1(__local_transform, i);
                        mapped_file[i].setLineCnt(var);
                        acc.total_line += var;
                        acc.max_line_width = std::max(acc.max_line_width, detail::__int_width(var));
                    }
                    if (count_word) {
                        var = __wc_word_0(__local_transform, i);
                        mapped_file[i].setWordCnt(var);
                        acc.total_word += var;
                        acc.max_word_width = std::max(acc.max_word_width, detail::__int_width(var));
                    }
                    if (count_bytes) {
                        var = __wc_char_1(__local_transform, i);
                        mapped_file[i].setBytesCnt(var);
                        acc.total_bytes += var;
                        acc.max_bytes_width = std::max(acc.max_bytes_width, detail::__int_width(var));
                    }
                    if (count_char) {
                        var = __wc_char_m(__local_transform, i);
                        mapped_file[i].setCharCnt(var);
                        acc.total_bytes += var;
                        acc.max_char_width = std::max(acc.max_char_width, detail::__int_width(var));
                    }
                }
            }));
        }
        // Wait for all tasks
        for (auto& future : futures) {
            future.get();
        }

        // Single-threaded reduction of results
        total_line = 0;
        total_word = 0;
        total_bytes = 0;
        total_char = 0;
        __max_line_width = 0;
        __max_word_width = 0;
        __max_bytes_width = 0;

        for (const auto& acc : accumulators) {
            total_line += acc.total_line;
            total_word += acc.total_word;
            total_bytes += acc.total_bytes;
            __max_line_width = std::max(__max_line_width, acc.max_line_width);
            __max_word_width = std::max(__max_word_width, acc.max_word_width);
            __max_bytes_width = std::max(__max_bytes_width, acc.max_bytes_width);
        }
    }
    
  private:
#if not defined(__wc_lib_use_std_atomic)
#define __wc_lib_private_instance
    static std::optional<__wc_internal_class<BitChar, Translation>> __instance;
#endif // __wc_lib_use_std_atomic
    static std::shared_mutex lock;
    static std::once_flag wc_flag;
    
    int argc;
    std::vector<std::string> argv;
    
    __wc_internal_class() {}
    
    __wc_internal_class(BitChar __bc, Translation translation) {}
    
    bool count_line;
    bool count_bytes;
    bool count_char;
    bool count_word;
    
    size_t total_line{};
    size_t total_bytes{};
    size_t total_char{};
    size_t total_word{};
    
    size_t __max_line_width{};
    size_t __max_word_width{};
    size_t __max_char_width{};
    size_t __max_bytes_width{};
    
    std::vector<fs::__wc_mapped_file> mapped_file;
    
    // argv is already a member of the class
    inline void __parse_argv() {
        // Check for --version and --help first
        for (const auto& arg : argv) {
            if (arg == "--version") {
                std::cout << "fast-wc (GNU coreutils) 1.0.0\n"
                          << "Copyright (C) 2026 Free Software Foundation, Inc.\n"
                          << "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>.\n"
                          << "This is free software: you are free to change and redistribute it.\n"
                          << "There is NO WARRANTY, to the extent permitted by law.\n"
                          << "Written by Sergio Randriamihoatra.\n";
                std::exit(0);
            }

            if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [OPTION]... [FILE]...\n"
                          << "  or:  " << argv[0] << " [OPTION]... --files0-from=F\n"
                          << "Print newline, word, and byte counts for each FILE, and a total line if\n"
                          << "more than one FILE is specified.  A word is a nonempty sequence of non white\n"
                          << "space delimited by white space characters or by start or end of input.\n"
                          << "With no FILE, or when FILE is -, read standard input.\n"
                          << "\n"
                          << "The options below may be used to select which counts are printed, always in\n"
                          << "the following order: newline, word, character, byte, maximum line length.\n"
                          << "  -c, --bytes            print the byte counts\n"
                          << "  -m, --chars            print the character counts\n"
                          << "  -l, --lines            print the newline counts\n"
                          << "      --files0-from=F    read input from the files specified by\n"
                          << "                           NUL-terminated names in file F;\n"
                          << "                           If F is - then read names from standard input\n"
                          << "  -L, --max-line-length  print the maximum display width\n"
                          << "  -w, --words            print the word counts\n"
                          << "      --total=WHEN       when to print a line with total counts;\n"
                          << "                           WHEN can be: auto, always, only, never\n"
                          << "      --help        display this help and exit\n"
                          << "      --version     output version information and exit\n"
                          << "\n"
                          << "Report bugs to: sergiorandriamihoatra@gmail.com\n";
                std::exit(0);
            }
        }
        auto pb = __wc_argparser_builder<>()
                  .flag<bool> ('l', "lines"sv)
                  .flag<bool> ('c', "bytes"sv)
                  .flag<bool> ('m', "chars"sv)
                  .flag<bool> ('w', "words"sv);
        auto parser = pb.build();
        // Need to convert back to const char **
        std::vector<const char *> ___argv;

        for (const auto& arg : argv) {
            ___argv.push_back(arg.c_str());
        }
        parser.parse(this->argc, ___argv.data());
        // Set flags from the parser object
        count_line = parser.has('l');
        count_bytes = parser.has('c');
        count_char = parser.has('m');
        count_word = parser.has('w');

        fs::__wc_mapped_file_mode __m = fs::__wc_mapped_file_mode::NeedMmap;
        if (UNLIKELY(count_bytes && !count_word && !count_char && !count_line))
            [[unlikely]] {
            __m = fs::__wc_mapped_file_mode::BytesOnly;
        }

        if (UNLIKELY(!count_word && !count_char && !count_line && !count_bytes))
            [[unlikely]] {
            count_bytes = count_line = count_word = true;
        }

        mapped_file.clear();
        bool has_files = false;
        for (const std::string &s : argv) {
            if (s != argv[0]) {
                // Handle "-" as stdin
                if (s == "-") {
                    mapped_file.push_back(fs::__wc_mapped_file());
                    has_files = true;
                }
                // Skip flags (starting with -)
                else if (s[0] != '-') {
                    mapped_file.push_back(fs::__wc_mapped_file(s, __m));
                    has_files = true;
                }
            }
        }
        // If no files provided, read from stdin
        if (!has_files) {
            mapped_file.push_back(fs::__wc_mapped_file());
        }
    }
};

// If __wc_lib_use_std_atomic is used, (which means
// another instance has already been declared,
// from __wc_internal_class::Instance()),
// do not declare the private member instance.

// Initializing and declaring static member
#if not defined(__wc_lib_use_std_atomic)
    template <class BitChar, class Translation>
    std::optional<__wc_internal_class<BitChar, Translation>>
    __wc_internal_class<BitChar, Translation>::instance;
#endif // __wc_lib_use_std_atomic
#ifndef _WIN32
    template <class BitChar, class Translation>
    std::shared_mutex __wc_internal_class<BitChar, Translation>::lock;
    
    template <class BitChar, class Translation>
    std::once_flag __wc_internal_class<BitChar, Translation>::wc_flag;
#endif // _WIN32
} // namespace wc_class

#ifdef _WIN32
    template <class BitChar, class Translation>
    std::shared_mutex wc_class::__wc_internal_class<BitChar, Translation>::lock;
    
    template <class BitChar, class Translation>
    std::once_flag wc_class::__wc_internal_class<BitChar, Translation>::wc_flag;
#endif // _WIN32

#endif // __wc_internal_class

template <class BitChar, class Translation>
inline constexpr wc_class::__wc_internal_class<BitChar, Translation>
wc_class_instance =
    wc_class::__wc_internal_class<BitChar, Translation>::Instance();


#define __cpp_lib_using_global_var


// Imagine if we could implement a {w,l} counter on compressed object.
// The same object, but with fewer lines.
//
// We have to study though if the time compressing the object + time counting
// remaining lines <= time counting full line ?
//
// Under which assumptions this last statement is true ?
int main(int argc, const char **argv) {
    std::cout.sync_with_stdio(false);
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    auto __wcObject0 =
        wc_class::__wc_internal_class<std::char_traits<char>,
        std::function<std::char_traits<char> (std::char_traits<char>) >>::Instance();
    auto __wcObject1 =
        wc_class::__wc_internal_class<std::char_traits<wchar_t>, std::vector<char>>::Instance();
#if __cplusplus >= 202002L
    auto __wcObject2 =
        wc_class::__wc_internal_class<std::char_traits<char8_t>, std::vector<char>>::Instance();
#endif // __cplusplus
    auto __wcObject3 =
        wc_class::__wc_internal_class<std::char_traits<char16_t>, std::vector<char>>::Instance();
    auto __wcObject4 =
        wc_class::__wc_internal_class<std::char_traits<char32_t>, std::vector<char>>::Instance();
    __wcObject0->init(argc, argv);
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    __wcObject0->wc_parallel_hybrid();
    __wcObject0->printTotal();
    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds> (end - start);
    std::cout << ns.count() << " ns\n";
    return 0;
}
