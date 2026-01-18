/*
 *
 *  Copyright (C) 2026, Sergio Randriamihoatra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

#include <sys/mman.h> 
#include <fcntl.h> 
#include <sys/stat.h>

#define __wc_lib_use_std_atomic

#if __cplusplus >= 20203L
#define __cpp_lib_use_likely
// Can use C++20 [[likely]] for branch 
// prediction. 

#endif // __cplusplus  

#define LIKELY(x) (__builtin_expect(!!(x), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))

#define __FORCE_INLINE __attribute__((always_inline)) inline 

#if not defined(__wc_argv_parse) 
#define __wc_argv_parse 

namespace detail 
{

  inline constexpr int MAX_OPTIONS = 32; 

  // FNV-1a hash 
  // (Fowller-Noll-Vo hash function) 
  constexpr __u32 hash_fnv1a(std::string_view __s) noexcept {
    uint32_t hash = 2166136261u; // FNV offset basis  

    for(const char &c: __s) {
      // Cast to sizeof(char), which means 
      // inside the range -127 ... 128 
      hash ^= static_cast<uint8_t>(c); 
      hash *= 16777619u;  // FNV prime 
    }

    return hash;
  }

  // The default one is too 
  // laggy (Glibc overhead). 
  // TODO: This function should be optimized further. 
  constexpr int __fast_atoi(const char* __s) noexcept {
    int res = 0; 
    bool __neg = false; 

    if (UNLIKELY(*__s == '-')) {
      __neg = true; 
      __s++; 
    }

    while (LIKELY(*__s >= '0' && *__s <= '9')) [[likely]] {
      res = 10 * res + (*__s++ - '0');  
    }

    return __neg ? -res : res;
  }

  // Option type 
  // For example
  // TODO: This design should be revised. 
  enum class OptionType : uint8_t { flag, val, is_multi_time_appear_true }; 

  template <class __DataType> 
    struct __wc_option { 
      char short_name; 
      std::string_view long_name; 
      OptionType __type; 
      __DataType __default_value; 

      constexpr __wc_option(char __s, std::string_view __l, OptionType __t, __DataType def = __DataType{})
        : short_name(__s), long_name(__l), __type(__t), __default_value(def) {}

      constexpr ~__wc_option() = default; 
    };  

#if not defined __wc_argparser_class 
#define __wc_argparser_class 

  // Will be built by a factory builder object 
  // (Factory builder design). 
  // TODO: Should explore another 
  // creational pattern. 
  template <size_t __max_opt = MAX_OPTIONS> 
    class __wc_argparser {
      public: 
        template<typename _Tp> 
          constexpr void add_opt(const __wc_option<_Tp>& __opt) {
            if (UNLIKELY(opt_count >= __max_opt)) [[unlikely]] {
              return; 
            }

            options[opt_count++] = 
            {
              __opt.short_name, 
              __opt.long_name.data(), 
              __opt.long_name.size(), 
              __opt.__type, 
              detail::hash_fnv1a(__opt.long_name)
            };
          }

        __wc_argparser() {} 
        virtual ~__wc_argparser() = default; 

        // Have same arguments as the main function 
        // Inside __wc_internal_class, There should be 
        // a function which convert std::vector<std::string> to 
        // char **. 
        void parse(int argc, const char** argv) noexcept {
          this->pcount = 0; 

          for(int i = 1; i < argc; ++i) {
            const char *__arg = argv[i]; 

            if ( __arg[0] != '-') continue; 
            if ( __arg[1] != '-') {
              __parse_short_options(__arg[1], i, argc, argv); 
              continue; 
            } 
            if ( __arg[2] != '\0') {
              __parse_long_options(__arg + 2, i, argc, argv); 
            }
          }
        }

        // Get option value (after the specified option) 
        [[nodiscard]] 
          std::optional<const char*> get(char short_name) const noexcept {
            for (size_t i = 0; i < pcount; ++i) {
              if (options[parsed[i].__idx].short_name == short_name) {
                return parsed[i].__v; 
              }
            }

            return std::nullopt; 
          }

        // Same but for long options 
        [[nodiscard]]
          std::optional<const char*> get(std::string_view long_name) const noexcept {
            auto hash = detail::hash_fnv1a(long_name);

            for (size_t i = 0; i < pcount; ++i) {
              auto& opt = options[parsed[i].__idx];
              if (opt.hash == hash) {
                return parsed[i].__v;
              }
            }
            return std::nullopt;
          }

        // Check if flag is set 
        // for short option 
        [[nodiscard]]
          __FORCE_INLINE bool has(char __sn) const noexcept 
          {
            return get(__sn).has_value();
          }

        // Check if flag is set 
        // for long option 
        [[nodiscard]]
          __FORCE_INLINE bool has(std::string_view __ln) const noexcept 
          {
            return get(__ln).has_value();
          }

        // Get as integer
        [[nodiscard]]
          __FORCE_INLINE int get_int(char __sn, int __default_value = 0) const noexcept {
            auto val = get(__sn);
            return val ? detail::__fast_atoi(*val) : __default_value;
          }

      private: 
        struct __parsed_option {
          uint8_t __idx; 
          const char* __v; 
        }; 

        // Option model 
        struct option_entry {
          char short_name; 
          const char *long_name; 
          size_t long_name_len; 
          OptionType type; 
          uint32_t hash; 
        }; 

        std::array<__parsed_option, __max_opt> parsed; 
        size_t pcount = 0; 

        std::array<option_entry, __max_opt> options; 
        size_t opt_count = 0;

        void __parse_short_options(char opt, int& idx, int argc, const char** argv) noexcept {
          for (size_t i = 0; i < opt_count; ++i) {
            if (options[i].short_name == opt) {
              if (pcount >= __max_opt) return;

              const char* value = nullptr;
              if (options[i].type != OptionType::flag) {
                if (idx + 1 < argc) {
                  value = argv[++idx];
                }
              }

              parsed[pcount++] = { static_cast<uint8_t>(i), value };
              return;
            }
          }
        }

        void __parse_long_options(const char* opt, int& idx, int argc, const char** argv) noexcept {
          const auto hash = detail::hash_fnv1a(opt);

          for (size_t i = 0; i < opt_count; ++i) {
            if (options[i].hash == hash) {
              if (pcount >= __max_opt) return;

              const char* value = nullptr;
              if (options[i].type != OptionType::flag) {
                if (idx + 1 < argc) {
                  value = argv[++idx];
                }
              }

              parsed[pcount++] = { static_cast<uint8_t>(i), value };
              return;
            }
          }
        }
    };

  // The factory builder class 
  template <size_t __max_opt = MAX_OPTIONS> 
    class __wc_argparser_builder {
      public: 
        constexpr __wc_argparser_builder() = default; 

        // To optimize memory, 
        // __wc_argparser_builder should be unique 
        //constexpr __wc_argparser_builder(const __wc_argparser_builder<__max_opt>& __obj_copy) = delete; 

        template <typename _Tn> 
          constexpr __wc_argparser_builder& flag(char short_name, std::string_view long_name) {
            parser.add_opt(__wc_option<bool>(short_name, long_name, OptionType::flag)); 
            return *this; 
          }

        template <typename _Tn> 
          constexpr __wc_argparser_builder& option(char short_name, std::string_view long_name) {
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

  struct __wc_mapped_file {
    void *data_ {}; 
    size_t size_ {}; 
    int __fd = -1; 

    std::string filename_; 

    explicit __wc_mapped_file(std::string_view __fn)
      :  filename_(__fn) { 
      struct stat __sb; 

      if (UNLIKELY((__fd = open(filename_.c_str(), O_RDONLY)) == -1)) [[unlikely]] {
        return; 
      } 

      if (UNLIKELY(fstat(__fd, &__sb) == -1)) [[unlikely]] {
        close(__fd); 
        __fd = -1; 
        return; 
      }

      if (UNLIKELY((size_ = __sb.st_size) == 0)) [[unlikely]] {
        close(__fd); 
        __fd = -1; 
        return; 
      }

      if (UNLIKELY((data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, __fd, 0)) == MAP_FAILED)) [[unlikely]] {
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
    }


    // No copy constructor (Object should be unique) 
    //__wc_mapped_file(const __wc_mapped_file&) = delete;

    virtual ~__wc_mapped_file() {}

    [[nodiscard]] __FORCE_INLINE std::span<const char> as_span() const noexcept 
    {
      return { static_cast<const char*>(data_), size_ };
    }

    [[nodiscard]] __FORCE_INLINE bool valid() const noexcept 
    {
      return data_ != nullptr; 
    }

    [[nodiscard]] __FORCE_INLINE size_t size() const noexcept 
    {
      return size_; 
    }

    [[nodiscard]] __FORCE_INLINE std::string filename() const noexcept 
    {
      return filename_; 
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
              instance = new __wc_internal_class("", std::identity{});
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

          } catch(std::bad_optional_access &__opt_access) {
            std::cout << __opt_access.what() << '\n'; 
          };
#endif // __wc_lib_private_instance

          static std::atomic<__wc_internal_class<BitChar,Translation> *> instance;

#else 
#error "Compile with C++17 or later" 
#endif

          // Here load() return a pointer to a __wc_internal_class.
          auto *mem = instance.load(std::memory_order_acquire);

          if (mem == nullptr) 
          {
            std::lock_guard<std::shared_mutex> __guard(lock);
            mem = instance.load(std::memory_order_relaxed);

            if (mem == nullptr) 
            {
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
        void init(int __argc, const char *__argv[]) 
        {
          this->argc = __argc; 
          auto views = std::ranges::subrange(__argv, __argv + __argc)
            | std::views::transform([](const char *s) { return std::string(s); })
#if __cplusplus >= 202302L
            // std::ranges::to was added in C++23, version below C++23 will make you compilator 
            // complain about it. It's better to add a safe alternatives. 
            | std::ranges::to<std::vector>();
          this->argv = views; 
#else   
          // Do not remove this semicolon
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
          size_t __wc_char_0(Translation translation = std::identity{}) 
          {
#ifdef __linux__
            // Just some fstat tricks for optimizations
            try {
              std::uintmax_t ____sz = std::filesystem::file_size(this->argv[1]); 

              return ____sz; 
            } catch (std::filesystem::filesystem_error &e) {
              std::cerr << e.what() << std::endl; 
            }

            return size_t(-1); 

#elif defined(_WIN32)
            // For future perspective. 
            // Not fully implemented in this version. 
            // Windows API (and programming) is slightly different, 
            // we have to use GetFileSizeEx()

            HANDLE hFile = CreateFileA(this->argv[1], 
                GENERIC_READ, 
                FILE_SHARE_READ, 
                NULL, 
                OPEN_EXISTING, 
                FILE_ATTRIBUTE_NORMAL, 
                NULL); 

            if (hFile != INVALID_HANDLE_VALUE)
            {
              LARGE_INTEGER __sz; 

              if (GetFileSizeEx(hFile, &size))
              {
                CloseHandle(hFile); 
                return static_cast<size_t>(size.QuadPart); 
              }

              CloseHandle(hFile); 
            }
#else 
            // POSIX standard API
            // If read is not available with the OS API. 
            size_t __c_count {}; 
            std::ifstream file(this->argv[1]); 

            for (const char &c: std::string(std::istreambuf_iterator<char>(file),
                  std::istreambuf_iterator<char>())) 
            {
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
          size_t __wc_char_2(Translation translation = std::identity{}) 
          {
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

            for (size_t i = 0; i < num_threads; ++i) 
            {
              size_t start = i * chunk_size;
              size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;

              threads.emplace_back([&buffer, start, end, &counts, i]() {
                  counts[i] = end - start;  
                  // Or with transformation:
                  // counts[i] = std::count_if(buffer.begin() + start, buffer.begin() + end, 
                  //                         [](char c) { return translation(c); });
                  });
            }

            for (auto& t : threads) {
              t.join();
            }

            return std::accumulate(counts.begin(), counts.end(), size_t(0));
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
          size_t __wc_line_0(Translation translation = std::identity{})
          {
            size_t __l_count {}; 
            std::ifstream file(this->argv[1], std::ios::binary || std::ios::ate); 

            std::stringstream buffer; 
            buffer << file.rdbuf();

            for(const char &c: std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>())) {
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
              while (LIKELY(ptr + 64 <= end)) 
              {
                __builtin_prefetch(ptr + 128, 0, 0);
                __m512i __chk = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
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
                return 0; 
              }


              size_t __l_count {};
              auto __data = mapped_file[f_idx].as_span(); 
              const char* ptr = __data.data();
              const char* end = ptr + __data.size();

              const __m256i newline = _mm256_set1_epi8('\n');

              // Same logic as with __AVX512F__ 
              // Just different instruction set 
              // More high level and understandable
              while (LIKELY(ptr + 32 <= end)) {
                __builtin_prefetch(ptr + 128, 0, 3);

                __m256i __chk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
                __m256i __cmp = _mm256_cmpeq_epi8(__chk, newline);
                int __m = _mm256_movemask_epi8(__cmp);

                __l_count += std::popcount(static_cast<uint32_t>(__m));
                ptr += 32;
              }

              while (LIKELY(ptr < end)) {
                __l_count += (*ptr++ == '\n');
              }

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
              const char* ptr = __data.data();
              const char* end = ptr + __data.size();

              const __m128i newline = _mm_set1_epi8('\n'); 

              while (LIKELY(ptr + 16 <= end)) {
                __builtin_prefetch(ptr + 64, 0, 3);

                __m128i __chk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
                __m128i __cmp = _mm_cmpeq_epi8(__chk, newline);
                int mask = _mm_movemask_epi8(__cmp);

                __l_count += std::popcount(static_cast<uint32_t>(mask));
                ptr += 16;
              }

              while (LIKELY(ptr < end)) {
                __l_count += (*ptr++ == '\n');
              }

              return __l_count;
#else // Fallback to a scalar implementation 
              __FORCE_INLINE size_t __wc_line_1(Translation translation = std::identity{}, size_t f_idx = 0) noexcept {
                size_t __l_count {};
                auto __data = mapped_file[f_idx].as_span(); 
                const char* ptr = __data.data();
                const char* end = ptr + __data.size();

                
                // Process 32 bytes at a time (Ah the motherfucker)
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
                size_t __w_count {}, pos = -1; 
                
                auto __data = mapped_file[f_idx].as_span(); 
                auto __str = std::string_view(__data.data()); 

                while (LIKELY(true)) [[likely]] {
                  if ((pos = __str.find_first_not_of(" \r\n\t", pos + 1)) == __str.npos) break; 
                  __w_count++;
                  if ((pos = __str.find_first_of(" \r\n\t", pos + 1)) == __str.npos) break; 
                }

                return __w_count; 
              }

              [[nodiscard]] __FORCE_INLINE size_t getTotalWord() const noexcept 
              {
                return total_word; 
              }

              [[nodiscard]] __FORCE_INLINE size_t getTotalLine() const noexcept 
              {
                return total_line; 
              }

              [[nodiscard]] __FORCE_INLINE size_t getTotalChar() const noexcept 
              {
                return total_char; 
              }

              [[nodiscard]] __FORCE_INLINE size_t getTotalBytes() const noexcept 
              {
                return total_bytes; 
              }
              
              // Last call 
              __FORCE_INLINE void printTotal() const noexcept 
              {
                if (count_line) std::cout << total_line << '\t'; 
                if (count_word) std::cout << total_word << '\t'; 
                if (count_char) std::cout << total_char << '\t'; 
                if (count_bytes) std::cout << total_bytes << '\t';
                std::cout << "total" << std::endl; 
              }

#if defined(_MSVC) 
#pragma endregion __WC_WORD_IMPL 
#endif // _MSVC 
              // Global wrapper for every command line options
              void wc(Translation __local_transform = std::identity{})
              {
                size_t var {}; 

                __parse_argv(); 
                for (int i=0; i < mapped_file.size(); ++i) {
                  if (count_line) {
                    var = __wc_line_1(__local_transform, i);
                    std::cout << var << '\t'; 
                    total_line += var; 
                  }

                  if (count_word) {
                    var = __wc_word_0(__local_transform, i); 
                    std::cout << var << '\t'; 
                    total_word += var; 
                  }
                  
                  if (count_bytes) {
                    var = __wc_char_1(__local_transform, i); 
                    std::cout << var << '\t'; 
                    total_bytes += var; 
                  }

                  std::cout << mapped_file[i].filename() << std::endl;  
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

              std::vector<fs::__wc_mapped_file> mapped_file; 

              // TODO: Implement __parse_argv()
              //
              // argv is already a member of the class
              inline void __parse_argv() {
                auto pb = __wc_argparser_builder<>()
                  .flag<bool>('l', "line"sv) 
                  .flag<bool>('c', "bytes"sv)
                  .flag<bool>('m', "chars"sv)
                  .flag<bool>('w', "words"sv); 


                auto parser = pb.build(); 

                // Need to convert back to const char ** 
                std::vector<const char*> __argv; 
                for(const auto& arg : argv) {
                  __argv.push_back(arg.c_str()); 
                }

                parser.parse(this->argc, __argv.data()); 

                // Set flags from the parser object 
                count_line = parser.has('l'); 
                count_bytes = parser.has('c'); 
                count_char = parser.has('m'); 
                count_word = parser.has('w');

                if (UNLIKELY(!count_word && !count_char && !count_line && !count_bytes)) [[unlikely]] {
                  count_bytes = true; 
                }

                mapped_file.clear();
                for (const std::string &s: argv) {
                  if (s[0] != '-' && s != argv[0]) mapped_file.push_back(fs::__wc_mapped_file(s)); 
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

        template <class BitChar, class Translation>
          std::shared_mutex __wc_internal_class<BitChar, Translation>::lock;

        template <class BitChar, class Translation>
          std::once_flag __wc_internal_class<BitChar, Translation>::wc_flag;
    } // namespace wc_class

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
  int main(int argc, const char *argv[]) {
    std::cout.sync_with_stdio(false); 
    std::ios_base::sync_with_stdio(false); 
    std::cin.tie(nullptr); 

    auto __wcObject0 =
      wc_class::__wc_internal_class<std::char_traits<char>, 
      std::function<std::char_traits<char>(std::char_traits<char>)>>::Instance();
    auto __wcObject1 =
      wc_class::__wc_internal_class<std::char_traits<wchar_t>, std::vector<char>>::Instance();

#if __cplusplus >= 20203L 
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
    __wcObject0->wc();
    __wcObject0->printTotal(); 
    auto end = std::chrono::high_resolution_clock::now(); 

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start); 
    std::cout << ns.count() << " ns\n"; 

    return 0;
  }
