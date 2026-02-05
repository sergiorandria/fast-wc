# fast-wc — Fast C++ Word Count Utility

**A compact, high-performance C++ reimplementation of GNU coreutils `wc`.**
This repository is a research project that explores the *low-boundary* performance of a correct `wc` implementation — i.e., how small the constant overhead can be while keeping correctness, portability and simplicity. The implementation is usable today, but several GNU `wc` flags are intentionally not implemented yet (see *Status / Limitations*).

---

## What / Abstract

`fast-wc` implements the common `wc` functionality (lines, words, characters, bytes) in modern C++ (C++17) with an emphasis on minimizing runtime overhead for small-to-medium text workloads. The project investigates microbenchmarkable implementation choices (I/O buffering, byte-wise vs block parsing, thread-parallel counting) while keeping the codebase compact and readable for measurement and reasoning.

---

## Features

* Correct counting for: **lines** (`-l`), **words** (`-w`), **characters** (`-m`), **bytes** (`-c`).
* Portable C++17 code with no heavy external dependencies.
* Minimal, readable implementation intended for benchmarking and experimentation.
* Optional parallel counting mode (research prototype) to explore throughput on multi-core systems.

---

## Status / Limitations

**Implemented**

* `-l` (lines)
* `-w` (words)
* `-m` (characters)
* `-c` (bytes)

**Not implemented / TODO**

* `-L` (max line length)
* `--total=WHEN`
* `--files0-from=F` (i.e. `--file0-from=F`)
* Full GNU compatibility edge-cases and some flags

The repository is both a usable tool and an ongoing research artifact. Contributions that preserve the repository’s microbenchmark-friendly nature are especially welcome.

---

## Benchmarks (measured with `hyperfine`)

Microbenchmarks were performed on `test.txt` repeated three times (`test.txt test.txt test.txt`). `hyperfine` warns about measurements shorter than ~5 ms: use larger inputs or `--shell=none` for more robust numbers.

**Summary (mean times):**

* `/usr/bin/wc -l -c -w -m test.txt test.txt test.txt` — **2.2 ms (mean)**
* `./wc -l -c -w -m test.txt test.txt test.txt` — **0.9392 ms (mean)**

That corresponds to approximately **2.34×** speedup on this microbenchmark (2.2 ms ÷ 0.9392 ms ≈ 2.34).

Full `hyperfine` outputs captured during testing:

```
Benchmark 1: /usr/bin/wc  -l -c -w -m test.txt test.txt test.txt
  Time (mean ± σ):       2.2 ms ±   0.5 ms    [User: 4.1 ms, System: 0.2 ms]
  Range (min … max):     1.3 ms …   5.4 ms    244 runs

  Warning: Command took less than 5 ms to complete. Note that the results might be inaccurate because hyperfine can not calibrate the shell startup time much more precise than this limit. You can try to use the `-N`/`--shell=none` option to disable the shell completely.

Benchmark 1: ./wc  -l -c -w -m test.txt test.txt test.txt
  Time (mean ± σ):     939.2 µs ± 604.0 µs    [User: 2920.2 µs, System: 1293.3 µs]
  Range (min … max):   142.3 µs … 4444.8 µs    332 runs

  Warning: Command took less than 5 ms to complete. Note that the results might be inaccurate because hyperfine can not calibrate the shell startup time much more precise than this limit. You can try to use the `-N`/`--shell=none` option to disable the shell completely.
```

**Benchmarking recommendations**

* Use large inputs (MBs–GBs) to reduce shell and startup noise.
* Run `hyperfine --warmup 5 --export-json results.json --shell=none "command"` for more robust results.
* Profile with `perf`/`callgrind` to find hotspots.

---

## Parallel counter — research notes

The repository contains a research prototype for a **parallel counter**. The parallel approach is intended to investigate how far throughput can be pushed on multi-core machines while preserving correctness.

Key ideas and trade-offs explored:

1. **Chunking & boundary correctness**

   * File is split into disjoint byte-ranges (chunks) assigned to worker threads.
   * Word counting requires care at chunk boundaries: a word might start in chunk `i` and finish in chunk `i+1`. To handle this correctly we:

     * scan the first and last bytes of a chunk to determine whether the chunk-initial byte continues a preceding word, or whether the chunk-final byte leaves a dangling partial token; or
     * overlap chunks by a small sentinel region (e.g., 1–2 bytes) and have each thread ignore overlap when reducing counts, or
     * post-process adjacent chunk results to correct for boundary-spanning tokens.
   * Correctness is crucial; micro-optimizations that break boundary handling are unacceptable for research claims.

2. **Per-thread counters & reduction**

   * Each thread keeps local counters (`lines`, `words`, `chars`, `bytes`) in thread-local storage to avoid atomic operations on the hot path.
   * After processing, threads reduce into global totals using a single synchronized step. Use a single `std::mutex` or atomic `fetch_add` only during the small reduction phase — this minimizes contention.
   * To avoid false sharing, pad per-thread counters to cache-line size (e.g., align to 64 bytes).

3. **I/O strategy**

   * Evaluate `read()` with a large buffer, `mmap()` with page-aligned chunking, and buffered `ifstream` approaches.
   * `mmap()` reduces copy overhead for very large files but requires careful handling for portability and memory usage. It can also change page-fault patterns that affect measurements.
   * Block reads with a large buffer and single-threaded prefetch can be simpler and competitive for many workloads.

4. **Work distribution**

   * Static chunk assignment is simple and has predictable overhead, but dynamic assignment (work-stealing) can help for skewed workloads.
   * For uniform large files, static division by file size is a good starting point.

5. **Synchronization / memory ordering**

   * Keep synchronization outside the inner parsing loop.
   * Use release/acquire semantics during reduction if necessary, but prefer simple `fetch_add` or one-shot `mutex` for clarity in a research setting.

6. **Microbenchmark pitfalls**

   * Speedups on small files can be dominated by process startup, shell overhead, and caching effects. Run large inputs and multiple warmups.
   * Measure both throughput (bytes/sec) and latency (time to completion) when evaluating parallelization.

7. **Correctness tests**

   * Add tests that compare outputs with `/usr/bin/wc` on many inputs including:

     * files with multi-byte UTF-8 sequences,
     * files with different newline conventions,
     * files with trailing/leading white-space,
     * extremely short files and files with very long lines.

Parallel counting is a research feature in this repo: a prototype illustrates the techniques above but is explicitly documented as experimental. Contributions that improve correctness, reduce synchronization overhead, or add robust tests are highly welcome.

---

## Build & Run

Build using the provided helper script:

```bash
# From the repo root:
chmod +x COMPILE
./COMPILE
# After running COMPILE a `wc` executable will be created in the current directory
./wc -l -w -c -m somefile.txt
```

Manual build example (if you prefer `g++` directly — adapt to your toolchain):

```bash
g++ -std=c++17 -O3 -march=native -pipe -o wc src/main.cpp src/wc_parser.cpp
```

*(If you want, I can add a `CMakeLists.txt` and `BUILD.md` with cross-platform instructions.)*

---

## Usage examples

Count lines, words, bytes, and characters:

```bash
./wc -l -w -c -m test.txt
```

Count multiple files:

```bash
./wc -l -w -c file1.txt file2.txt
```

Benchmark example (reproduce earlier measurements):

```bash
hyperfine -i "/usr/bin/wc -l -c -w -m test.txt test.txt test.txt"
hyperfine -i "./wc -l -c -w -m test.txt test.txt test.txt"
```

---

## Reproducibility & benchmarking scripts

A `bench/` script is recommended (and I can provide one) that:

* runs `hyperfine` with `--shell=none` and `--warmup`,
* exports JSON results,
* runs with different file sizes to plot scaling,
* toggles single-threaded vs parallel mode for comparative plots.

---

## Contributing

Contributions welcome, especially:

* Implementations of remaining GNU options while preserving benchmarkability.
* Improvements to the parallel counter that keep correctness across chunk boundaries.
* Tests and CI (GitHub Actions) that run unit tests and a small benchmark on push.
* Portability work (Windows/macOS), or a CMake-based build system.

Please open an issue or a PR. If a change is proposed that affects microbenchmarks, prefer small, focused commits with rationale so measurement regressions/changes are explainable.

---

## Citation & License

This repository is a research artifact. If you use `fast-wc` or its results in academic work, please cite the repository.

License: **GPL-3.0** (see `LICENSE` file).

---



