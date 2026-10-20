// Controlled CUDA IPC dependency probe. This is a standalone diagnostic, not
// NCCL instrumentation. Each rank passes GPU-resident tables of mapped IPC
// addresses; the launcher performs no allocation or host-to-device copy.
//
// A slab has slots = repeats * rounds * world. For slot
// ((repeat * rounds + round) * world + logical_source), it contains
// payload_words protocol words followed by an int32 generation flag in a
// separate flag slab. The caller zeros every slab before each launch.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint64_t global_timer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void store_release_sys(int32_t* address, int32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(value), "l"(address) : "memory");
#else
  __threadfence_system();
  atomicExch_system(address, value);
#endif
}

__device__ __forceinline__ int32_t load_acquire_sys(const int32_t* address) {
  int32_t value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
#else
  value = atomicAdd_system(const_cast<int32_t*>(address), 0);
  __threadfence_system();
#endif
  return value;
}

template <typename Word>
__device__ __forceinline__ void copy_words(Word* destination, const Word* source, int count) {
  for (int i = threadIdx.x; i < count; i += blockDim.x) destination[i] = source[i];
}

template <typename Word>
__global__ void dependency_probe_kernel(const int64_t* data_ptrs, const int64_t* flag_ptrs,
                                        int rank, int world, int words, int rounds, int repeats,
                                        int dependent, int64_t* phases, Word* final_output) {
  if (blockIdx.x != 0) return;
  const size_t payload_stride = static_cast<size_t>(words) * sizeof(Word);
  const int phase_stride = 3 + 2 * rounds;

  for (int repeat = 0; repeat < repeats; ++repeat) {
    __syncthreads();
    const int phase_base = repeat * phase_stride;
    if (threadIdx.x == 0) phases[phase_base] = static_cast<int64_t>(global_timer());
    __syncthreads();

    // Independent mode publishes all rounds before waiting. Dependent mode
    // waits after every round, then forwards the received logical lane.
    for (int round = 0; round < rounds; ++round) {
      const int destination = (rank + 1) % world;
      const int logical_lane = dependent ? ((rank - round + world * rounds) % world) : rank;
      const size_t slot = static_cast<size_t>(repeat * rounds + round) * world;
      const size_t previous_slot = static_cast<size_t>(repeat * rounds + (round == 0 ? 0 : round - 1)) * world;
      auto* own_base = reinterpret_cast<uint8_t*>(data_ptrs[rank]);
      auto* destination_base = reinterpret_cast<uint8_t*>(data_ptrs[destination]);
      const Word* input = nullptr;
      if (dependent && round > 0) {
        input = reinterpret_cast<const Word*>(own_base + previous_slot * payload_stride +
                                              logical_lane * words * sizeof(Word));
      } else {
        input = reinterpret_cast<const Word*>(own_base + slot * payload_stride +
                                              rank * words * sizeof(Word));
      }
      auto* output = reinterpret_cast<Word*>(destination_base + slot * payload_stride +
                                             logical_lane * words * sizeof(Word));
      copy_words(output, input, words);
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        auto* destination_flags = reinterpret_cast<int32_t*>(flag_ptrs[destination]);
        store_release_sys(destination_flags + slot + logical_lane, round + 1);
        phases[phase_base + 1 + 2 * round] = static_cast<int64_t>(global_timer());
      }
      __syncthreads();

      if (dependent) {
        const int incoming_lane = (rank - round - 1 + world * rounds) % world;
        auto* own_flags = reinterpret_cast<int32_t*>(flag_ptrs[rank]);
        while (load_acquire_sys(own_flags + slot + incoming_lane) < round + 1) {}
        __syncthreads();
        if (threadIdx.x == 0) phases[phase_base + 2 + 2 * round] = static_cast<int64_t>(global_timer());
      }
    }

    if (!dependent) {
      auto* own_flags = reinterpret_cast<int32_t*>(flag_ptrs[rank]);
      const int incoming_lane = (rank - 1 + world) % world;
      for (int round = 0; round < rounds; ++round) {
        const size_t slot = static_cast<size_t>(repeat * rounds + round) * world;
        while (load_acquire_sys(own_flags + slot + incoming_lane) < round + 1) {}
        __syncthreads();
        if (threadIdx.x == 0) phases[phase_base + 2 + 2 * round] = static_cast<int64_t>(global_timer());
      }
    }

    const int final_lane = dependent ? ((rank - rounds + world * rounds) % world)
                                     : ((rank - 1 + world) % world);
    const size_t final_slot = static_cast<size_t>(repeat * rounds + rounds - 1) * world;
    const auto* final_input = reinterpret_cast<const Word*>(
        reinterpret_cast<const uint8_t*>(data_ptrs[rank]) + final_slot * payload_stride +
        final_lane * words * sizeof(Word));
    if (repeat == repeats - 1) copy_words(final_output, final_input, words);
    __syncthreads();
    if (threadIdx.x == 0) phases[phase_base + phase_stride - 1] = static_cast<int64_t>(global_timer());
    __syncthreads();
  }
}

template <typename Word>
__global__ void initialize_probe_kernel(const int64_t* data_ptrs, int rank, int world,
                                        int words, int rounds, int repeats) {
  if (blockIdx.x != 0) return;
  const size_t payload_stride = static_cast<size_t>(words) * sizeof(Word);
  auto* base = reinterpret_cast<uint8_t*>(data_ptrs[rank]);
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (int round = 0; round < rounds; ++round) {
      const size_t slot = static_cast<size_t>(repeat * rounds + round) * world;
      auto* local = reinterpret_cast<Word*>(base + slot * payload_stride + rank * words * sizeof(Word));
      for (int i = threadIdx.x; i < words; i += blockDim.x) {
        local[i] = static_cast<Word>(rank * 1000003 + round * 1009 + i);
      }
    }
  }
}

__global__ void clear_probe_flags_kernel(const int64_t* flag_ptrs, int rank, int slots, int world) {
  if (blockIdx.x != 0) return;
  auto* flags = reinterpret_cast<int32_t*>(flag_ptrs[rank]);
  // `slots` already includes the world dimension: it is
  // repeats * rounds * world, matching the flat lane allocation.
  const int count = slots;
  for (int i = threadIdx.x; i < count; i += blockDim.x) flags[i] = 0;
}

void validate_table(const torch::Tensor& table, int world, const char* name) {
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kLong && table.is_contiguous(),
              name, " must be a contiguous CUDA int64 tensor");
  TORCH_CHECK(table.numel() == world, name, " must contain exactly world pointers");
}

// Seed every local-source lane before the caller's inter-process barrier.
// This is separate from probe_launch so rank skew cannot race initialization
// against a remote forwarding write.
void probe_initialize(torch::Tensor data_ptrs, int rank, int world, int payload_words,
                      int rounds, int repeats, int protocol, int threads) {
  TORCH_CHECK(world >= 2 && world <= 8, "world must be in [2, 8]");
  TORCH_CHECK(rank >= 0 && rank < world, "invalid rank");
  TORCH_CHECK(payload_words > 0 && rounds > 0 && repeats > 0, "invalid dimensions");
  TORCH_CHECK(protocol == 0 || protocol == 1, "protocol must be 0 (bf16-sized) or 1 (int32)");
  TORCH_CHECK(threads == 64 || threads == 256, "threads must be 64 or 256");
  validate_table(data_ptrs, world, "data_ptrs");
  auto stream = at::cuda::getCurrentCUDAStream(data_ptrs.device().index()).stream();
  if (protocol == 0) {
    initialize_probe_kernel<uint16_t><<<1, threads, 0, stream>>>(
        data_ptrs.data_ptr<int64_t>(), rank, world, payload_words, rounds, repeats);
  } else {
    initialize_probe_kernel<int32_t><<<1, threads, 0, stream>>>(
        data_ptrs.data_ptr<int64_t>(), rank, world, payload_words, rounds, repeats);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void probe_clear_flags(torch::Tensor flag_ptrs, int rank, int world, int slots, int threads) {
  TORCH_CHECK(world >= 2 && world <= 8, "world must be in [2, 8]");
  TORCH_CHECK(rank >= 0 && rank < world, "invalid rank");
  TORCH_CHECK(slots > 0, "slots must be positive");
  TORCH_CHECK(threads == 64 || threads == 256, "threads must be 64 or 256");
  validate_table(flag_ptrs, world, "flag_ptrs");
  auto stream = at::cuda::getCurrentCUDAStream(flag_ptrs.device().index()).stream();
  clear_probe_flags_kernel<<<1, threads, 0, stream>>>(flag_ptrs.data_ptr<int64_t>(), rank, slots, world);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// mode=0 publishes all rounds from initially ready local inputs, then waits
// for every incoming edge. mode=1 waits after each round and forwards the
// previous received payload. Pointer tables are GPU tensors; timestamps are
// rank-local globaltimer values and are never subtracted across devices.
void probe_launch(torch::Tensor data_ptrs, torch::Tensor flag_ptrs, int rank, int world,
                  int payload_words, int rounds, int repeats, int mode, int protocol,
                  int threads, torch::Tensor phases, torch::Tensor final_output) {
  TORCH_CHECK(world >= 2 && world <= 8, "world must be in [2, 8]");
  TORCH_CHECK(rank >= 0 && rank < world, "invalid rank");
  TORCH_CHECK(payload_words > 0 && rounds > 0 && repeats > 0, "invalid dimensions");
  TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (independent) or 1 (dependent)");
  TORCH_CHECK(protocol == 0 || protocol == 1, "protocol must be 0 (bf16-sized) or 1 (int32)");
  TORCH_CHECK(threads == 64 || threads == 256, "threads must be 64 or 256");
  validate_table(data_ptrs, world, "data_ptrs");
  validate_table(flag_ptrs, world, "flag_ptrs");
  TORCH_CHECK(phases.is_cuda() && phases.scalar_type() == at::kLong && phases.is_contiguous() &&
                  phases.numel() >= repeats * (3 + 2 * rounds),
              "phases must be a contiguous CUDA int64 tensor");
  TORCH_CHECK(final_output.is_cuda() && final_output.is_contiguous() && final_output.numel() >= payload_words,
              "final_output must be a contiguous CUDA tensor with payload_words elements");
  auto stream = at::cuda::getCurrentCUDAStream(final_output.device().index()).stream();
  if (protocol == 0) {
    TORCH_CHECK(final_output.scalar_type() == at::kShort, "BF16-sized output must use int16 storage");
    dependency_probe_kernel<uint16_t><<<1, threads, 0, stream>>>(
        data_ptrs.data_ptr<int64_t>(), flag_ptrs.data_ptr<int64_t>(), rank, world, payload_words,
        rounds, repeats, mode, phases.data_ptr<int64_t>(),
        reinterpret_cast<uint16_t*>(final_output.data_ptr<int16_t>()));
  } else {
    TORCH_CHECK(final_output.scalar_type() == at::kInt, "int32 output must use int32 storage");
    dependency_probe_kernel<int32_t><<<1, threads, 0, stream>>>(
        data_ptrs.data_ptr<int64_t>(), flag_ptrs.data_ptr<int64_t>(), rank, world, payload_words,
        rounds, repeats, mode, phases.data_ptr<int64_t>(), final_output.data_ptr<int32_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_launch", &probe_launch, "CUDA IPC causal dependency probe");
  m.def("probe_initialize", &probe_initialize, "Initialize local probe payload slots");
  m.def("probe_clear_flags", &probe_clear_flags, "Clear local probe generation flags");
}
