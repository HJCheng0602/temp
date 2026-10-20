// Minimal row/column-major, strided-batch BF16 cuBLASLt bridge.
// Plans are cached so CUDA Graph capture only launches the already-created op.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
void check_cuda(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + " failed, cublas status=" + std::to_string(status));
}

struct Plan {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  torch::Tensor workspace;
  uint64_t workspace_bytes = 64ull * 1024ull * 1024ull;

  Plan(const torch::Tensor& a, const torch::Tensor& b, int64_t algorithm_index) {
    const auto m = a.size(1), k = a.size(2), n = b.size(2), g = a.size(0);
    try {
      check_cuda(cublasLtCreate(&handle), "cublasLtCreate");
      check_cuda(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "matmulDescCreate");
      cublasOperation_t trans = CUBLAS_OP_N;
      check_cuda(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)), "set TRANSA");
      check_cuda(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)), "set TRANSB");
      int64_t ald = a.stride(2) == 1 ? a.stride(1) : a.stride(2);
      int64_t bld = b.stride(2) == 1 ? b.stride(1) : b.stride(2);
      check_cuda(cublasLtMatrixLayoutCreate(&ad, CUDA_R_16BF, m, k, ald), "layout A");
      check_cuda(cublasLtMatrixLayoutCreate(&bd, CUDA_R_16BF, k, n, bld), "layout B");
      auto out = torch::empty({g, m, n}, a.options());
      check_cuda(cublasLtMatrixLayoutCreate(&cd, CUDA_R_16BF, m, n, out.stride(1)), "layout C");
      cublasLtOrder_t a_order = a.stride(2) == 1 ? CUBLASLT_ORDER_ROW : CUBLASLT_ORDER_COL;
      cublasLtOrder_t b_order = b.stride(2) == 1 ? CUBLASLT_ORDER_ROW : CUBLASLT_ORDER_COL;
      cublasLtOrder_t c_order = CUBLASLT_ORDER_ROW;
      check_cuda(cublasLtMatrixLayoutSetAttribute(ad, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_order, sizeof(a_order)), "order A");
      check_cuda(cublasLtMatrixLayoutSetAttribute(bd, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)), "order B");
      check_cuda(cublasLtMatrixLayoutSetAttribute(cd, CUBLASLT_MATRIX_LAYOUT_ORDER, &c_order, sizeof(c_order)), "order C");
      int64_t astride = a.stride(0), bstride = b.stride(0), cstride = out.stride(0);
      check_cuda(cublasLtMatrixLayoutSetAttribute(ad, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &astride, sizeof(astride)), "stride A");
      check_cuda(cublasLtMatrixLayoutSetAttribute(bd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &bstride, sizeof(bstride)), "stride B");
      check_cuda(cublasLtMatrixLayoutSetAttribute(cd, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &cstride, sizeof(cstride)), "stride C");
      int32_t batch = static_cast<int32_t>(g);
      check_cuda(cublasLtMatrixLayoutSetAttribute(ad, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)), "batch A");
      check_cuda(cublasLtMatrixLayoutSetAttribute(bd, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)), "batch B");
      check_cuda(cublasLtMatrixLayoutSetAttribute(cd, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)), "batch C");
      check_cuda(cublasLtMatmulPreferenceCreate(&pref), "preferenceCreate");
      check_cuda(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "workspace limit");
      cublasLtMatmulHeuristicResult_t results[32];
      int returned = 0;
      check_cuda(cublasLtMatmulAlgoGetHeuristic(handle, op, ad, bd, cd, cd, pref, 32, results, &returned), "get heuristic");
      TORCH_CHECK(algorithm_index < returned, "heuristic algorithm index ", algorithm_index, " unavailable (returned ", returned, ")");
      heuristic = results[algorithm_index];
      workspace = torch::empty({static_cast<long>(workspace_bytes)}, a.options().dtype(at::kByte));
    } catch (...) {
      if (pref) cublasLtMatmulPreferenceDestroy(pref);
      if (ad) cublasLtMatrixLayoutDestroy(ad);
      if (bd) cublasLtMatrixLayoutDestroy(bd);
      if (cd) cublasLtMatrixLayoutDestroy(cd);
      if (op) cublasLtMatmulDescDestroy(op);
      if (handle) cublasLtDestroy(handle);
      throw;
    }
  }
  ~Plan() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (ad) cublasLtMatrixLayoutDestroy(ad);
    if (bd) cublasLtMatrixLayoutDestroy(bd);
    if (cd) cublasLtMatrixLayoutDestroy(cd);
    if (op) cublasLtMatmulDescDestroy(op);
    if (handle) cublasLtDestroy(handle);
  }
};

std::mutex plans_mutex;
std::unordered_map<std::string, std::shared_ptr<Plan>> plans;

std::string key_for(const torch::Tensor& a, const torch::Tensor& b, int64_t algo) {
  return std::to_string(a.device().index()) + ":" + std::to_string(a.size(0)) + ":" +
      std::to_string(a.size(1)) + ":" + std::to_string(a.size(2)) + ":" +
      std::to_string(b.size(2)) + ":" + std::to_string(a.stride(0)) + ":" +
      std::to_string(a.stride(1)) + ":" + std::to_string(a.stride(2)) + ":" +
      std::to_string(b.stride(0)) + ":" + std::to_string(b.stride(1)) + ":" +
      std::to_string(b.stride(2)) + ":" + std::to_string(algo);
}

torch::Tensor bmm(torch::Tensor a, torch::Tensor b, int64_t algorithm_index) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "cuBLASLt inputs must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16, "inputs must be BF16");
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0), "expected [G,M,K] and [G,K,N]");
  TORCH_CHECK(a.size(2) == b.size(1), "K mismatch");
  TORCH_CHECK((a.stride(2) == 1 || a.stride(1) == 1) && (b.stride(2) == 1 || b.stride(1) == 1), "unsupported matrix strides");
  TORCH_CHECK(algorithm_index >= 0, "algorithm index must be non-negative");
  auto key = key_for(a, b, algorithm_index);
  std::shared_ptr<Plan> plan;
  {
    std::lock_guard<std::mutex> lock(plans_mutex);
    auto it = plans.find(key);
    if (it == plans.end()) it = plans.emplace(key, std::make_shared<Plan>(a, b, algorithm_index)).first;
    plan = it->second;
  }
  auto out = torch::empty({a.size(0), a.size(1), b.size(2)}, a.options());
  const float alpha = 1.0f, beta = 0.0f;
  auto stream = at::cuda::getCurrentCUDAStream(a.device().index()).stream();
  check_cuda(cublasLtMatmul(plan->handle, plan->op, &alpha, a.data_ptr(), plan->ad, b.data_ptr(), plan->bd,
                            &beta, out.data_ptr(), plan->cd, out.data_ptr(), plan->cd, &plan->heuristic.algo,
                            plan->workspace.data_ptr(), plan->workspace_bytes, stream), "matmul");
  return out;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bmm", &bmm, "BF16 strided-batch row/column-major cuBLASLt GEMM");
}
