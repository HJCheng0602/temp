// Small cached-plan bridge for the production BF16 wo_a layout.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace {
void ck(cublasStatus_t s, const char* what) {
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, what, " failed: ", static_cast<int>(s));
}
struct Plan {
  cublasLtHandle_t h{}; cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t ad{}, bd{}, cd{};
  cublasLtMatmulHeuristicResult_t algo{}; torch::Tensor ws;
  Plan(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& out, int index) {
    const int64_t g=a.size(0), m=a.size(1), k=a.size(2), n=b.size(2);
    ck(cublasLtCreate(&h), "cublasLtCreate");
    ck(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    cublasOperation_t no=CUBLAS_OP_N;
    ck(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &no, sizeof(no)), "transa");
    ck(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &no, sizeof(no)), "transb");
    int64_t ald=a.stride(2)==1?a.stride(1):a.stride(2), bld=b.stride(2)==1?b.stride(1):b.stride(2);
    ck(cublasLtMatrixLayoutCreate(&ad, CUDA_R_16BF, m, k, ald), "layout a");
    ck(cublasLtMatrixLayoutCreate(&bd, CUDA_R_16BF, k, n, bld), "layout b");
    ck(cublasLtMatrixLayoutCreate(&cd, CUDA_R_16BF, m, n, out.stride(1)), "layout c");
    cublasLtOrder_t ao=a.stride(2)==1?CUBLASLT_ORDER_ROW:CUBLASLT_ORDER_COL;
    cublasLtOrder_t bo=b.stride(2)==1?CUBLASLT_ORDER_ROW:CUBLASLT_ORDER_COL;
    cublasLtOrder_t ro=CUBLASLT_ORDER_ROW;
    ck(cublasLtMatrixLayoutSetAttribute(ad,CUBLASLT_MATRIX_LAYOUT_ORDER,&ao,sizeof(ao)),"order a");
    ck(cublasLtMatrixLayoutSetAttribute(bd,CUBLASLT_MATRIX_LAYOUT_ORDER,&bo,sizeof(bo)),"order b");
    ck(cublasLtMatrixLayoutSetAttribute(cd,CUBLASLT_MATRIX_LAYOUT_ORDER,&ro,sizeof(ro)),"order c");
    int64_t as=a.stride(0), bs=b.stride(0), cs=out.stride(0); int32_t batch=g;
    for (auto x : {std::pair<cublasLtMatrixLayout_t,int64_t>{ad,as}, {bd,bs}, {cd,cs}})
      ck(cublasLtMatrixLayoutSetAttribute(x.first,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&x.second,sizeof(x.second)),"stride");
    ck(cublasLtMatrixLayoutSetAttribute(ad,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)),"batch a");
    ck(cublasLtMatrixLayoutSetAttribute(bd,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)),"batch b");
    ck(cublasLtMatrixLayoutSetAttribute(cd,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)),"batch c");
    cublasLtMatmulPreference_t pref{}; ck(cublasLtMatmulPreferenceCreate(&pref),"preference");
    uint64_t bytes=64ull*1024ull*1024ull;
    ck(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)),"workspace");
    cublasLtMatmulHeuristicResult_t r[8]{}; int returned=0;
    ck(cublasLtMatmulAlgoGetHeuristic(h,op,ad,bd,cd,cd,pref,8,r,&returned),"heuristic");
    TORCH_CHECK(index >= 0 && index < returned, "cuBLASLt heuristic index unavailable: ", index);
    algo=r[index]; ws=torch::empty({static_cast<long>(algo.workspaceSize)},a.options().dtype(at::kByte));
    cublasLtMatmulPreferenceDestroy(pref);
  }
  ~Plan(){ if(cd)cublasLtMatrixLayoutDestroy(cd); if(bd)cublasLtMatrixLayoutDestroy(bd); if(ad)cublasLtMatrixLayoutDestroy(ad); if(op)cublasLtMatmulDescDestroy(op); if(h)cublasLtDestroy(h); }
};
std::mutex mu; std::unordered_map<std::string,std::shared_ptr<Plan>> cache;
std::string key(const torch::Tensor&a,const torch::Tensor&b,const torch::Tensor&c,int i){
  std::string result=std::to_string(a.device().index())+":"+std::to_string(i);
  for (const auto& tensor : {a,b,c}) {
    for (int dim=0;dim<3;++dim) result+=":"+std::to_string(tensor.size(dim))+":"+std::to_string(tensor.stride(dim));
  }
  return result;
}
torch::Tensor bmm_out(torch::Tensor a, torch::Tensor b, torch::Tensor out, int index){
  TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&a.scalar_type()==at::kBFloat16&&b.scalar_type()==at::kBFloat16,"expected CUDA BF16");
  TORCH_CHECK(a.dim()==3&&b.dim()==3&&a.size(0)==b.size(0)&&a.size(2)==b.size(1),"expected [G,M,K]@[G,K,N]");
  TORCH_CHECK(out.is_cuda()&&out.scalar_type()==at::kBFloat16&&out.dim()==3&&out.size(0)==a.size(0)&&out.size(1)==a.size(1)&&out.size(2)==b.size(2),"invalid output");
  TORCH_CHECK(a.device()==b.device()&&a.device()==out.device(),"operands on different devices");
  TORCH_CHECK(out.stride(2)==1,"output must be row major");
  auto kx=key(a,b,out,index); std::shared_ptr<Plan> p;
  {std::lock_guard<std::mutex> l(mu); auto it=cache.find(kx);
    if(it==cache.end()) {
      cudaStreamCaptureStatus status;
      TORCH_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(a.device().index()).stream(),&status)==cudaSuccess,"capture status failed");
      TORCH_CHECK(status==cudaStreamCaptureStatusNone,"plan not warmed before CUDA graph capture");
      it=cache.emplace(kx,std::make_shared<Plan>(a,b,out,index)).first;
    }
    p=it->second;
  }
  float alpha=1,beta=0; auto s=at::cuda::getCurrentCUDAStream(a.device().index()).stream();
  ck(cublasLtMatmul(p->h,p->op,&alpha,a.data_ptr(),p->ad,b.data_ptr(),p->bd,&beta,out.data_ptr(),p->cd,out.data_ptr(),p->cd,&p->algo.algo,p->ws.data_ptr(),p->ws.numel(),s),"matmul"); return out;
}
torch::Tensor bmm(torch::Tensor a, torch::Tensor b, int index){
  auto out=torch::empty({a.size(0),a.size(1),b.size(2)},a.options()); return bmm_out(a,b,out,index);
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("bmm",&bmm,"cached BF16 cuBLASLt bmm"); m.def("bmm_out",&bmm_out,"cached BF16 cuBLASLt bmm into strided output");}
