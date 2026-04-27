#include "concat.cuh"

// Contiguous concat kernel templated on element type T and concat dimension d.
// Grid must be (num_blocks_ne0, ne1_dst, ne2_dst); the caller loops over ne3.
// ne_split is the src0 size along d (i.e. src0->ne[d]).
template<typename T, int d>
static __global__ void concat_dim(const T * x, const T * y, T * dst, const int ne0, const int ne_split) {
    static_assert(d >= 0 && d <= 2, "d must be 0, 1, or 2");

    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if constexpr (d == 0) {
        if (nidx < ne_split) {
            int offset_src = nidx + blockIdx.y * ne_split + blockIdx.z * ne_split * gridDim.y;
            dst[offset_dst] = x[offset_src];
        } else {
            int ne1_src1 = ne0 - ne_split;
            int offset_src = (nidx - ne_split) + blockIdx.y * ne1_src1 + blockIdx.z * ne1_src1 * gridDim.y;
            dst[offset_dst] = y[offset_src];
        }
    } else if constexpr (d == 1) {
        if (blockIdx.y < (unsigned)ne_split) {
            int offset_src = nidx + blockIdx.y * ne0 + blockIdx.z * ne0 * ne_split;
            dst[offset_dst] = x[offset_src];
        } else {
            int ne1_src1 = gridDim.y - ne_split;
            int offset_src = nidx + (blockIdx.y - ne_split) * ne0 + blockIdx.z * ne0 * ne1_src1;
            dst[offset_dst] = y[offset_src];
        }
    } else { // d == 2
        if (blockIdx.z < (unsigned)ne_split) {
            int offset_src = nidx + blockIdx.y * ne0 + blockIdx.z * ne0 * gridDim.y;
            dst[offset_dst] = x[offset_src];
        } else {
            int ne2_src1 = gridDim.z - ne_split;
            int offset_src = nidx + blockIdx.y * ne0 + (blockIdx.z - ne_split) * ne0 * gridDim.y;
            dst[offset_dst] = y[offset_src];
        }
    }
}

// non-contiguous kernel (slow), templated on element type
template<typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        const T * x;

        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const T *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const T *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const T *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);

    // dim == 3: each source is one contiguous block -> simple memcpy
    if (dim == 3) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data,           src0->data, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *)dst->data + ggml_nbytes(src0), src1->data, ggml_nbytes(src1), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        // For contiguous tensors we use the contiguous element-level kernels which
        // correctly interleave slices from src0 and src1 along the concat dimension.
        // We iterate over the outermost dimension (ne3) in a host loop so the 3D
        // kernel grid only needs to cover (ne0, ne1, ne2) for each i3 slice.
        int num_blocks = (dst->ne[0] + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
        auto launch_cont = [&](auto dim_tag) {
            constexpr int d = std::decay_t<decltype(dim_tag)>::value;
            dim3 gridDim(num_blocks, dst->ne[1], dst->ne[2]);
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                auto src0_i3 = (const char *)src0->data + i3 * src0->nb[3];
                auto src1_i3 = (const char *)src1->data + i3 * src1->nb[3];
                auto  dst_i3 = (      char *) dst->data + i3 *  dst->nb[3];
                if (src0->type == GGML_TYPE_F32) {
                    concat_dim<float, d><<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                        (const float *)src0_i3, (const float *)src1_i3, (float *)dst_i3,
                        (int)dst->ne[0], (int)src0->ne[d]);
                } else if (src0->type == GGML_TYPE_F16) {
                    concat_dim<half, d><<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                        (const half *)src0_i3, (const half *)src1_i3, (half *)dst_i3,
                        (int)dst->ne[0], (int)src0->ne[d]);
                } else {
                    GGML_ABORT("ggml_cuda_op_concat: unsupported type %d for contiguous concat", src0->type);
                }
            }
        };
        switch (dim) {
            case 0: launch_cont(std::integral_constant<int, 0>{}); break;
            case 1: launch_cont(std::integral_constant<int, 1>{}); break;
            case 2: launch_cont(std::integral_constant<int, 2>{}); break;
            default: GGML_ABORT("Invalid dim: %d", dim);
        }
    } else {
        // Non-contiguous: use element-level kernels for float types only
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim_tag) {
            constexpr int d = std::decay_t<decltype(dim_tag)>::value;
            if (src0->type == GGML_TYPE_F32) {
                concat_non_cont<float, d><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            } else if (src0->type == GGML_TYPE_F16) {
                concat_non_cont<half, d><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            } else {
                GGML_ABORT("ggml_cuda_op_concat: unsupported type %d for non-contiguous concat", src0->type);
            }
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}
