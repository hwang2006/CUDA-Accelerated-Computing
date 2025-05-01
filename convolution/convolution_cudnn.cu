#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
//#include <cudnn.h>
#include <cudnn_v8.h>

// Helper function to check CUDA and cuDNN errors
inline void checkCUDNN(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkCUDA(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // 1. Initialize cuDNN
    cudnnHandle_t cudnnHandle;
    checkCUDNN(cudnnCreate(&cudnnHandle));

    // 2. Define Tensor Dimensions
    int n = 1;      // Batch size
    int c = 3;      // Input channels (e.g., RGB)
    int h = 32;     // Input height
    int w = 32;     // Input width
    int k = 16;     // Output channels
    int kh = 3;     // Kernel height
    int kw = 3;     // Kernel width
    int pad_h = 1;  // Padding height
    int pad_w = 1;  // Padding width
    int stride_h = 1; // Stride height
    int stride_w = 1; // Stride width
    int dilation_h = 1; // Dilation height
    int dilation_w = 1; // Dilation width

    // 3. Create Tensor Descriptors
    cudnnTensorDescriptor_t inputDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    checkCUDNN(cudnnSetTensorDescriptor(inputDesc, CUDNN_FORMAT_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    cudnnFilterDescriptor_t filterDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnSetFilterDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_FORMAT_NCHW, k, c, kh, kw));

    cudnnTensorDescriptor_t outputDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&outputDesc));

    // 4. Create Convolution Descriptor
    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CONVOLUTION));

    // 5. Determine Output Tensor Dimensions
    int out_n, out_c, out_h, out_w;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));
    checkCUDNN(cudnnSetTensorDescriptor(outputDesc, CUDNN_FORMAT_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

    std::cout << "Output Tensor Dimensions: " << out_n << "x" << out_c << "x" << out_h << "x" << out_w << std::endl;

    // 6. Allocate Device Memory
    float *d_input;
    float *d_filter;
    float *d_output;
    size_t inputSize;
    size_t filterSize;
    size_t outputSize;

    checkCUDNN(cudnnGetTensorSizeInBytes(inputDesc, &inputSize));
    checkCUDA(cudaMalloc((void**)&d_input, inputSize));

    checkCUDNN(cudnnGetFilterSizeInBytes(filterDesc, &filterSize));
    checkCUDA(cudaMalloc((void**)&d_filter, filterSize));

    checkCUDNN(cudnnGetTensorSizeInBytes(outputDesc, &outputSize));
    checkCUDA(cudaMalloc((void**)&d_output, outputSize));

    // 7. Initialize Input and Filter Data (Example)
    std::vector<float> h_input(n * c * h * w);
    std::vector<float> h_filter(k * c * kh * kw);
    for (size_t i = 0; i < h_input.size(); ++i) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < h_filter.size(); ++i) h_filter[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f; // Small weights

    checkCUDA(cudaMemcpy(d_input, h_input.data(), inputSize, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(d_filter, h_filter.data(), filterSize, cudaMemcpyHostToDevice));

    // 8. Choose Convolution Algorithm
    cudnnConvolutionFwdAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo
    ));

    size_t workspaceSize;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        algo,
        &workspaceSize
    ));

    void *d_workspace = nullptr;
    if (workspaceSize > 0) {
        checkCUDA(cudaMalloc(&d_workspace, workspaceSize));
    }

    // 9. Perform the Convolution
    float alpha = 1.0f;
    float beta = 0.0f;

    checkCUDNN(cudnnConvolutionForward(
        cudnnHandle,
        &alpha,
        inputDesc,
        d_input,
        filterDesc,
        d_filter,
        convDesc,
        algo,
        d_workspace,
        workspaceSize,
        &beta,
        outputDesc,
        d_output
    ));

    // 10. Copy Output Back to Host (Optional for Verification)
    std::vector<float> h_output(out_n * out_c * out_h * out_w);
    checkCUDA(cudaMemcpy(h_output.data(), d_output, outputSize, cudaMemcpyDeviceToHost));

    std::cout << "Convolution Finished. First few output elements: " << std::endl;
    for (int i = 0; i < std::min((int)h_output.size(), 10); ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // 11. Clean Up
    checkCUDA(cudaFree(d_input));
    checkCUDA(cudaFree(d_filter));
    checkCUDA(cudaFree(d_output));
    if (d_workspace != nullptr) {
        checkCUDA(cudaFree(d_workspace));
    }
    checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN(cudnnDestroy(cudnnHandle));

    return 0;
}
