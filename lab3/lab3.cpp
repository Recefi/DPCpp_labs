#include <iostream>
#include <sycl/sycl.hpp>
#include <cmath>

using _float_t = double;

int fillRandVect(_float_t v, size_t size) {

}

int fillRandMatr(_float_t m, size_t size) {

}

int main(int argc, char* argv[]) {
    const size_t N = std::stoi(argv[1]);
    const _float_t TARGET_ACCURACY = std::stod(argv[2]);
    const size_t MAX_ITERS = std::stoi(argv[3]);


    const std::string deviceStr = argv[4];
    auto my_selector_v = [&](sycl::device device) {
        if (deviceStr == "cpu")
            return device.is_cpu() ? 1 : -1;
        if (deviceStr == "gpu")
            return device.is_gpu() ? 1 : -1;
        else
            return (device == sycl::device(sycl::default_selector_v)) ? 1 : -1;
    };


    return 0;
}
