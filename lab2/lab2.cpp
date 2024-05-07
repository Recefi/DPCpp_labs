#include <iostream>
#include <sycl/sycl.hpp>
#include <cmath>

int main(int argc, char* argv[]) {
    std::size_t iteration_count = std::stoi(argv[1]);
    std::string str_device(argv[2]);

    auto cpu_gpu_selector = [=](sycl::device device) {
        if (str_device == "cpu")
            return device.is_cpu() ? 1 : -1;
        if (str_device == "gpu")
            return device.is_gpu() ? 1 : -1;
        else
            return 0;
    };

    sycl::queue queue(cpu_gpu_selector);
    std::cout << "Using " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;



    // Compute the first n_items values in a well known sequence
    constexpr int n_items = 16;
    int *items = sycl::malloc_shared<int>(n_items, queue);
    queue.parallel_for(sycl::range<1>(n_items), [items] (sycl::id<1> i) {
        double x1 = std::pow((1.0 + sqrt(5.0))/2, i);
        double x2 = std::pow((1.0 - sqrt(5.0))/2, i);
        items[i] = round((x1 - x2)/sqrt(5));
    }).wait();

    for(int i = 0 ; i < n_items ; ++i) {
        std::cout << items[i] << std::endl;
    }
    free(items, queue);

    return 0;
}