#include <iostream>
#include <sycl/sycl.hpp>
#include <cmath>

int main(int argc, char* argv[]) {
    // sin(x)cos(y)dxdy, x from 0 to 1, y from 0 to 1 ~= 0.3868223
    constexpr double EXP_VAL = 0.3868223;

    constexpr std::size_t GROUP_SIZE = 16;
    const std::size_t intervals = std::stoi(argv[1]);
    const double dx = 1.0 / intervals;
    const double dy = 1.0 / intervals;

    const std::string deviceStr(argv[2]);
    auto my_selector_v = [&](sycl::device device) {
        if (deviceStr == "cpu")
            return device.is_cpu() ? 1 : -1;
        if (deviceStr == "gpu")
            return device.is_gpu() ? 1 : -1;
        else
            return (device == sycl::device(sycl::default_selector_v)) ? 1 : -1;
    };

    sycl::queue queue(my_selector_v);
    std::cout << "Using " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<2>(  // global size / local size = number of work-groups (on Nvidia must be int)
                        sycl::range<2>(intervals, intervals),  // global range, a number of work-items
                        sycl::range<2>(GROUP_SIZE, GROUP_SIZE)),  // local range, a number of work-items in a work-group
        [=](sycl::nd_item<2> item){
            double x = dx * (item.get_global_id(0) + 0.5);
            double y = dy * (item.get_global_id(1) + 0.5);
            double val = sycl::sin(x)*sycl::cos(y)*dx*dy;
            double sum = sycl::reduce_over_group(item.get_group(), val, std::plus<double>());
        });
    });

    return 0;
}