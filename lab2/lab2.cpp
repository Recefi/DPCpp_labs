#include <iostream>
#include <sycl/sycl.hpp>
#include <cmath>

using _float_t = double;

int main(int argc, char* argv[]) {
    // sin(x)cos(y)dxdy, x from 0 to 1, y from 0 to 1 ~= 0.3868223
    constexpr _float_t EXP_RES = 0.3868223;

    constexpr size_t GROUP_SIZE = 16;
    const size_t intervalsCount = std::stoi(argv[1]);
    std::cout << "Iteration space: " << intervalsCount << " x " << intervalsCount << ", "
                                            << GROUP_SIZE << " x " << GROUP_SIZE << "\n";
    const _float_t dx = 1.0 / intervalsCount;
    const _float_t dy = 1.0 / intervalsCount;
    const size_t groupsCount = intervalsCount / GROUP_SIZE + 1;

    const std::string deviceStr = argv[2];
    auto my_selector_v = [&](sycl::device device) {
        if (deviceStr == "cpu")
            return device.is_cpu() ? 1 : -1;
        if (deviceStr == "gpu")
            return device.is_gpu() ? 1 : -1;
        else
            return (device == sycl::device(sycl::default_selector_v)) ? 1 : -1;
    };

    sycl::queue queue(my_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}});
    std::cout << "Using " << queue.get_device().get_info<sycl::info::device::name>() << "\n\n";

    std::vector<_float_t> res(groupsCount*groupsCount, 0.0);
    {
        sycl::buffer<_float_t> buffer(res.data(), res.size());
        sycl::event event = queue.submit([&](sycl::handler &cgh) {
            sycl::accessor accessor{buffer, cgh, sycl::write_only};
            //sycl::stream s(4096, 80, cgh);
            cgh.parallel_for(sycl::nd_range<2>(  // global size/local size=number of work-groups (on Nvidia must be int)
                        sycl::range<2>(intervalsCount, intervalsCount),  // global range, a number of work-items
                        sycl::range<2>(GROUP_SIZE, GROUP_SIZE)),  // local range, a number of work-items in a work-group
            [=](sycl::nd_item<2> item){
                _float_t x = dx * (item.get_global_id(0) + 0.5);
                _float_t y = dy * (item.get_global_id(1) + 0.5);
                _float_t itemVal = sycl::sin(x)*sycl::cos(y)*dx*dy;
                //s << item.get_global_id(0) << " " << item.get_global_id(1) << ": " << sycl::sin(x)*sycl::cos(y) << " * " << dx*dy << " = "  << sycl::sin(x)*sycl::cos(y)*dx*dy << " " << itemVal << sycl::endl;
                _float_t groupSum = sycl::reduce_over_group(item.get_group(), itemVal, std::plus<_float_t>());
                if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
                    accessor[item.get_group(0) * item.get_group_range(0) + item.get_group(1)] = groupSum;
                }
            });
        });
        event.wait();

        uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        std::cout << "time: " << end - start << " ns\n";
    }

    _float_t calc_res = 0;
    for (size_t i = 0; i < res.size(); i++)
        calc_res += res[i];
    
    std::cout << "expected result: " << EXP_RES << "\n";
    std::cout << "calculated result: " << calc_res << "\n";
    std::cout << "error: " << std::abs(EXP_RES - calc_res) << "\n";

    return 0;
}
