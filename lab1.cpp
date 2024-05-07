#include <iostream>
#include <sycl/sycl.hpp>

int main() {
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    for (size_t i = 0; i < platforms.size(); i++) {
        std::cout << "Platform #" << i << ": " << platforms[i].get_info<sycl::info::platform::name>() << std::endl;
        std::vector<sycl::device> devices = platforms[i].get_devices();
        for (size_t j = 0; j < devices.size(); j++) {  
            std::cout << "--- Device #" << j << ": " << devices[j].get_info<sycl::info::device::name>() << std::endl;
        }
    }

    std::cout << "\n";

    for (size_t i = 0; i < platforms.size(); i++) {
        std::vector<sycl::device> devices = platforms[i].get_devices();
        for (size_t j = 0; j < devices.size(); j++) {  
            std::cout << devices[j].get_info<sycl::info::device::name>() << std::endl;
            sycl::queue queue(devices[j]);
            queue.submit([&](sycl::handler &cgh) {
                sycl::stream s(1024, 80, cgh);  // bufferSize, maxStatementSize
                cgh.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> id) {
                    s << "[" << id.get(0) << "] Hello from platform " << i << " and device " << j << sycl::endl;
                });
            }).wait();  // for device output immediately under its name
        }
    }

    return 0;
}
