#include <iostream>
#include <sycl/sycl.hpp>
#include <cmath>
#include <random>
#include <iomanip>

using _float_t = double;

_float_t getRandomVal(_float_t l, _float_t r) {
    static std::random_device rd;
    static std::mt19937 mersenne(rd());
    std::uniform_real_distribution<_float_t> distr(l, r);  // [l, r)
    return distr(mersenne);
}

void fillRandom(std::vector<_float_t>& v, const size_t N, _float_t left, _float_t right) {
    for (auto& elem : v)
        elem = getRandomVal(left, right);
    if (v.size() > N) {
        for (size_t i = 0; i < N; ++i) {
            _float_t sum = 0.0;
            for (size_t j = 0; j < N; ++j)
                if (j != i)
                    sum += fabs(v[i*N + j]);
            v[i*N + i] = getRandomVal(left + 1.1*sum, right + 1.1*sum);
        }
    }
}

_float_t normArr(_float_t* v, const size_t N) {
    _float_t sum = 0.0;
    for (size_t i = 0; i < N; i++)
        sum += pow(v[i], 2);
    return sqrt(sum);
}

_float_t normDiffArr(_float_t* v1, _float_t* v2, const size_t N) {
    _float_t sum = 0.0;
    for (size_t i = 0; i < N; i++)
        sum += pow(v1[i] - v2[i], 2);
    return sqrt(sum);
}

_float_t calcFinalAccuracy(const std::vector<_float_t>& A, const std::vector<_float_t>& x,
                                                            const std::vector<_float_t>& b) {
    const size_t N = x.size();
    std::vector<_float_t> tmp(N, 0.0);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++)
            tmp[i] += A[i*N + j] * x[j];
        tmp[i] -= b[i];
    }
    return normArr(tmp.data(), N);
}

struct Result {
    std::string memory;
    double time;
    _float_t accuracy;
    int iters;
};

std::ostream& operator<< (std::ostream& os, const Result& res) {
    os << std::left << std::setw(12) << res.memory << " Time: " << std::left << std::setw(7) << res.time
                                                                                << " ms  Accuracy: " << res.accuracy;
    return os;
}

Result runOnAccessors(sycl::queue &queue, const std::vector<_float_t> &A, const std::vector<_float_t> &b,
                                                            const _float_t MAX_ITERS, const _float_t TARGET_ACCURACY) {
    const size_t N = b.size();

    std::vector<_float_t> x_prev(N);
    for (std::size_t i = 0; i < N; i++)
        x_prev[i] = b[i] / A[i * N + i];
    std::vector<_float_t> x_cur(N);

    sycl::buffer<_float_t> buf_A(A.data(), A.size());
    sycl::buffer<_float_t> buf_b(b.data(), b.size());

    uint64_t time = 0;
    int iter = 0;
    do {
        sycl::buffer<_float_t> buf_x_prev(x_prev.data(), x_prev.size());
        sycl::buffer<_float_t> buf_x_cur(x_cur.data(), x_cur.size());

        sycl::event event = queue.submit([&](sycl::handler &cgh) {
			sycl::accessor acc_A{buf_A, cgh, sycl::read_only};
			sycl::accessor acc_b{buf_b, cgh, sycl::read_only};
			sycl::accessor acc_x_prev{buf_x_prev, cgh, sycl::read_write};
			sycl::accessor acc_x_cur{buf_x_cur, cgh, sycl::read_write};

			cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
				int i = id.get(0);
				_float_t sum = 0.0;

				for (int j = 0; j < N; j++)
					if (j != i)
                        sum += acc_A[i * N + j] * acc_x_prev[j];

				acc_x_cur[i] = (acc_b[i] - sum) / acc_A[i * N + i];
				std::swap(acc_x_prev[i], acc_x_cur[i]);
			});
        });
        queue.wait();
        uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        time += (end - start);
        iter++;
    } while (iter < MAX_ITERS
                        && (normDiffArr(x_cur.data(), x_prev.data(), N) / normArr(x_cur.data(), N)) >= TARGET_ACCURACY);

    return Result{"[Accessors]", time/1e6, calcFinalAccuracy(A, x_prev, b), iter};
}

Result runOnSharedMemory(sycl::queue &queue, const std::vector<_float_t> &A, const std::vector<_float_t> &b,
                                                            const _float_t MAX_ITERS, const _float_t TARGET_ACCURACY) {
    const size_t N = b.size();

    _float_t *sh_A = sycl::malloc_shared<_float_t>(A.size(), queue);
    _float_t *sh_b = sycl::malloc_shared<_float_t>(b.size(), queue);
    _float_t *sh_x_prev = sycl::malloc_shared<_float_t>(N, queue);
    _float_t *sh_x_cur = sycl::malloc_shared<_float_t>(N, queue);

    queue.memcpy(sh_A, A.data(), A.size() * sizeof(_float_t)).wait();
    queue.memcpy(sh_b, b.data(), b.size() * sizeof(_float_t)).wait();

    for (std::size_t i = 0; i < N; i++)
        sh_x_prev[i] = sh_b[i] / sh_A[i * N + i];

    uint64_t time = 0;
    int iter = 0;
    do {
        sycl::event event = queue.submit([&](sycl::handler &cgh) {
			cgh.parallel_for(sycl::range<1>(N), [=](sycl::item<1> item) {
				int i = item.get_id(0);
				_float_t sum = 0.0;

				for (int j = 0; j < N; j++)
					if (j != i)
                        sum += sh_A[i * N + j] * sh_x_prev[j];

				sh_x_cur[i] = (sh_b[i] - sum) / sh_A[i * N + i];
				std::swap(sh_x_prev[i], sh_x_cur[i]);
			});
        });
        queue.wait();
        uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        time += (end - start);
        iter++;
    } while(iter < MAX_ITERS && (normDiffArr(sh_x_cur, sh_x_prev, N) / normArr(sh_x_cur, N)) >= TARGET_ACCURACY);
    
    std::vector<_float_t> x_prev(sh_x_prev, sh_x_prev + N);
    sycl::free(sh_A, queue);
    sycl::free(sh_b, queue);
    sycl::free(sh_x_prev, queue);
    sycl::free(sh_x_cur, queue);

    return Result{"[Shared]", time/1e6, calcFinalAccuracy(A, x_prev, b), iter};
}

Result runOnDeviceMemory(sycl::queue &queue, const std::vector<_float_t> &A, const std::vector<_float_t> &b,
                                                            const _float_t MAX_ITERS, const _float_t TARGET_ACCURACY) {
    const size_t N = b.size();

    std::vector<_float_t> x_prev(N);
    for (std::size_t i = 0; i < N; i++)
        x_prev[i] = b[i] / A[i*N + i];
    std::vector<_float_t> x_cur(N);

    _float_t *dev_A = sycl::malloc_device<_float_t>(A.size(), queue);
    _float_t *dev_b = sycl::malloc_device<_float_t>(b.size(), queue);
    _float_t *dev_x_prev = sycl::malloc_device<_float_t>(N, queue);
    _float_t *dev_x_cur = sycl::malloc_device<_float_t>(N, queue);

    queue.memcpy(dev_A, A.data(), A.size() * sizeof(_float_t)).wait();
    queue.memcpy(dev_b, b.data(), b.size() * sizeof(_float_t)).wait();
    queue.memcpy(dev_x_prev, x_prev.data(), x_prev.size() * sizeof(_float_t)).wait();

    uint64_t time = 0;
    int iter = 0;
    do {
        sycl::event event = queue.submit([&](sycl::handler &cgh) {
			cgh.parallel_for(sycl::range<1>(N), [=](sycl::item<1> item) {
				int i = item.get_id(0);
				int n = item.get_range(0);
				_float_t sum = 0.0;

				for (int j = 0; j < n; j++)
					if (j != i)
                        sum += dev_A[i * n + j] * dev_x_prev[j];

				dev_x_cur[i] = (dev_b[i] - sum) / dev_A[i * n + i];
				std::swap(dev_x_prev[i], dev_x_cur[i]);
            });
        });
        queue.wait();
        queue.memcpy(x_prev.data(), dev_x_prev, N*sizeof(_float_t)).wait();
        queue.memcpy(x_cur.data(), dev_x_cur, N*sizeof(_float_t)).wait();
        uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        time += (end - start);
        iter++;
    } while(iter < MAX_ITERS
                        && (normDiffArr(x_cur.data(), x_prev.data(), N) / normArr(x_cur.data(), N)) >= TARGET_ACCURACY);

    sycl::free(dev_A, queue);
    sycl::free(dev_b, queue);
    sycl::free(dev_x_prev, queue);
    sycl::free(dev_x_cur, queue);

    return Result{"[Device]", time/1e6, calcFinalAccuracy(A, x_prev, b), iter};
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

    std::vector<_float_t> A(N*N);
    fillRandom(A, N, -100, 100);
    std::vector<_float_t> b(N);
    fillRandom(b, N, -100, 100);

    sycl::queue queue(my_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}});
    std::cout << "Using " << queue.get_device().get_info<sycl::info::device::name>() << "\n\n";

    runOnAccessors(queue, A, b, MAX_ITERS, TARGET_ACCURACY);
    std::cout << runOnAccessors(queue, A, b, MAX_ITERS, TARGET_ACCURACY) << "\n";

    runOnSharedMemory(queue, A, b, MAX_ITERS, TARGET_ACCURACY);
    std::cout << runOnSharedMemory(queue, A, b, MAX_ITERS, TARGET_ACCURACY) << "\n";

    runOnDeviceMemory(queue, A, b, MAX_ITERS, TARGET_ACCURACY);
    std::cout << runOnDeviceMemory(queue, A, b, MAX_ITERS, TARGET_ACCURACY) << "\n";

    return 0;
}
