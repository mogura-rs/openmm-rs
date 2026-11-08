#include <openmm/internal/QTBIntegratorUtilities.h>
#include <openmm/internal/ThreadPool.h>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace OpenMM;

void values(const std::vector<double>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

void run(double temperature, double friction, double dt, int n) {
    ThreadPool threads(1);
    std::vector<double> theta, deconvolved;
    auto start = std::chrono::steady_clock::now();
    QTBIntegratorUtilities::calculateSpectrum(temperature, friction, dt, n, theta, deconvolved, threads);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << "{\"temperature\":" << temperature << ",\"friction\":" << friction
              << ",\"dt\":" << dt << ",\"num_freq\":" << n << ",\"seconds\":" << seconds << ",\"theta\":";
    values(theta);
    std::cout << ",\"deconvolved\":";
    values(deconvolved);
    std::cout << "}\n";
}

int main() {
    std::cout << std::setprecision(17);
    for (int n : {2, 3, 5, 6, 16, 64, 128})
        for (double temperature : {0., 70., 300., 600.})
            for (double friction : {1., 10.})
                run(temperature, friction, 0.001, n);
    run(300, 10, 0.001, 512);
    run(300, 10, 0.001, 1500);
}
