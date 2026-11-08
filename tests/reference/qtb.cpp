#include <OpenMM.h>
#include <ReferencePlatform.h>
#include <openmm/internal/QTBIntegratorUtilities.h>
#include <openmm/internal/ThreadPool.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace OpenMM;
class SerialReference : public ReferencePlatform {
public:
    void contextCreated(ContextImpl& context, const std::map<std::string, std::string>&) const override {
        ReferencePlatform::contextCreated(context, {{"Threads", "1"}});
    }
};

void scalars(const std::vector<double>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

void deterministic(int length, int types, int constrained, int site, int routing) {
    System system;
    for (double mass : {1., 1., 2., 2., 0., 3.}) system.addParticle(mass);
    if (site) {
        system.addParticle(0);
        system.setVirtualSite(6, new TwoParticleAverageSite(0, 3, 0.3, 0.7));
    }
    if (constrained) system.addConstraint(0, 1, 0.37);
    auto* bond = new HarmonicBondForce();
    bond->addBond(0, 1, 0.37, 1.2);
    bond->setForceGroup(1);
    system.addForce(bond);
    auto* external = new CustomExternalForce("lambda*(0.1*x^2+0.03*y-0.05*z)");
    external->addGlobalParameter("lambda", 0.7);
    for (int i = 0; i < system.getNumParticles(); ++i) external->addParticle(i, {});
    external->setForceGroup(3);
    system.addForce(external);
    auto* qtb = new QTBIntegrator(0, 1.3, 0.003);
    qtb->setSegmentLength((length-0.25)*0.003);
    qtb->setCutoffFrequency(1e-5);
    qtb->setDefaultAdaptationRate(0);
    qtb->setConstraintTolerance(1e-11);
    qtb->setRandomNumberSeed(191);
    if (types) {
        qtb->setParticleType(0, -7);
        qtb->setParticleType(1, -7);
        qtb->setParticleType(2, 12345);
        qtb->setParticleType(3, 12345);
    }
    std::unique_ptr<Integrator> top(qtb);
    if (routing) {
        auto* root = new CompoundIntegrator();
        auto* branch = new CompoundIntegrator();
        branch->addIntegrator(top.release());
        root->addIntegrator(branch);
        root->addIntegrator(new VerletIntegrator(0.001));
        root->setCurrentIntegrator(routing == 2 ? 1 : 0);
        top.reset(root);
    }
    SerialReference platform;
    Context context(system, *top, platform);
    std::vector<Vec3> p, v;
    for (int i = 0; i < system.getNumParticles(); ++i) {
        p.emplace_back(0.37*i, 0.2*(i/2), -0.1*(i/2));
        v.emplace_back(0.1*i-0.2, 0.06*i-0.1, 0.03*i-0.04);
    }
    context.setPositions(p);
    context.computeVirtualSites();
    context.setVelocities(v);
    context.applyVelocityConstraints(1e-11);
    context.setTime(1.7);
    context.setStepCount(11);
    for (int phase = 0; phase < 6; ++phase) {
        if (phase == 1) {
            qtb->setStepSize(0.006);
            qtb->setFriction(7);
            qtb->setSegmentLength(0.03);
            qtb->setCutoffFrequency(500);
            qtb->setDefaultAdaptationRate(999);
            qtb->setTypeAdaptationRate(-7, 999);
            qtb->setParticleType(5, 99);
        }
        if (phase == 2) { qtb->setIntegrationForceGroups(1<<1); context.setParameter("lambda", 1.4); }
        if (phase == 3) { qtb->setStepSize(0.003); context.setVelocitiesToTemperature(0, 17); }
        if (phase == 4) qtb->setIntegrationForceGroups((1<<1)|(1<<3));
        for (int sample = 0; sample < 3; ++sample) {
            double before = context.getState(0).getTime();
            if (routing == 1) top->step(1); else qtb->step(1);
            double recorded = context.getState(0).getTime();
            // Repair only the reference clock; physical integration uses captured dt.
            context.setTime(before+0.003);
            State state = context.getState(State::Positions|State::Velocities|State::Forces|State::Energy);
            std::vector<double> gamma;
            qtb->getAdaptedFriction(0, gamma);
            std::cout << "{\"kind\":\"deterministic\",\"length\":" << length << ",\"types\":" << types
                      << ",\"constrained\":" << constrained << ",\"site\":" << site << ",\"routing\":" << routing
                      << ",\"phase\":" << phase << ",\"sample\":" << sample << ",\"time\":" << state.getTime()
                      << ",\"source_time\":" << recorded << ",\"step_count\":" << state.getStepCount()
                      << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy()
                      << ",\"positions\":";
            vectors(state.getPositions()); std::cout << ",\"velocities\":"; vectors(state.getVelocities());
            std::cout << ",\"forces\":"; vectors(state.getForces());
            std::cout << ",\"gamma\":"; scalars(gamma); std::cout << "}\n";
        }
    }
}

void spectrum(int length, double temperature, double friction, double cutoff) {
    int frequencies = 3*length/2+1;
    double dt = 0.001;
    // The approved FFT-grid correction uses 2*pi*k/(3*length*dt).
    double referenceDt = 3*length*dt/(2*frequencies);
    ThreadPool threads(1);
    std::vector<double> theta, deconvolved;
    QTBIntegratorUtilities::calculateSpectrum(temperature, friction, referenceDt, frequencies, theta, deconvolved, threads);
    std::cout << "{\"kind\":\"noise_spectrum\",\"length\":" << length << ",\"temperature\":" << temperature
              << ",\"friction\":" << friction << ",\"cutoff\":" << cutoff << ",\"dt\":" << dt << ",\"theta\":";
    scalars(theta); std::cout << ",\"deconvolved\":"; scalars(deconvolved); std::cout << "}\n";
}

void oscillator(double temperature, double omega) {
    const int particles = 32, count = 10000;
    System system;
    auto* force = new CustomExternalForce("0.5*k*(x*x+y*y+z*z)");
    force->addGlobalParameter("k", omega*omega);
    for (int i = 0; i < particles; ++i) { system.addParticle(1); force->addParticle(i, {}); }
    system.addForce(force);
    QTBIntegrator integrator(temperature, 10, 0.001);
    integrator.setSegmentLength(0.50075);
    integrator.setRandomNumberSeed(769);
    for (int i = 0; i < particles; ++i) integrator.setParticleType(i, 0);
    SerialReference platform;
    Context context(system, integrator, platform);
    context.setPositions(std::vector<Vec3>(particles));
    context.setVelocities(std::vector<Vec3>(particles));
    integrator.step(10000);
    double mean = 0, m2 = 0, kinetic = 0;
    for (int i = 0; i < count; ++i) {
        integrator.step(10);
        State state = context.getState(State::Energy);
        double value = state.getPotentialEnergy();
        double delta = value-mean;
        mean += delta/(i+1);
        m2 += delta*(value-mean);
        kinetic += state.getKineticEnergy()/count;
    }
    std::cout << "{\"kind\":\"oscillator\",\"temperature\":" << temperature << ",\"omega\":" << omega
              << ",\"count\":" << count << ",\"mean_potential\":" << mean << ",\"variance_potential\":" << m2/count
              << ",\"mean_raw_kinetic\":" << kinetic << ",\"time\":" << context.getState(0).getTime() << "}\n";
}

void sourceProbes() {
    for (int length : {2, 3, 4, 5}) {
        System system;
        system.addParticle(1);
        QTBIntegrator integrator(0, 1, 0.001);
        integrator.setSegmentLength((length-0.25)*0.001);
        integrator.setCutoffFrequency(1e-5);
        integrator.setDefaultAdaptationRate(0);
        integrator.setRandomNumberSeed(77);
        SerialReference platform;
        Context context(system, integrator, platform);
        context.setPositions({Vec3()});
        context.setVelocities({Vec3()});
        double maximum = 0;
        for (int i = 0; i < 12; ++i) {
            integrator.step(1);
            Vec3 v = context.getState(State::Velocities).getVelocities()[0];
            maximum = std::max(maximum, std::sqrt(v.dot(v)));
        }
        std::cout << "{\"kind\":\"source_unfiltered_nyquist\",\"length\":" << length
                  << ",\"maximum_speed\":" << maximum << "}\n";
    }
    System system;
    system.addParticle(2);
    auto* force = new CustomExternalForce("-3*x");
    force->addParticle(0, {});
    system.addForce(force);
    QTBIntegrator integrator(0, 1, 0.001);
    integrator.setSegmentLength(0.00275);
    integrator.setCutoffFrequency(1e-5);
    SerialReference platform;
    Context context(system, integrator, platform);
    context.setPositions({Vec3()});
    integrator.setStepSize(0.002);
    context.setVelocitiesToTemperature(0, 17);
    std::cout << "{\"kind\":\"source_thermal_offset\",\"velocity_x\":"
              << context.getState(State::Velocities).getVelocities()[0][0] << "}\n";
}

int main() {
    std::cout << std::setprecision(17);
    sourceProbes();
    for (int length : {3, 5}) for (int types : {0, 1}) for (int constrained : {0, 1})
        for (int site : {0, 1}) for (int routing : {0, 1, 2}) deterministic(length, types, constrained, site, routing);
    for (double temperature : {70., 300.}) for (double omega : {20., 100.}) oscillator(temperature, omega);
    for (int length : {2, 3, 8, 15}) {
        spectrum(length, 0, 10, 500);
        spectrum(length, 300, 1, 500);
    }
}
