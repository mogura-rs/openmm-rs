#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/LocalEnergyMinimizer.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace OpenMM;

void run(const std::string& name) {
    System system;
    for (int i=0;i<3;++i) system.addParticle((name=="fixed" && i==0) || (name=="virtual" && i==2) ? 0 : 12+2*i);
    auto* external = new CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)");
    for (auto parameter : {"k","x0","y0","z0"}) external->addPerParticleParameter(parameter);
    external->addParticle(0,{10,0,0,0});
    external->addParticle(1,{15,0.8,0,0});
    external->addParticle(2,{20,0.2,0.6,0});
    external->setForceGroup(1);
    system.addForce(external);
    auto* bonds = new HarmonicBondForce();
    bonds->addBond(0,1,0.8,100);
    if (name!="virtual") bonds->addBond(1,2,std::sqrt(0.72),40);
    bonds->setForceGroup(1);
    system.addForce(bonds);
    if (name=="constrained") system.addConstraint(0,1,0.65);
    if (name=="virtual") system.setVirtualSite(2,new TwoParticleAverageSite(0,1,0.5,0.5));
    if (name=="groups") {
        auto* ignored = new CustomExternalForce("1000*(x-3)^2");
        ignored->addParticle(0,{});
        ignored->setForceGroup(4);
        system.addForce(ignored);
    }
    VerletIntegrator integrator(0.001);
    integrator.setIntegrationForceGroups(2);
    integrator.setConstraintTolerance(1e-6);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3(0.2,-0.3,0.1),Vec3(1.4,0.4,-0.2),Vec3(0.6,1.1,0.3)});
    LocalEnergyMinimizer::minimize(context,1e-4,name=="limited" ? 1 : 0);
    auto state = context.getState(State::Positions|State::Energy|State::Forces,false,2);
    std::cout << "{\"case\":\"" << name << "\",\"energy\":" << state.getPotentialEnergy() << ",\"positions\":[";
    for (int i=0;i<3;++i) {
        if(i) std::cout << ',';
        const auto& p=state.getPositions()[i];
        std::cout << '[' << p[0] << ',' << p[1] << ',' << p[2] << ']';
    }
    std::cout << "]}\n";
}

int main() {
    std::cout << std::setprecision(17);
    for (auto name : {"free","fixed","constrained","virtual","groups","limited"}) run(name);
}
