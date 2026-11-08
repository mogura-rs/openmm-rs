#include <openmm/CMAPTorsionForce.h>
#include <openmm/Context.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace OpenMM;

std::vector<double> mapValues(int size, int map, int phase) {
    std::vector<double> values;
    for (int j=0; j<size; ++j) for (int i=0; i<size; ++i) {
        double x=2*M_PI*i/size, y=2*M_PI*j/size;
        values.push_back((1+0.2*phase)*(std::sin(x)+0.3*std::cos(2*y)+0.2*std::sin(x-2*y))+0.1*map+0.05*phase);
    }
    return values;
}

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0; i<values.size(); ++i) { if (i) std::cout << ','; std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']'; }
    std::cout << ']';
}

void run(int size, bool periodic) {
    System system;
    for (int i=0; i<8; ++i) system.addParticle(1+0.2*i);
    system.setDefaultPeriodicBoxVectors(Vec3(4,0,0),Vec3(0.3,4,0),Vec3(-0.2,0.4,4));
    auto* force=new CMAPTorsionForce();
    for (int map=0; map<2; ++map) force->addMap(size+map,mapValues(size+map,map,0));
    force->addTorsion(0,0,1,2,3,4,5,6,7);
    force->addTorsion(1,0,1,2,3,1,2,3,4);
    force->setUsesPeriodicBoundaryConditions(periodic);
    force->setForceGroup(7);
    system.addForce(force);
    VerletIntegrator integrator(0.0002);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int phase=0; phase<2; ++phase) {
        if (phase) {
            for (int map=0; map<2; ++map) force->setMapParameters(map,size+map,mapValues(size+map,map,phase));
            force->setTorsionParameters(0,1,0,1,2,3,4,5,6,7);
            force->setTorsionParameters(1,0,0,1,2,3,1,2,3,4);
            force->updateParametersInContext(context);
        }
        int point=0;
        for (double phi:{-M_PI,-M_PI+1e-7,-2.3,-1.2,-1e-7,0.0,1e-7,0.4,1.1,2.3,M_PI-1e-7,M_PI}) {
            double psi=0.7-1.3*phi;
            std::vector<Vec3> positions={Vec3(0,1,0),Vec3(0,0,0),Vec3(1,0,0),Vec3(1,std::cos(phi),std::sin(phi)),Vec3(0,2.6,0.2),Vec3(0,1.6,0.2),Vec3(1,1.6,0.2),Vec3(1,1.6+std::cos(psi),0.2+std::sin(psi))};
            if (periodic) {
                positions[0]+=Vec3(4,0,0); positions[3]+=Vec3(0.3,4,0); positions[6]+=Vec3(-0.2,0.4,4);
            }
            context.setPositions(positions);
            std::vector<Vec3> velocities;
            for (int i=0; i<8; ++i) velocities.push_back(Vec3(0.01*i,-0.02,0.03-0.004*i));
            context.setVelocities(velocities);
            context.setTime(0);
            for (int steps:{0,4}) {
                if (steps) integrator.step(steps);
                auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities,false,1<<7);
                auto excluded=context.getState(State::Energy|State::Forces,false,1<<6);
                if (excluded.getPotentialEnergy()!=0) throw std::runtime_error("incorrect force group selection");
                std::cout << "{\"size\":" << size << ",\"periodic\":" << (periodic?"true":"false") << ",\"phase\":" << phase << ",\"point\":" << point << ",\"steps\":" << steps << ",\"initial_positions\":";
                vectors(positions); std::cout << ",\"initial_velocities\":"; vectors(velocities);
                std::cout << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy() << ",\"forces\":"; vectors(state.getForces());
                std::cout << ",\"positions\":"; vectors(state.getPositions()); std::cout << ",\"velocities\":"; vectors(state.getVelocities());
                std::cout << "}\n";
            }
            point++;
        }
    }
}

int main() {
    std::cout << std::setprecision(17);
    for (int size:{2,3,4,12,24}) for (bool periodic:{false,true}) run(size,periodic);
}
