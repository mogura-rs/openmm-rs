#include <openmm/Context.h>
#include <openmm/CustomCentroidBondForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/TabulatedFunction.h>
#include <openmm/VerletIntegrator.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0; i<values.size(); ++i) { if (i) std::cout << ','; std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']'; }
    std::cout << ']';
}
std::vector<double> tableValues(int phase) {
    std::vector<double> result;
    for (int i=0;i<9;++i) result.push_back((1+0.2*phase)*std::sin(i-4.0));
    return result;
}

std::vector<double> groupWeights(int mode, int group, int size) {
    if (mode==0 || (mode==2 && group%2==0)) return {};
    if (mode==2 && group==1) return {2.0,-0.25};
    if (mode==2 && group==3) return {0.5,1.0,0.0};
    std::vector<double> weights;
    for (int i=0;i<size;++i) weights.push_back(1+0.2*group+0.3*i);
    return weights;
}
void connectivity() {
    System system;
    for (int i=0;i<9;++i) system.addParticle(1);
    auto* force=new CustomCentroidBondForce(2,"distance(g1,g2)^2");
    for (int i=0;i<4;++i) force->addGroup({2*i,2*i+1});
    force->addBond({2,3},{}); system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    const auto& molecules=context.getMolecules();
    std::cout << "{\"record\":\"connectivity\",\"molecules\":[";
    for (size_t i=0;i<molecules.size();++i) {
        if (i) std::cout << ','; std::cout << '[';
        for (size_t j=0;j<molecules[i].size();++j) { if (j) std::cout << ','; std::cout << molecules[i][j]; }
        std::cout << ']';
    }
    std::cout << "]}\n";
}
void run(int mode, int expression, bool periodic) {
    std::vector<std::string> expressions={"distance(g1,g2)^2","angle(g1,g2,g3)","sin(dihedral(g1,g2,g3,g4))","f(distance(g1,g3))","distance(g1,g4)*angle(g2,g3,g4)+x1*z2","pointdistance(x1,y1,z1,x3+0.1,y3,z3)"};
    std::string energy="g*k*("+expressions[expression]+")+0.02*x1*z4";
    System system;
    for (int i=0;i<8;++i) system.addParticle(mode==2 && i==7 ? 0 : 1+0.2*i);
    system.setDefaultPeriodicBoxVectors(Vec3(4,0,0),Vec3(0.3,4.2,0),Vec3(-0.2,0.4,4.4));
    auto* force=new CustomCentroidBondForce(4,energy);
    std::vector<std::vector<int>> groups={{0,1},{2,3},{1,4,5},{5,6,7},{0,2,6}};
    for (int i=0;i<groups.size();++i) force->addGroup(groups[i],groupWeights(mode,i,groups[i].size()));
    force->addGlobalParameter("g",1.3); force->addEnergyParameterDerivative("g"); force->addPerBondParameter("k");
    force->addBond({2,0,4,3},{0.7}); force->addBond({4,1,2,0},{0.9});
    force->setUsesPeriodicBoundaryConditions(periodic);
    auto* table=new Continuous1DFunction(tableValues(0),-4,4); force->addTabulatedFunction("f",table);
    force->setForceGroup(9); system.addForce(force);
    VerletIntegrator integrator(0.0001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int phase=0;phase<3;++phase) {
        if (phase==1) context.setParameter("g",0.8);
        if (phase==2) {
            force->setBondParameters(0,{2,0,4,3},{1.2}); force->setBondParameters(1,{4,1,2,0},{0.4});
            force->setGroupParameters(0,{7,0},{3,1});
            table->setFunctionParameters(tableValues(2),-4,4); force->updateParametersInContext(context);
        }
        double scale=1+0.1*phase;
        context.setPeriodicBoxVectors(Vec3(4*scale,0,0),Vec3(0.3,4.2*scale,0),Vec3(-0.2,0.4,4.4*scale));
        for (int point=0;point<4;++point) {
            double phi=-2.7+1.7*point;
            std::vector<Vec3> positions={Vec3(0,1,0),Vec3(0,0,0),Vec3(1,0,0),Vec3(1,std::cos(phi),std::sin(phi)),Vec3(0.3,1.7,0.2),Vec3(1.2,1.6,0.4),Vec3(1.7,2.3,1.1),Vec3(2,2,2)};
            if (periodic) { positions[0]+=Vec3(4,0,0); positions[3]+=Vec3(0.3,4.2,0); positions[6]+=Vec3(-0.2,0.4,4.4); }
            std::vector<Vec3> velocities;
            for (int i=0;i<8;++i) velocities.push_back(Vec3(0.01*i,-0.02,0.03-0.004*i));
            context.setPositions(positions); context.setVelocities(velocities); context.setTime(0);
            for (int steps:{0,3}) {
                if (steps) integrator.step(steps);
                auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities|State::ParameterDerivatives,false,1<<9);
                std::cout << "{\"weights\":" << mode << ",\"expression\":" << expression << ",\"energy_function\":\"" << energy << "\",\"periodic\":" << (periodic?"true":"false") << ",\"phase\":" << phase << ",\"point\":" << point << ",\"steps\":" << steps << ",\"initial_positions\":";
                vectors(positions); std::cout << ",\"initial_velocities\":"; vectors(velocities);
                std::cout << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy() << ",\"parameter_derivative\":" << state.getEnergyParameterDerivatives().at("g") << ",\"forces\":"; vectors(state.getForces());
                std::cout << ",\"positions\":"; vectors(state.getPositions()); std::cout << ",\"velocities\":"; vectors(state.getVelocities()); std::cout << "}\n";
            }
        }
    }
}
int main() {
    std::cout << std::setprecision(17);
    connectivity();
    for (int mode=0;mode<3;++mode) for (int expression=0;expression<6;++expression) for (bool periodic:{false,true}) run(mode,expression,periodic);
}
