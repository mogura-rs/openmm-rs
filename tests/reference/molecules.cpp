#include <openmm/Context.h>
#include <openmm/CustomBondForce.h>
#include <openmm/HarmonicAngleForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/NonbondedForce.h>
#include <openmm/PeriodicTorsionForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <iomanip>
#include <iostream>

using namespace OpenMM;

int main() {
    std::cout << std::setprecision(17);
    for (double skew : {0.0,0.4,-0.4}) {
        System system;
        for (int i=0;i<12;++i) system.addParticle(i==3 ? 0 : 1);
        system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(skew,3,0),Vec3(-skew,skew,3));
        auto* bond = new HarmonicBondForce();
        bond->addBond(0,1,0.5,1);
        system.addForce(bond);
        auto* angle = new HarmonicAngleForce();
        angle->addAngle(1,2,4,1,1);
        system.addForce(angle);
        system.addConstraint(5,6,0.5);
        auto* torsion = new PeriodicTorsionForce();
        torsion->addTorsion(6,7,8,9,2,0,1);
        system.addForce(torsion);
        auto* custom = new CustomBondForce("r");
        custom->addBond(10,11,{});
        system.addForce(custom);
        auto* nonbonded = new NonbondedForce();
        for (int i=0;i<12;++i) nonbonded->addParticle(0,1,0);
        nonbonded->addException(4,5,0,1,0);
        system.addForce(nonbonded);
        system.setVirtualSite(3,new TwoParticleAverageSite(1,2,0.3,0.7));
        VerletIntegrator integrator(0.001);
        Context context(system,integrator,Platform::getPlatformByName("Reference"));
        std::vector<Vec3> positions;
        for (int i=0;i<12;++i) positions.emplace_back(-4.2+0.17*i,6.4+0.23*i,-1.5+0.11*i);
        context.setPositions(positions);
        context.computeVirtualSites();
        auto molecules = context.getMolecules();
        auto wrapped = context.getState(State::Positions,true).getPositions();
        auto raw = context.getState(State::Positions).getPositions();
        std::cout << "{\"skew\":" << skew << ",\"molecules\":[";
        for (int i=0;i<molecules.size();++i) {
            if(i) std::cout << ',';
            std::cout << '[';
            for(int j=0;j<molecules[i].size();++j) { if(j) std::cout << ','; std::cout << molecules[i][j]; }
            std::cout << ']';
        }
        std::cout << "]";
        for (auto entry : {std::make_pair("wrapped",wrapped),std::make_pair("raw",raw)}) {
            std::cout << ",\"" << entry.first << "\":[";
            for(int i=0;i<12;++i) { if(i) std::cout << ','; std::cout << '[' << entry.second[i][0] << ',' << entry.second[i][1] << ',' << entry.second[i][2] << ']'; }
            std::cout << ']';
        }
        std::cout << "}\n";
    }
}
