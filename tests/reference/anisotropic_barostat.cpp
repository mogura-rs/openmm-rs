#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/CustomVolumeForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/MonteCarloAnisotropicBarostat.h>
#include <openmm/NonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0;i<values.size();++i) {
        if (i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

Vec3 cartesianPressure(Context& context, const System& system, bool rigid, int groups) {
    const auto state=context.getState(State::Positions|State::Velocities);
    const auto original=state.getPositions();
    Vec3 box[3];state.getPeriodicBoxVectors(box[0],box[1],box[2]);
    const double volume=box[0][0]*box[1][1]*box[2][2];
    Vec3 kinetic;
    if (rigid) {
        for (const auto& molecule:context.getMolecules()) {
            Vec3 momentum;double mass=0;
            for (int atom:molecule) {mass+=system.getParticleMass(atom);momentum+=state.getVelocities()[atom]*system.getParticleMass(atom);}
            if (mass>0) for (int axis=0;axis<3;++axis) kinetic[axis]+=0.5*momentum[axis]*momentum[axis]/mass;
        }
    } else {
        for (int atom=0;atom<system.getNumParticles();++atom)
            for (int axis=0;axis<3;++axis) kinetic[axis]+=0.5*system.getParticleMass(atom)*std::pow(state.getVelocities()[atom][axis],2);
    }
    Vec3 pressure;
    for (int axis=0;axis<3;++axis) {
        double energies[2];
        for (int sign=0;sign<2;++sign) {
            const double factor=sign==0?1.001:0.999;
            std::vector<Vec3> positions=original;
            if (rigid) {
                for (const auto& molecule:context.getMolecules()) {
                    double center=0;
                    for (int atom:molecule) center+=original[atom][axis];
                    center/=molecule.size();
                    for (int atom:molecule) positions[atom][axis]+=(factor-1)*center;
                }
            } else for (auto& position:positions) position[axis]*=factor;
            Vec3 scaled[3]={box[0],box[1],box[2]};
            for (auto& vector:scaled) vector[axis]*=factor;
            context.setPeriodicBoxVectors(scaled[0],scaled[1],scaled[2]);
            context.setPositions(positions);
            energies[sign]=context.getState(State::Energy,false,groups).getPotentialEnergy();
        }
        pressure[axis]=(2*kinetic[axis]/volume-(energies[0]-energies[1])/(0.002*volume))/0.0602214076;
    }
    context.setPeriodicBoxVectors(box[0],box[1],box[2]);context.setPositions(original);
    return pressure;
}

void pressure(bool rigid, int variant, int mask, bool triclinic) {
    System system;
    for (double mass : {1.0,3.0,2.0,0.0,0.0,4.0}) system.addParticle(mass);
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(triclinic?0.4:0,3.3,0),Vec3(triclinic?-0.2:0,triclinic?0.3:0,3.5));
    auto* bond = new HarmonicBondForce();
    bond->addBond(0,1,0.55,17);
    bond->addBond(2,5,0.7,23);
    bond->setUsesPeriodicBoundaryConditions(true);
    bond->setForceGroup(1);
    system.addForce(bond);
    auto* external = new CustomExternalForce("k*(x*x+0.7*y*y+0.3*z*z)");
    external->addGlobalParameter("k",0.4);
    for (int i=0;i<6;++i) external->addParticle(i);
    external->setForceGroup(2);
    system.addForce(external);
    auto* volume = new CustomVolumeForce("0.03*v+0.07*ax*by");
    volume->setForceGroup(5);
    system.addForce(volume);
    if (variant>=1) system.addConstraint(0,1,0.7);
    if (variant==2) {
        system.setVirtualSite(4,new TwoParticleAverageSite(0,1,0.3,0.7));
        auto* nb=new NonbondedForce();
        nb->setNonbondedMethod(NonbondedForce::PME);
        nb->setCutoffDistance(0.9);
        nb->setPMEParameters(3.2,16,18,20);
        for (int i=0;i<6;++i) nb->addParticle((i%2 ? -1 : 1)*0.08,0.2,0.01);
        nb->addException(0,1,0,0.2,0);
        nb->setForceGroup(3);
        nb->setReciprocalSpaceForceGroup(4);
        system.addForce(nb);
    }
    auto* barostat=new MonteCarloAnisotropicBarostat(Vec3(1.3,1.7,2.1),300,true,true,true,0,rigid);
    barostat->setRandomNumberSeed(17);
    barostat->setForceGroup(31);
    system.addForce(barostat);
    VerletIntegrator integrator(0.00002);
    integrator.setConstraintTolerance(1e-10);
    integrator.setIntegrationForceGroups(mask);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int phase=0;phase<3;++phase) {
        double s=1+phase*0.1;
        context.setPeriodicBoxVectors(Vec3(3,0,0)*s,Vec3(triclinic?0.4:0,3.3,0)*s,Vec3(triclinic?-0.2:0,triclinic?0.3:0,3.5)*s);
        context.setPositions({{0.2,0.3,0.4},{0.9,0.3,0.4},{1.3,1.1,0.7},{0.5,1.7,1.3},{1.4,0.8,1.5},{1.9,1.2,0.8}});
        context.setVelocities({{0.3,-0.2,0.1},{-0.1,0.4,0.2},{0.2,0.1,-0.3},{0.7,0.8,0.9},{-0.3,0.6,0.5},{0.1,-0.3,0.4}});
        context.computeVirtualSites();
        context.applyVelocityConstraints(1e-10);
        context.setTime(4.7);
        context.setStepCount(19);
        context.setParameter("k",0.4+phase*0.1);
        context.setParameter(MonteCarloAnisotropicBarostat::PressureX(),2+phase);
        context.setParameter(MonteCarloAnisotropicBarostat::PressureY(),3+phase);
        context.setParameter(MonteCarloAnisotropicBarostat::PressureZ(),4+phase);
        context.setParameter(MonteCarloAnisotropicBarostat::Temperature(),260+10*phase);
        barostat->setDefaultPressure(Vec3(11+phase,12+phase,13+phase));
        barostat->setDefaultTemperature(410+phase);
        int done=0;
        for (int steps : {0,1,4}) {
            integrator.step(steps-done);done=steps;
            auto before=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy);
            Vec3 source=barostat->computeCurrentPressure(context);
            Vec3 value=cartesianPressure(context,system,rigid,mask);
            if (!triclinic) for (int axis=0;axis<3;++axis)
                if (std::abs(source[axis]-value[axis])>1e-8) throw std::runtime_error("orthorhombic pressure mismatch");
            auto after=context.getState(State::Positions|State::Velocities);
            if (before.getPositions()!=after.getPositions() || before.getVelocities()!=after.getVelocities() || before.getTime()!=after.getTime())
                throw std::runtime_error("pressure changed state");
            std::cout << "{\"case\":\"pressure\",\"rigid\":" << (rigid?"true":"false")
                << ",\"triclinic\":" << (triclinic?"true":"false") << ",\"variant\":" << variant << ",\"mask\":" << mask << ",\"phase\":" << phase
                << ",\"steps\":" << steps << ",\"pressure\":[" << value[0] << "," << value[1] << "," << value[2] << "]"
                << ",\"source_pressure\":[" << source[0] << "," << source[1] << "," << source[2] << "]"
                << ",\"energy\":" << before.getPotentialEnergy() << ",\"kinetic\":" << before.getKineticEnergy()
                << ",\"positions\":"; vectors(before.getPositions());
            std::cout << ",\"velocities\":"; vectors(before.getVelocities());
            std::cout << ",\"forces\":"; vectors(before.getForces());
            std::cout << "}\n";
        }
    }
}

void schedule(bool rigid, int axes) {
    System system;
    system.addParticle(1);system.addParticle(3);
    auto* bond=new HarmonicBondForce();bond->addBond(0,1,0.7,0);
    bond->setUsesPeriodicBoundaryConditions(true);system.addForce(bond);
    auto* barostat=new MonteCarloAnisotropicBarostat(Vec3(0,0,0),0,axes&1,axes&2,axes&4,0,rigid);
    barostat->setRandomNumberSeed(73);system.addForce(barostat);
    VerletIntegrator integrator(0.001);integrator.setIntegrationForceGroups(0);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions({{0.2,0.3,0.4},{0.9,0.3,0.4}});
    for (int step=0;step<30;++step) {
        if (step==5) barostat->setFrequency(4);
        if (step==13) barostat->setFrequency(0);
        if (step==19) barostat->setFrequency(3);
        context.setStepCount(step%3);
        auto before=context.getState(State::Positions);
        Vec3 old[3],next[3];context.getState(0).getPeriodicBoxVectors(old[0],old[1],old[2]);
        integrator.step(1);
        context.getState(0).getPeriodicBoxVectors(next[0],next[1],next[2]);
        bool changed=false;
        for (int axis=0;axis<3;++axis) changed|=old[axis][axis]!=next[axis][axis];
        std::cout << "{\"case\":\"schedule\",\"rigid\":" << (rigid?"true":"false")
                  << ",\"axes\":" << axes << ",\"step\":" << step << ",\"changed\":" << (changed?"true":"false") << "}\n";
    }
}

void idealGas(bool rigid, int axis) {
    const int count=128, blocks=40, samples=250, stride=20;
    const double kt=1.380649e-23*6.02214076e23/1000*300;
    const double pressure=20*0.0602214076+0.2;
    const int scaled=rigid?count/2:count;
    double length=std::cbrt((scaled+1)*kt/pressure);
    System system;
    std::vector<Vec3> positions;
    for (int i=0;i<count;++i) {system.addParticle(1);positions.push_back(Vec3(0.1+0.02*i,0.2,0.3));}
    system.setDefaultPeriodicBoxVectors(Vec3(length,0,0),Vec3(0,length,0),Vec3(0,0,length));
    auto* bond=new HarmonicBondForce();
    for (int i=0;i<count;i+=2) bond->addBond(i,i+1,0.02,0);
    bond->setUsesPeriodicBoundaryConditions(true);system.addForce(bond);
    auto* bias=new CustomVolumeForce("q*v");bias->addGlobalParameter("q",0.2);system.addForce(bias);
    auto* barostat=new MonteCarloAnisotropicBarostat(Vec3(5,9,11),150,axis==0,axis==1,axis==2,1,rigid);barostat->setRandomNumberSeed(91);system.addForce(barostat);
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));context.setPositions(positions);
    context.setParameter(MonteCarloAnisotropicBarostat::PressureX(),20);
    context.setParameter(MonteCarloAnisotropicBarostat::PressureY(),20);
    context.setParameter(MonteCarloAnisotropicBarostat::PressureZ(),20);
    context.setParameter(MonteCarloAnisotropicBarostat::Temperature(),300);
    barostat->setDefaultPressure(Vec3(99,99,99));barostat->setDefaultTemperature(50);
    integrator.step(10000);
    std::cout << "{\"case\":\"ideal\",\"rigid\":" << (rigid?"true":"false") << ",\"axis\":" << axis << ",\"blocks\":[";
    for (int block=0;block<blocks;++block) {
        double sum=0,sum2=0;
        for (int j=0;j<samples;++j) {
            integrator.step(stride);
            Vec3 a,b,c;context.getState(0).getPeriodicBoxVectors(a,b,c);double v=a[0]*b[1]*c[2];sum+=v;sum2+=v*v;
        }
        if (block) std::cout << ',';
        std::cout << '[' << sum/samples << ',' << sum2/samples << ']';
    }
    std::cout << "]}\n";
}

int main() {
    std::cout << std::setprecision(17);
    for (bool rigid:{false,true}) {
        for (bool triclinic:{false,true}) for (int variant=0;variant<3;++variant) for (int mask:{-1,2,8,16,32,0}) pressure(rigid,variant,mask,triclinic);
        for (int axes=0;axes<8;++axes) schedule(rigid,axes);
        for (int axis=0;axis<3;++axis) idealGas(rigid,axis);
    }
}
