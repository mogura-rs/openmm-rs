#include <openmm/DPDIntegrator.h>
#include <openmm/CompoundIntegrator.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VirtualSite.h>
#include <iomanip>
#include <cmath>
#include <iostream>
using namespace OpenMM;
void vectors(const std::vector<Vec3>& v) {
 std::cout<<'[';for(size_t i=0;i<v.size();++i) {if(i)std::cout<<',';std::cout<<'['<<v[i][0]<<','<<v[i][1]<<','<<v[i][2]<<']';}std::cout<<']';
}
void run(int box, int types, int constrained, int site, int routing) {
 int n=types?64:27;System s;for(int i=0;i<n;++i)s.addParticle(1+i%4);
 if(site) {s.addParticle(0);s.setVirtualSite(n,new TwoParticleAverageSite(0,26,0.3,0.7));}
 if(constrained)s.addConstraint(0,1,0.37);
 Vec3 a(2.8,0,0),b(box==2?0.7:0,2.6,0),c(box==2?-0.6:0,box==2?0.55:0,2.4);
 s.setDefaultPeriodicBoxVectors(a,b,c);
 auto* bond=new HarmonicBondForce();bond->addBond(2,26,0.9,1.2);bond->setUsesPeriodicBoundaryConditions(box!=0);bond->setForceGroup(1);s.addForce(bond);
 auto* ext=new CustomExternalForce("0.1*x^2+0.03*y-0.05*z");for(int i=0;i<s.getNumParticles();++i)ext->addParticle(i,{});ext->setForceGroup(3);s.addForce(ext);
 auto* dpd=new DPDIntegrator(0,1.3,0.85,0.003);dpd->setRandomNumberSeed(191);dpd->setConstraintTolerance(1e-11);
 int labels[]={-7,0,12345};
 if(types) {
  for(int i=0;i<n;++i)dpd->setParticleType(i,labels[i%3]);
  dpd->addTypePair(-7,0,2.3,1.05);dpd->addTypePair(12345,12345,0.4,1.12);dpd->addTypePair(0,-7,0.6,0.82);dpd->addTypePair(88,89,7,50);
 }
 if(site) {dpd->setParticleType(n,99);for(int label: labels)dpd->addTypePair(99,label,0,1e-8);}
 CompoundIntegrator root;
 Integrator* top=dpd;
 if(routing) {auto* branch=new CompoundIntegrator();branch->addIntegrator(dpd);root.addIntegrator(branch);root.addIntegrator(new VerletIntegrator(0.001));root.setCurrentIntegrator(routing==2?1:0);top=&root;}
 {
 Context context(s,*top,Platform::getPlatformByName("Reference"));
 std::vector<Vec3> p,v;
 for(int i=0;i<n;++i) {
  Vec3 value(0.37*(i%3)-0.3,0.39*((i/3)%3)-0.6,0.41*(i/9)-0.5);
  if(box && i>=2) {if(i%5==0)value+=c;if(i%7==0)value-=b;if(i%11==0)value+=a;}
  p.push_back(value);v.push_back(Vec3(0.12*(i%4)-0.15,0.09*(i%5)-0.11,0.07*(i%3)-0.08));
 }
 if(site) {p.push_back(Vec3());v.push_back(Vec3(0.2,0.3,0.4));}
 context.setPositions(p);context.computeVirtualSites();context.setVelocities(v);context.applyVelocityConstraints(1e-11);context.setTime(1.7);context.setStepCount(11);
 int masks[]={-1,2,8,10};
 for(int event=0;event<12;++event) {
  dpd->setStepSize(0.001+0.0003*(event%4));dpd->setIntegrationForceGroups(masks[event%4]);dpd->setConstraintTolerance(1e-10/(1+event%3));
  if(event==3) {dpd->setDefaultFriction(8);dpd->setDefaultCutoff(0.01);dpd->setParticleType(3,999);if(types)dpd->setTypePairParameters(0,0,0,7,0.01);dpd->addTypePair(0,0,50,0.001);}
  if(event==5 && box) {a*=1.04;b*=0.97;c*=1.02;context.setPeriodicBoxVectors(a,b,c);}
  if(event==7)context.setVelocitiesToTemperature(0,72);
  for(int sample=0;sample<2;++sample) {
   if(sample) {if(routing==1)top->step(2);else dpd->step(2);}
   auto state=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy);
   std::cout<<"{\"box\":"<<box<<",\"types\":"<<types<<",\"constrained\":"<<constrained<<",\"site\":"<<site<<",\"routing\":"<<routing<<",\"event\":"<<event<<",\"sample\":"<<sample
    <<",\"time\":"<<state.getTime()<<",\"step_count\":"<<state.getStepCount()<<",\"kinetic\":"<<state.getKineticEnergy()<<",\"energy\":"<<state.getPotentialEnergy()<<",\"dt\":"<<dpd->getStepSize()<<",\"positions\":";
   vectors(state.getPositions());std::cout<<",\"velocities\":";vectors(state.getVelocities());std::cout<<",\"forces\":";vectors(state.getForces());std::cout<<"}\n";
  }
 }
 }
 if(!routing)delete dpd;
}
void moments(double temperature, double friction) {
 System s;s.addParticle(2);s.addParticle(5);
 DPDIntegrator integrator(0,friction,1,0.01);integrator.setRandomNumberSeed(713);integrator.setConstraintTolerance(1e-10);
 Context c(s,integrator,Platform::getPlatformByName("Reference"));integrator.setTemperature(temperature);
 double sum=0,sum2=0,maxMomentum=0;int count=16000;
 for(int k=0;k<count;++k) {
  c.setPositions({Vec3(),Vec3(0.6,0,0)});c.setVelocities({Vec3(0.3,0.2,-0.1),Vec3(-0.4,0.2,-0.1)});integrator.step(1);
  auto state=c.getState(State::Velocities);auto v=state.getVelocities();double relative=v[1][0]-v[0][0];sum+=relative;sum2+=relative*relative;
  maxMomentum=std::max(maxMomentum,std::abs(2*v[0][0]+5*v[1][0]+1.4));
 }
 double mean=sum/count;
 std::cout<<"{\"kind\":\"pair_moments\",\"temperature\":"<<temperature<<",\"friction\":"<<friction<<",\"count\":"<<count<<",\"mean\":"<<mean<<",\"variance\":"<<sum2/count-mean*mean<<",\"max_momentum_error\":"<<maxMomentum<<"}\n";
}
void equilibrium(double temperature, double friction) {
 int n=64;System s;for(int i=0;i<n;++i)s.addParticle(1);
 s.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0,3,0),Vec3(0,0,3));
 auto* marker=new HarmonicBondForce();marker->setUsesPeriodicBoundaryConditions(true);s.addForce(marker);
 DPDIntegrator integrator(temperature,friction,1.2,0.002);integrator.setRandomNumberSeed(719);integrator.setIntegrationForceGroups(0);
 Context c(s,integrator,Platform::getPlatformByName("Reference"));
 std::vector<Vec3> p;for(int i=0;i<n;++i)p.push_back(Vec3(0.75*(i%4),0.75*((i/4)%4),0.75*(i/16)));
 c.setPositions(p);c.setVelocities(std::vector<Vec3>(n));integrator.step(10000);
 double sum=0,sum2=0,maxMomentum=0;int count=5000;
 for(int k=0;k<count;++k) {
  integrator.step(10);auto state=c.getState(State::Velocities|State::Energy);double ke=state.getKineticEnergy();sum+=ke;sum2+=ke*ke;
  Vec3 momentum;for(auto v: state.getVelocities())momentum+=v;for(int axis=0;axis<3;++axis)maxMomentum=std::max(maxMomentum,std::abs(momentum[axis]));
 }
 double mean=sum/count;
 std::cout<<"{\"kind\":\"equilibrium\",\"temperature\":"<<temperature<<",\"friction\":"<<friction<<",\"count\":"<<count<<",\"mean\":"<<mean<<",\"variance\":"<<sum2/count-mean*mean<<",\"max_momentum_error\":"<<maxMomentum<<"}\n";
}
void invalidPairs() {
 for(int mode=0;mode<4;++mode) {
  System s;s.addParticle(1);s.addParticle(mode<2?0:2);
  DPDIntegrator integrator(mode==1?300:0,mode==3?0:1,1,0.01);
  Context c(s,integrator,Platform::getPlatformByName("Reference"));
  c.setPositions({Vec3(),Vec3(mode<2?0.5:0,0,0)});c.setVelocities({Vec3(),Vec3()});integrator.step(1);
  auto state=c.getState(State::Positions|State::Velocities);
  bool finite=std::isfinite(state.getPositions()[0][0]) && std::isfinite(state.getVelocities()[0][0]);
  std::cout<<"{\"kind\":\"invalid_pair\",\"mode\":"<<mode<<",\"finite\":"<<(finite?"true":"false")<<"}\n";
 }
}
int main() {std::cout<<std::setprecision(17);for(int b=0;b<3;++b)for(int t=0;t<2;++t)for(int c=0;c<2;++c)for(int v=0;v<2;++v)for(int r=0;r<3;++r)run(b,t,c,v,r);for(double t: {50.,300.,600.})for(double f: {1.,20.})moments(t,f);for(double t: {70.,300.,600.})equilibrium(t,10);invalidPairs();}
