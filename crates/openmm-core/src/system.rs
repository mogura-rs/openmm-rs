use crate::constraints::Constraint;
use crate::context::Context;
use crate::units::BoxVectors;
use crate::virtual_sites::VirtualSite;
use burn::prelude::Backend;

use crate::forces::Force;
use crate::integrators::Integrator;

/// Particle definition: mass in Da.
#[derive(Debug, Clone)]
pub struct Particle {
    pub mass: f64,
}

pub(crate) fn validate_box_vectors(vectors: &BoxVectors) {
    let a = vectors[0];
    let b = vectors[1];
    let c = vectors[2];

    assert!(
        a[1] == 0.0 && a[2] == 0.0,
        "first periodic box vector must be parallel to x"
    );
    assert!(
        b[2] == 0.0,
        "second periodic box vector must be in the x-y plane"
    );
    assert!(
        a[0] > 0.0
            && b[1] > 0.0
            && c[2] > 0.0
            && a[0] >= 2.0 * b[0].abs()
            && a[0] >= 2.0 * c[0].abs()
            && b[1] >= 2.0 * c[1].abs(),
        "periodic box vectors must be in reduced form"
    );
}

/// Holds all particles, box geometry, and force field terms.
/// Generic over burn Backend so forces can use tensors.
#[derive(Clone)]
pub struct System<B: Backend> {
    particles: Vec<Particle>,
    box_vectors: Option<BoxVectors>,
    forces: Vec<Box<dyn Force<B>>>,
    constraints: Vec<Constraint>,
    virtual_sites: Vec<Option<VirtualSite>>,
}

impl<B: Backend> System<B> {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            box_vectors: Some([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
            forces: Vec::new(),
            constraints: Vec::new(),
            virtual_sites: Vec::new(),
        }
    }

    /// Returns the index of the added particle.
    pub fn add_particle(&mut self, mass: f64) -> usize {
        assert!(
            mass.is_finite() && mass >= 0.0,
            "particle mass must be nonnegative and finite"
        );
        self.particles.push(Particle { mass });
        self.virtual_sites.push(None);
        self.particles.len() - 1
    }

    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    pub fn set_virtual_site(&mut self, index: usize, site: VirtualSite) {
        self.virtual_sites[index] = Some(site);
    }
    pub fn is_virtual_site(&self, index: usize) -> bool {
        self.virtual_sites[index].is_some()
    }
    pub fn virtual_site(&self, index: usize) -> &VirtualSite {
        self.virtual_sites[index]
            .as_ref()
            .expect("particle is not a virtual site")
    }
    pub(crate) fn virtual_sites(&self) -> &[Option<VirtualSite>] {
        &self.virtual_sites
    }

    pub fn particle(&self, index: usize) -> &Particle {
        &self.particles[index]
    }

    pub fn masses(&self) -> Vec<f64> {
        self.particles.iter().map(|p| p.mass).collect()
    }

    pub fn molecules(&self) -> Vec<Vec<usize>> {
        let mut parents: Vec<usize> = (0..self.num_particles()).collect();
        fn root(parents: &mut [usize], mut particle: usize) -> usize {
            while parents[particle] != particle {
                parents[particle] = parents[parents[particle]];
                particle = parents[particle];
            }
            particle
        }
        let mut join = |a: usize, b: usize| {
            assert!(
                a < parents.len() && b < parents.len(),
                "bonded particle index out of range"
            );
            let a = root(&mut parents, a);
            let b = root(&mut parents, b);
            parents[a.max(b)] = a.min(b);
        };
        for constraint in &self.constraints {
            join(constraint.particle1, constraint.particle2);
        }
        for force in &self.forces {
            for (a, b) in force.bonded_particles() {
                join(a, b);
            }
        }
        for (i, site) in self.virtual_sites.iter().enumerate() {
            if let Some(site) = site {
                for &particle in site.particles() {
                    join(i, particle);
                }
            }
        }
        let mut groups = vec![Vec::new(); parents.len()];
        for i in 0..parents.len() {
            groups[root(&mut parents, i)].push(i);
        }
        groups
            .into_iter()
            .filter(|group| !group.is_empty())
            .collect()
    }

    pub fn set_particle_mass(&mut self, index: usize, mass: f64) {
        assert!(
            mass.is_finite() && mass >= 0.0,
            "particle mass must be nonnegative and finite"
        );
        self.particles[index].mass = mass;
    }

    pub fn add_constraint(&mut self, particle1: usize, particle2: usize, distance: f64) -> usize {
        self.constraints.push(Constraint {
            particle1,
            particle2,
            distance,
        });
        self.constraints.len() - 1
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    pub fn constraint_parameters(&self, index: usize) -> &Constraint {
        &self.constraints[index]
    }

    pub fn set_constraint_parameters(
        &mut self,
        index: usize,
        particle1: usize,
        particle2: usize,
        distance: f64,
    ) {
        self.constraints[index] = Constraint {
            particle1,
            particle2,
            distance,
        };
    }

    pub fn remove_constraint(&mut self, index: usize) {
        self.constraints.remove(index);
    }

    /// Set orthogonal or triclinic periodic box. Vectors in nm, row-major.
    pub fn set_default_periodic_box_vectors(&mut self, vectors: BoxVectors) {
        validate_box_vectors(&vectors);
        self.box_vectors = Some(vectors);
    }

    pub fn default_periodic_box_vectors(&self) -> Option<&BoxVectors> {
        self.box_vectors.as_ref()
    }

    /// Returns the index of the added force.
    pub fn add_force(&mut self, force: Box<dyn Force<B>>) -> usize {
        self.forces.push(force);
        self.forces.len() - 1
    }

    pub fn num_forces(&self) -> usize {
        self.forces.len()
    }

    pub fn forces(&self) -> &[Box<dyn Force<B>>] {
        &self.forces
    }

    pub fn forces_mut(&mut self) -> &mut [Box<dyn Force<B>>] {
        &mut self.forces
    }

    pub fn uses_periodic_boundary_conditions(&self) -> bool {
        self.forces
            .iter()
            .any(|force| force.uses_periodic_boundary_conditions())
    }
}

impl<B: Backend> System<B> {
    pub fn create_context(
        &self,
        integrator: Box<dyn Integrator<B>>,
        device: B::Device,
    ) -> Context<B> {
        Context::new(self.clone(), integrator, device)
    }
}

impl<B: Backend> Default for System<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::nonbonded::{NonbondedForce, NonbondedMethod};
    use crate::VerletIntegrator;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_uses_periodic_boundary_conditions() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(1.0);
        assert!(!system.uses_periodic_boundary_conditions());

        let mut nonbonded = NonbondedForce::new();
        nonbonded.set_nonbonded_method(NonbondedMethod::CutoffPeriodic);
        system.add_force(Box::new(nonbonded));
        assert!(system.uses_periodic_boundary_conditions());
    }

    #[test]
    fn test_clone_preserves_forces() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(1.0);

        let mut nonbonded = NonbondedForce::new();
        nonbonded.add_particle(1.0, 1.0, 0.0);
        nonbonded.add_particle(-1.0, 1.0, 0.0);
        system.add_force(Box::new(nonbonded));

        let cloned = system.clone();
        assert_eq!(cloned.num_particles(), 2);
        assert_eq!(cloned.num_forces(), 1);
        assert!(!cloned.uses_periodic_boundary_conditions());
    }

    #[test]
    fn test_create_multiple_contexts_from_same_system() {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(1.0);

        let mut nonbonded = NonbondedForce::new();
        nonbonded.add_particle(1.0, 1.0, 0.0);
        nonbonded.add_particle(-1.0, 1.0, 0.0);
        system.add_force(Box::new(nonbonded));

        let device: <B as Backend>::Device = Default::default();
        let context1 = system.create_context(Box::new(VerletIntegrator::new(0.001)), device);
        let context2 = system.create_context(Box::new(VerletIntegrator::new(0.002)), device);

        assert_eq!(context1.get_step_count(), 0);
        assert_eq!(context2.get_step_count(), 0);
        assert_eq!(system.num_forces(), 1);
    }
}
