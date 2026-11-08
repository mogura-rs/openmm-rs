use pyo3::prelude::*;
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::{Rc, Weak},
};

thread_local! {
    static OWNERS: RefCell<HashMap<usize, Weak<()>>> = RefCell::new(HashMap::new());
}

pub(crate) struct ForceOwner {
    token: Rc<()>,
    objects: Vec<usize>,
}

impl Default for ForceOwner {
    fn default() -> Self {
        Self {
            token: Rc::new(()),
            objects: Vec::new(),
        }
    }
}

impl ForceOwner {
    pub(crate) fn claim(&mut self, force: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = force.as_ptr() as usize;
        OWNERS.with(|owners| {
            let mut owners = owners.borrow_mut();
            if owners.get(&key).and_then(Weak::upgrade).is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "the Force object is already owned by a System or CustomCVForce",
                ));
            }
            owners.insert(key, Rc::downgrade(&self.token));
            Ok(())
        })?;
        self.objects.push(key);
        Ok(())
    }

    pub(crate) fn release(&mut self, force: &Bound<'_, PyAny>) {
        let key = force.as_ptr() as usize;
        self.objects.retain(|&object| object != key);
        OWNERS.with(|owners| {
            owners.borrow_mut().remove(&key);
        });
    }
}

impl Drop for ForceOwner {
    fn drop(&mut self) {
        let _ = OWNERS.try_with(|owners| {
            let mut owners = owners.borrow_mut();
            for key in &self.objects {
                if owners
                    .get(key)
                    .is_some_and(|owner| owner.as_ptr() == Rc::as_ptr(&self.token))
                {
                    owners.remove(key);
                }
            }
        });
    }
}
