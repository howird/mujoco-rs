use std::ffi::CString;

/// An error when adding a file to a [`Vfs`] via [`Vfs::add_file()`]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum AddError {
    VfsFull,
    RepeatedName,
}
impl std::fmt::Display for AddError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}
impl std::error::Error for AddError {}

pub struct Vfs {
    pub(crate) vfs: Box<mujoco_rs_sys::no_render::mjVFS_>,
}
impl Vfs {
    /// Initializes a new empty `Vfs`
    pub fn new() -> Self {
        Self::default()
    }

    /// Deletes a file from the `Vfs` if it exists, and returns if such a file
    /// was found
    pub fn delete_file(&mut self, filename: &str) -> bool {
        let c_str = CString::new(filename).unwrap();
        let result = unsafe {
            mujoco_rs_sys::no_render::mj_deleteFileVFS(&mut *self.vfs, c_str.as_ptr())
        };
        debug_assert!(result == 0 || result == -1);
        result != -1
    }

    /// Adds a file to the `Vfs` from some given contents
    pub fn add_file(
        &mut self,
        filename: &str,
        contents: &[u8],
    ) -> Result<(), AddError> {
        let filename = CString::new(filename).unwrap();
        let file_size = contents.len();
        let add_errno = unsafe {
            mujoco_rs_sys::no_render::mj_addBufferVFS(
                &mut *self.vfs,
                filename.as_ptr(),
                contents.as_ptr() as *const std::os::raw::c_void,
                file_size as std::os::raw::c_int,
            )
        };

        match add_errno {
            1 => Err(AddError::VfsFull),
            2 => Err(AddError::RepeatedName),
            0 => Ok(()),
            _ => unreachable!(),
        }?;

        Ok(())
    }
}
impl Default for Vfs {
    fn default() -> Self {
        let mut result = Self {
            vfs: unsafe {
                Box::from_raw(std::alloc::alloc(std::alloc::Layout::new::<
                    mujoco_rs_sys::no_render::mjVFS_,
                >()) as *mut _)
            }, // Default::default(),
        };
        unsafe { mujoco_rs_sys::no_render::mj_defaultVFS(&mut *result.vfs) };
        result
    }
}
impl Drop for Vfs {
    fn drop(&mut self) {
        unsafe { mujoco_rs_sys::no_render::mj_deleteVFS(&mut *self.vfs) }
    }
}
