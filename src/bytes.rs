// Portions Copyright 2019 Red Hat, Inc.
//
// Portions Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE-BSD-3-Clause file.
//
// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause

//! Define the `ByteValued` trait to mark that it is safe to instantiate the struct with random
//! data.

use std::io::{Read, Write};
use std::result::Result;
use std::sync::atomic::Ordering;
use zerocopy::{AsBytes, FromBytes};

use crate::atomic_integer::AtomicInteger;


/// A trait used to identify types which can be accessed atomically by proxy.
pub trait AtomicAccess: FromBytes
    // Could not find a more succinct way of stating that `Self` can be converted
    // into `Self::A::V`, and the other way around.
    + From<<<Self as AtomicAccess>::A as AtomicInteger>::V>
    + Into<<<Self as AtomicAccess>::A as AtomicInteger>::V>
{
    /// The `AtomicInteger` that atomic operations on `Self` are based on.
    type A: AtomicInteger;
}

macro_rules! impl_atomic_access {
    ($T:ty, $A:path) => {
        impl AtomicAccess for $T {
            type A = $A;
        }
    };
}

impl_atomic_access!(i8, std::sync::atomic::AtomicI8);
impl_atomic_access!(i16, std::sync::atomic::AtomicI16);
impl_atomic_access!(i32, std::sync::atomic::AtomicI32);
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x"
))]
impl_atomic_access!(i64, std::sync::atomic::AtomicI64);

impl_atomic_access!(u8, std::sync::atomic::AtomicU8);
impl_atomic_access!(u16, std::sync::atomic::AtomicU16);
impl_atomic_access!(u32, std::sync::atomic::AtomicU32);
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc64",
    target_arch = "s390x"
))]
impl_atomic_access!(u64, std::sync::atomic::AtomicU64);

impl_atomic_access!(isize, std::sync::atomic::AtomicIsize);
impl_atomic_access!(usize, std::sync::atomic::AtomicUsize);

/// A container to host a range of bytes and access its content.
///
/// Candidates which may implement this trait include:
/// - anonymous memory areas
/// - mmapped memory areas
/// - data files
/// - a proxy to access memory on remote
pub trait Bytes<A> {
    /// Associated error codes
    type E;

    /// Writes a slice into the container at `addr`.
    ///
    /// Returns the number of bytes written. The number of bytes written can
    /// be less than the length of the slice if there isn't enough room in the
    /// container.
    fn write(&self, buf: &[u8], addr: A) -> Result<usize, Self::E>;

    /// Reads data from the container at `addr` into a slice.
    ///
    /// Returns the number of bytes read. The number of bytes read can be less than the length
    /// of the slice if there isn't enough data within the container.
    fn read(&self, buf: &mut [u8], addr: A) -> Result<usize, Self::E>;

    /// Writes the entire content of a slice into the container at `addr`.
    ///
    /// # Errors
    ///
    /// Returns an error if there isn't enough space within the container to write the entire slice.
    /// Part of the data may have been copied nevertheless.
    fn write_slice(&self, buf: &[u8], addr: A) -> Result<(), Self::E>;

    /// Reads data from the container at `addr` to fill an entire slice.
    ///
    /// # Errors
    ///
    /// Returns an error if there isn't enough data within the container to fill the entire slice.
    /// Part of the data may have been copied nevertheless.
    fn read_slice(&self, buf: &mut [u8], addr: A) -> Result<(), Self::E>;

    /// Writes an object into the container at `addr`.
    ///
    /// # Errors
    ///
    /// Returns an error if the object doesn't fit inside the container.
    fn write_obj<T: AsBytes>(&self, val: T, addr: A) -> Result<(), Self::E> {
        self.write_slice(val.as_bytes(), addr)
    }

    /// Reads an object from the container at `addr`.
    ///
    /// Reading from a volatile area isn't strictly safe as it could change mid-read.
    /// However, as long as the type T is plain old data and can handle random initialization,
    /// everything will be OK.
    ///
    /// # Errors
    ///
    /// Returns an error if there's not enough data inside the container.
    fn read_obj<T: AsBytes + FromBytes>(&self, addr: A) -> Result<T, Self::E> {
        let mut result = T::new_zeroed();
        self.read_slice(result.as_bytes_mut(), addr).map(|_| result)
    }

    /// Reads up to `count` bytes from an object and writes them into the container at `addr`.
    ///
    /// Returns the number of bytes written into the container.
    ///
    /// # Arguments
    /// * `addr` - Begin writing at this address.
    /// * `src` - Copy from `src` into the container.
    /// * `count` - Copy `count` bytes from `src` into the container.
    #[deprecated(
        note = "Use `.read_volatile_from` or the functions of the `ReadVolatile` trait instead"
    )]
    fn read_from<F>(&self, addr: A, src: &mut F, count: usize) -> Result<usize, Self::E>
    where
        F: Read;

    /// Reads exactly `count` bytes from an object and writes them into the container at `addr`.
    ///
    /// # Errors
    ///
    /// Returns an error if `count` bytes couldn't have been copied from `src` to the container.
    /// Part of the data may have been copied nevertheless.
    ///
    /// # Arguments
    /// * `addr` - Begin writing at this address.
    /// * `src` - Copy from `src` into the container.
    /// * `count` - Copy exactly `count` bytes from `src` into the container.
    #[deprecated(
        note = "Use `.read_exact_volatile_from` or the functions of the `ReadVolatile` trait instead"
    )]
    fn read_exact_from<F>(&self, addr: A, src: &mut F, count: usize) -> Result<(), Self::E>
    where
        F: Read;

    /// Reads up to `count` bytes from the container at `addr` and writes them it into an object.
    ///
    /// Returns the number of bytes written into the object.
    ///
    /// # Arguments
    /// * `addr` - Begin reading from this address.
    /// * `dst` - Copy from the container to `dst`.
    /// * `count` - Copy `count` bytes from the container to `dst`.
    #[deprecated(
        note = "Use `.write_volatile_to` or the functions of the `WriteVolatile` trait instead"
    )]
    fn write_to<F>(&self, addr: A, dst: &mut F, count: usize) -> Result<usize, Self::E>
    where
        F: Write;

    /// Reads exactly `count` bytes from the container at `addr` and writes them into an object.
    ///
    /// # Errors
    ///
    /// Returns an error if `count` bytes couldn't have been copied from the container to `dst`.
    /// Part of the data may have been copied nevertheless.
    ///
    /// # Arguments
    /// * `addr` - Begin reading from this address.
    /// * `dst` - Copy from the container to `dst`.
    /// * `count` - Copy exactly `count` bytes from the container to `dst`.
    #[deprecated(
        note = "Use `.write_all_volatile_to` or the functions of the `WriteVolatile` trait instead"
    )]
    fn write_all_to<F>(&self, addr: A, dst: &mut F, count: usize) -> Result<(), Self::E>
    where
        F: Write;

    /// Atomically store a value at the specified address.
    fn store<T: AtomicAccess>(&self, val: T, addr: A, order: Ordering) -> Result<(), Self::E>;

    /// Atomically load a value from the specified address.
    fn load<T: AtomicAccess>(&self, addr: A, order: Ordering) -> Result<T, Self::E>;
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]
    use super::*;

    use std::cell::RefCell;
    use std::fmt::Debug;
    use std::mem::{align_of, size_of};

    use zerocopy::FromZeroes;

    use crate::VolatileSlice;

    // Helper method to test atomic accesses for a given `b: Bytes` that's supposed to be
    // zero-initialized.
    pub fn check_atomic_accesses<A, B>(b: B, addr: A, bad_addr: A)
    where
        A: Copy,
        B: Bytes<A>,
        B::E: Debug,
    {
        let val = 100u32;

        assert_eq!(b.load::<u32>(addr, Ordering::Relaxed).unwrap(), 0);
        b.store(val, addr, Ordering::Relaxed).unwrap();
        assert_eq!(b.load::<u32>(addr, Ordering::Relaxed).unwrap(), val);

        assert!(b.load::<u32>(bad_addr, Ordering::Relaxed).is_err());
        assert!(b.store(val, bad_addr, Ordering::Relaxed).is_err());
    }

    fn check_byte_valued_type<T>()
    where
        T: FromBytes + AsBytes + PartialEq + Debug + Default + Copy,
    {
        let mut data = [0u8; 48];
        let pre_len = {
            let (pre, _, _) = unsafe { data.align_to::<T>() };
            pre.len()
        };
        {
            let aligned_data = &mut data[pre_len..pre_len + size_of::<T>()];
            {
                let mut val: T = Default::default();
                assert_eq!(T::ref_from(aligned_data), Some(&val));
                assert_eq!(T::mut_from(aligned_data), Some(&mut val));
                assert_eq!(val.as_bytes(), aligned_data);
                assert_eq!(val.as_bytes_mut(), aligned_data);
            }
        }
        for i in 1..size_of::<T>().min(align_of::<T>()) {
            let begin = pre_len + i;
            let end = begin + size_of::<T>();
            let unaligned_data = &mut data[begin..end];
            {
                if align_of::<T>() != 1 {
                    assert_eq!(T::ref_from(unaligned_data), None);
                    assert_eq!(T::mut_from(unaligned_data), None);
                }
            }
        }
        // Check the early out condition
        {
            assert!(T::ref_from(&data).is_none());
            assert!(T::mut_from(&mut data).is_none());
        }
    }

    #[test]
    fn test_byte_valued() {
        check_byte_valued_type::<u8>();
        check_byte_valued_type::<u16>();
        check_byte_valued_type::<u32>();
        check_byte_valued_type::<u64>();
        check_byte_valued_type::<u128>();
        check_byte_valued_type::<usize>();
        check_byte_valued_type::<i8>();
        check_byte_valued_type::<i16>();
        check_byte_valued_type::<i32>();
        check_byte_valued_type::<i64>();
        check_byte_valued_type::<i128>();
        check_byte_valued_type::<isize>();
    }

    pub const MOCK_BYTES_CONTAINER_SIZE: usize = 10;

    pub struct MockBytesContainer {
        container: RefCell<[u8; MOCK_BYTES_CONTAINER_SIZE]>,
    }

    impl MockBytesContainer {
        pub fn new() -> Self {
            MockBytesContainer {
                container: RefCell::new([0; MOCK_BYTES_CONTAINER_SIZE]),
            }
        }

        pub fn validate_slice_op(&self, buf: &[u8], addr: usize) -> Result<(), ()> {
            if MOCK_BYTES_CONTAINER_SIZE - buf.len() <= addr {
                return Err(());
            }

            Ok(())
        }
    }

    impl Bytes<usize> for MockBytesContainer {
        type E = ();

        fn write(&self, _: &[u8], _: usize) -> Result<usize, Self::E> {
            unimplemented!()
        }

        fn read(&self, _: &mut [u8], _: usize) -> Result<usize, Self::E> {
            unimplemented!()
        }

        fn write_slice(&self, buf: &[u8], addr: usize) -> Result<(), Self::E> {
            self.validate_slice_op(buf, addr)?;

            let mut container = self.container.borrow_mut();
            container[addr..addr + buf.len()].copy_from_slice(buf);

            Ok(())
        }

        fn read_slice(&self, buf: &mut [u8], addr: usize) -> Result<(), Self::E> {
            self.validate_slice_op(buf, addr)?;

            let container = self.container.borrow();
            buf.copy_from_slice(&container[addr..addr + buf.len()]);

            Ok(())
        }

        fn read_from<F>(&self, _: usize, _: &mut F, _: usize) -> Result<usize, Self::E>
        where
            F: Read,
        {
            unimplemented!()
        }

        fn read_exact_from<F>(&self, _: usize, _: &mut F, _: usize) -> Result<(), Self::E>
        where
            F: Read,
        {
            unimplemented!()
        }

        fn write_to<F>(&self, _: usize, _: &mut F, _: usize) -> Result<usize, Self::E>
        where
            F: Write,
        {
            unimplemented!()
        }

        fn write_all_to<F>(&self, _: usize, _: &mut F, _: usize) -> Result<(), Self::E>
        where
            F: Write,
        {
            unimplemented!()
        }

        fn store<T: AtomicAccess>(
            &self,
            _val: T,
            _addr: usize,
            _order: Ordering,
        ) -> Result<(), Self::E> {
            unimplemented!()
        }

        fn load<T: AtomicAccess>(&self, _addr: usize, _order: Ordering) -> Result<T, Self::E> {
            unimplemented!()
        }
    }

    #[test]
    fn test_bytes() {
        let bytes = MockBytesContainer::new();

        assert!(bytes.write_obj(std::u64::MAX, 0).is_ok());
        assert_eq!(bytes.read_obj::<u64>(0).unwrap(), std::u64::MAX);

        assert!(bytes
            .write_obj(std::u64::MAX, MOCK_BYTES_CONTAINER_SIZE)
            .is_err());
        assert!(bytes.read_obj::<u64>(MOCK_BYTES_CONTAINER_SIZE).is_err());
    }

    #[repr(C)]
    #[derive(Copy, Clone, Default, FromBytes, FromZeroes, AsBytes)]
    struct S {
        a: u32,
        b: u32,
    }

    #[test]
    fn byte_valued_slice() {
        let a: [u8; 8] = [0, 0, 0, 0, 1, 1, 1, 1];
        let mut s: S = Default::default();
        VolatileSlice::from(s.as_bytes_mut()).copy_from(&a);
        assert_eq!(s.a, 0);
        assert_eq!(s.b, 0x0101_0101);
    }
}
