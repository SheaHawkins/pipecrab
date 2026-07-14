use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

pub struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: The caller provides the allocation contract required by `GlobalAlloc`.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: The caller provides the deallocation contract required by `GlobalAlloc`.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GA: Counting = Counting;

pub fn allocs(f: impl FnOnce()) -> u64 {
    let start = ALLOCS.with(Cell::get);
    f();
    ALLOCS.with(Cell::get) - start
}
