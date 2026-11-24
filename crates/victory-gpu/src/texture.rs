//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    last: Mutex<Last>,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after,
/// with room to read the memory into.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents live on
/// the host until the title writes their memory again. Holds the write-watch
/// asking current when the copy was recorded, or `None` when the texture's
/// memory is not write-watched.
static HOST_WRITTEN: Mutex<Option<HashMap<usize, Option<u64>>>> = Mutex::new(None);

/// Whether any page in `written` (from [`crate::watch::writes`]) was written
/// after asking `since`.
fn written_since(written: &[(usize, u64)], since: u64) -> bool {
    written.iter().any(|&(_, w)| w > since)
}

/// The title memory range `[start, end)` of `resource`'s plane 0, if it has
/// a console layout.
fn memory_range(resource: This) -> Option<(usize, usize)> {
    if let Some(s) = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|map| map.get(&(resource as usize)))
        .cloned()
    {
        let start = s.memory as usize;
        let size = s.layout.size as usize;
        return (size > 0).then_some((start, start + size));
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let layout = crate::resource::texture_layout(&desc)?;
    let start = memory as usize;
    let size = usize::try_from(layout.planes[0].layout.size).ok()?;
    if size == 0 {
        return None;
    }
    Some((start, start.checked_add(size)?))
}

/// A host copy or resolve writes `resource`.
///
/// Called when the copy is recorded rather than when the GPU runs it, so
/// later commands recorded before submission already see the texture as
/// host-written; a title write to its memory between recording and
/// submission gets a later asking and supersedes the host copy on the next
/// [`upload`].
pub(crate) fn host_written(resource: This) {
    if crate::resource::host_resource(resource).is_some_and(|t| !title_written(t)) {
        return;
    }
    let asked = memory_range(resource)
        .and_then(|(start, end)| crate::watch::writes(start, end, false))
        .map(|(now, _)| now);
    let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(HashMap::new)
        .insert(resource as usize, asked);
}

/// Whether `resource` still keeps its host-written contents. If its memory
/// is write-watched and a page was written after the copy was recorded,
/// clears the host-written state and resets any staging cursor so the next
/// refresh detiles the new contents.
///
/// shortcut: memory that is not write-watched cannot be checked here, so
/// host-written contents keep winning there until the resource is released.
fn host_still_owns(resource: This) -> bool {
    let key = resource as usize;
    let recorded = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|map| map.get(&key).copied());
    let Some(since) = recorded else {
        return false;
    };
    let Some(since) = since else {
        return true;
    };
    let Some((start, end)) = memory_range(resource) else {
        return true;
    };
    let Some((_, written)) = crate::watch::writes(start, end, false) else {
        return true;
    };
    if !written_since(&written, since) {
        return true;
    }
    {
        let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut()
            && map.get(&key) == Some(&Some(since))
        {
            map.remove(&key);
        }
    }
    if let Some(s) = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|map| map.get(&key))
        .cloned()
    {
        let mut last = s.last.lock().unwrap_or_else(|e| e.into_inner());
        last.asked = 0;
        last.hash = None;
    }
    false
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    if host_still_owns(resource) || !list.first_use(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
            if last.asked != 0 && !written_since(&written, last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            self.detile(&last.read);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return 0;
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK, guid, slot};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    const RGBA8: u32 = 28;
    const LINEAR: u32 = 1;

    type Release = extern "system" fn(This) -> u32;

    fn solid(rgba: [u8; 4]) -> [u8; 256] {
        std::array::from_fn(|i| rgba[i % 4])
    }

    #[test]
    fn written_since_detects_pages_written_after_asking() {
        assert!(!written_since(&[], 0));
        assert!(!written_since(&[], 5));
        assert!(!written_since(&[(1, 3), (2, 5)], 5));
        assert!(written_since(&[(1, 3), (2, 6)], 5));
        assert!(written_since(&[(1, 1)], 0));
    }

    #[test]
    fn host_written_yields_after_write_in_watched_memory() {
        let size = crate::watch::PAGE;
        // SAFETY: a fresh write-watched page, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!p.is_null());

        let watched = crate::resource::test_texture_at(p as u64, 64, 1, RGBA8, LINEAR);
        // A write before the host copy must not invalidate it.
        // SAFETY: committed write-watched memory of ours.
        unsafe { p.write(0x11) };
        host_written(watched);
        assert!(host_still_owns(watched));

        // A write after the host copy clears host-written ownership.
        // SAFETY: committed write-watched memory of ours.
        unsafe { p.write(0x22) };
        assert!(!host_still_owns(watched));
        assert!(!host_still_owns(watched));

        // A later host copy claims ownership again until the next write.
        host_written(watched);
        assert!(host_still_owns(watched));

        // Non-write-watched memory keeps host contents even after a write.
        let mut plain_mem = vec![0u8; 4096];
        let plain =
            crate::resource::test_texture_at(plain_mem.as_ptr() as u64, 64, 1, RGBA8, LINEAR);
        host_written(plain);
        plain_mem[0] = 0x33;
        assert!(host_still_owns(plain));

        assert_eq!(slot::<Release>(watched, 0x10)(watched), 0);
        assert_eq!(slot::<Release>(plain, 0x10)(plain), 0);
        crate::watch::forget_pages(p as usize, p as usize + size);
        // SAFETY: our allocation, no resource refers to it anymore.
        unsafe { VirtualFree(p.cast(), 0, MEM_RELEASE) };
    }

    /// A 64x1 RGBA8 title texture in write-watched memory keeps the pixels of
    /// a host `CopyResource` until the title writes its memory again, after
    /// which `upload()` detiles the new pixels into the host texture.
    #[test]
    fn title_write_after_host_copy_reaches_the_host_texture() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };

        let page = crate::watch::PAGE;
        // Two distinct watch pages so writes to `src` do not touch `dst`.
        // SAFETY: a fresh write-watched allocation, freed after all resources
        // and lists are released.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * page,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!base.is_null());
        let dst_mem = base;
        // SAFETY: second page of the 2-page allocation.
        let src_mem = unsafe { base.add(page) };

        let initial = solid([0x11, 0x22, 0x33, 0xff]);
        let copied = solid([0xaa, 0xbb, 0xcc, 0xff]);
        let updated = solid([0x44, 0x55, 0x66, 0xff]);
        // SAFETY: each page is at least 256 bytes of committed memory.
        unsafe {
            std::ptr::copy_nonoverlapping(initial.as_ptr(), dst_mem, initial.len());
            std::ptr::copy_nonoverlapping(copied.as_ptr(), src_mem, copied.len());
        }

        let dst = crate::resource::test_texture_at(dst_mem as u64, 64, 1, RGBA8, LINEAR);
        let src = crate::resource::test_texture_at(src_mem as u64, 64, 1, RGBA8, LINEAR);

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let null = std::ptr::null();
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        let close_and_submit = |list: This| {
            assert_eq!(
                slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
                S_OK
            );
            slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
            assert_eq!(
                slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50)(
                    list, allocator, null
                ),
                S_OK
            );
        };

        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let back = host.new_list().unwrap();

        let read_pixels = |resource: This| -> [u8; 256] {
            const COPY_SOURCE: u32 = 0x800;
            let host_tex = crate::resource::host_resource(resource).unwrap();
            crate::list::ensure_state(&back, resource, COPY_SOURCE);
            let out_to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(&readback) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            Width: 64,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let out_from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(host_tex) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            // SAFETY: an open list and live resources.
            unsafe {
                back.list
                    .CopyTextureRegion(&out_to, 0, 0, 0, &out_from, None)
            };
            back.close();
            host.execute(&[&back]);
            back.reset();
            let mut mapped = std::ptr::null_mut();
            let mut pixels = [0u8; 256];
            // SAFETY: the copy finished (`execute` waits); the buffer is 256 bytes.
            unsafe {
                readback
                    .Map(
                        0,
                        Some(&D3D12_RANGE { Begin: 0, End: 256 }),
                        Some(&mut mapped),
                    )
                    .unwrap();
                std::ptr::copy_nonoverlapping(mapped.cast::<u8>(), pixels.as_mut_ptr(), 256);
                readback.Unmap(0, None);
            }
            pixels
        };

        let host_dst = crate::resource::host_resource(dst).unwrap();
        let host_list = crate::list::host_list(list).unwrap();

        // 1. Initial upload from title memory into the host texture.
        upload(host_list, dst, host_dst);
        close_and_submit(list);
        assert_eq!(read_pixels(dst), initial, "initial upload");

        // 2. Host CopyResource overwrites `dst` with `src`.
        slot::<extern "system" fn(This, This, This)>(list, 0x88)(list, dst, src);
        close_and_submit(list);
        assert_eq!(read_pixels(dst), copied, "host copy");

        // 3. Without a title write, a later upload keeps the host-copied pixels.
        upload(host_list, dst, host_dst);
        close_and_submit(list);
        assert_eq!(read_pixels(dst), copied, "kept without title write");

        // 4. After the title writes `dst`'s memory, upload detiles the new pixels.
        // SAFETY: committed write-watched memory of ours.
        unsafe { std::ptr::copy_nonoverlapping(updated.as_ptr(), dst_mem, updated.len()) };
        upload(host_list, dst, host_dst);
        close_and_submit(list);
        assert_eq!(read_pixels(dst), updated, "title write after host copy");

        drop(back);
        drop(readback);
        assert_eq!(slot::<Release>(list, 0x10)(list), 0);
        assert_eq!(slot::<Release>(allocator, 0x10)(allocator), 0);
        assert_eq!(slot::<Release>(queue, 0x10)(queue), 0);
        assert_eq!(slot::<Release>(dst, 0x10)(dst), 0);
        assert_eq!(slot::<Release>(src, 0x10)(src), 0);
        crate::watch::forget_pages(base as usize, base as usize + 2 * page);
        // SAFETY: our allocation, all resources and lists released above.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }
}
