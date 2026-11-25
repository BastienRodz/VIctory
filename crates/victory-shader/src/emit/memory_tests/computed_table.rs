//! Observe the texel and the complete read bitmap for a computed texture-table entry.
use super::{S_ENDPGM, S_WAITCNT_ALL, ZERO, buffer_store_dword, s_load, s_mov_b32, v_mov_b32};
use crate::emit::host::{Dispatch, Host, Root, View};
use crate::emit::{HostShader, Options, VIEW_MARK_NONE, VIEW_READS, emit_with};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, io, range, resource};

fn shader(computed: bool) -> HostShader {
    // Separate tables for the output buffer at s0 and the textures at s4.
    let mut w = vec![s_mov_b32(1, ZERO as u8)];
    w.extend(s_load(2, 8, 0, 0));
    w.push(s_mov_b32(5, ZERO as u8));
    let mut texture = s_load(3, 12, 4, 0);
    if computed {
        // Turn the runtime root constant s6 into the table byte offset.
        w.extend([v_mov_b32(1, 6), 0x7e0c_0501, 0x8f06_8506]); // readfirstlane s6,v1; s6 <<= 5
        texture[1] = 6 << 25;
    }
    w.extend(texture);
    w.extend([S_WAITCNT_ALL, v_mov_b32(0, ZERO), v_mov_b32(1, ZERO)]);
    w.extend([0xf000_1f08, 0x0003_0200, S_WAITCNT_ALL]); // image_load v[2:5], (0,0), s[12:19]
    for channel in 0..4 {
        w.extend(buffer_store_dword(2 + channel, 8, 4 * u16::from(channel)));
    }
    w.push(S_ENDPGM);
    let param = |root, user_data, slot| RootParam {
        root,
        user_data,
        slot,
        sgprs: 1,
        base_register: 0,
        table_offset_dwords: 0,
    };
    let binding = |kind, user_data, slot, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            param(RootType::Table, 0, 0),
            param(RootType::Table, 4, 1),
            param(RootType::Constants, 6, 2),
        ],
        used: vec![
            binding(ResourceType::Uav, 0, 0, 4),
            binding(ResourceType::Srv, 4, 1, 8),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 7 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let (p, st) = range::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = st.unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let c = Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![],
    };
    let shader = emit_with(
        &c,
        &native,
        &p,
        &st,
        &io,
        &res,
        Options { view_marks: true },
    )
    .unwrap();
    assert!(!shader.heap && !shader.memory && shader.view_marks);
    assert!(
        shader.root_cbvs.is_empty() && shader.root_srvs.is_empty() && shader.root_uavs.is_empty()
    );
    shader
}

fn set_bits(bitmap: &[u32]) -> Vec<u32> {
    let mut marked = Vec::new();
    for (word, &value) in bitmap.iter().enumerate() {
        let mut bits = value;
        while bits != 0 {
            marked.push(word as u32 * 32 + bits.trailing_zeros());
            bits &= bits - 1;
        }
    }
    marked
}

#[test]
fn computed_texture_table_entries_report_the_sampled_slots() {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
    };
    let _one = crate::emit::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let red = [1.0, 0.0, 0.0, 1.0];
    let blue = [0.0, 0.0, 1.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];
    let (red_tex, blue_tex, green_tex) = ([red], [blue], [green]);
    let views = [
        View::RawUav(4),
        View::Texture(1, 1, &green_tex),
        View::Texture(1, 1, &red_tex),
        View::Texture(1, 1, &blue_tex),
    ];
    let bitmap_words = (VIEW_READS + views.len() as u32).div_ceil(32);
    let mut observed = Vec::new();
    let mut devices = Vec::new();
    for warp in [false, true] {
        let Ok(host) = Host::new(warp) else { continue };
        // SAFETY: resolve the adapter from the actual device used for dispatch.
        let desc = unsafe {
            let f: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter1 = f.EnumAdapterByLuid(host.device.GetAdapterLuid()).unwrap();
            adapter.GetDesc1().unwrap()
        };
        let luid = (desc.AdapterLuid.HighPart as u64) << 32 | u64::from(desc.AdapterLuid.LowPart);
        if devices.contains(&luid) {
            continue; // Host::new(false) may already have selected WARP on this run.
        }
        devices.push(luid);
        eprintln!(
            "adapter {} LUID={luid:016x} vendor={:04x} device={:04x} flags={:#x}",
            String::from_utf16_lossy(&desc.Description).trim_end_matches('\0'),
            desc.VendorId,
            desc.DeviceId,
            desc.Flags
        );
        for computed in [false, true] {
            let shader = shader(computed);
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            // Change the entry within one table, then keep the entry and move the table base.
            for (base, entry, groups) in [(2, 0, 0), (2, 0, 1), (2, 1, 1), (1, 1, 1)] {
                let mut data = vec![0; shader.user_data_dwords().max(1) as usize];
                for u in &shader.user_data {
                    data[u.index as usize] = match (u.root, u.slot) {
                        (RootType::Table, 0) => 0,
                        (RootType::Table, 1) => base * 32,
                        (RootType::Constants, 2) => entry,
                        _ => panic!("unexpected root input"),
                    };
                }
                // Writes to the output view have no tracked pages. The bitmap
                // therefore contains exactly the texture read marks, if any.
                let roots = vec![
                    (0, Root::Constants(data)),
                    (
                        1,
                        Root::View {
                            srv: true,
                            data: vec![VIEW_MARK_NONE; views.len() * 2],
                        },
                    ),
                    (2, Root::Uav(bitmap_words)),
                ];
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots,
                            views: &views,
                            samplers: &[],
                            groups: [groups, 1, 1],
                        },
                    )
                    .unwrap();
                let slot = if computed { base + entry } else { base };
                let color = if groups == 0 {
                    [0.0; 4]
                } else {
                    [green, red, blue][slot as usize - 1]
                };
                let bits = set_bits(&out[1]);
                eprintln!(
                    "computed={computed} base={base} entry={entry} groups={groups}: color={:?} set_bits={bits:?}",
                    out[0]
                        .iter()
                        .copied()
                        .map(f32::from_bits)
                        .collect::<Vec<_>>()
                );
                assert_eq!(out[0], color.map(f32::to_bits));
                let mark = (computed && groups != 0).then_some(VIEW_READS + slot);
                observed.push((luid, computed, base, entry, groups, bits, mark));
            }
        }
    }
    assert!(!devices.is_empty(), "no SM 6.6 device");
    for (luid, computed, base, entry, groups, bits, mark) in observed {
        assert_eq!(
            bits,
            mark.as_slice(),
            "adapter={luid:016x} computed={computed} base={base} entry={entry} groups={groups}"
        );
    }
}
