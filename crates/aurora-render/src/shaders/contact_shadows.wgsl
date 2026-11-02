// Screen-space contact shadows of the sun (or the moon), from the
// single-sample depth prepass: a WGSL port of Bend Studio's screen space
// shadows (bend_sss_gpu.h, as shipped in Days Gone), run in the dispatches
// that contact_shadows.rs lays out. The result is the shadow in `.r` (0 lit,
// 1 in contact shadow: `clear` empties the target first each frame, so a
// pixel no wavefront writes, the light's own one, stays lit) that the shading
// combines with the sun's shadow map (`contact_shadow` in common.wgsl).
//
// Original notice:
//
// Copyright 2023 Sony Interactive Entertainment.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// How it works: each workgroup is one "wavefront" of WAVE_SIZE pixels lined
// up along a ray towards the light's point on screen. Its threads read the
// depth of their pixel and the baseline pixels further towards
// the light, divide them by their distance to the light point (in that
// space every ray of light is parallel), share them in workgroup memory,
// and a pixel is in shadow when one of the depths before it along the ray
// comes within a surface thickness of its own. Changes from the original:
// integer texel loads instead of a clamp-to-border sampler (outside the
// screen reads the far depth), the early-out of a whole wavefront through
// workgroup memory (no wave intrinsics in WGSL), the shadow stored rather
// than the light, no debug outputs, occluders taken as deep as the ray went
// under them (up to `tuning.w`) rather than a shell (`depth_delta`), more
// samples in a photo larger than the viewfinder (the four constants below
// replaced in a pipeline of its own, `shader_source` in contact_shadows.rs:
// the shadow keeps its length in the image), Bend's defaults otherwise
// (BilinearSamplingOffsetMode and IgnoreEdgePixels off, no precision
// offset).

struct Params {
    // x, y: the light on screen (pixels, y down), z: its depth, w: 1 when
    // the light is in front of the camera, -1 behind (LightCoordinate)
    light: vec4<f32>,
    // xy: this dispatch's first wave, relative to the light (WaveOffset)
    wave_offset: vec4<i32>,
    // x surface thickness, y bilinear threshold, z shadow contrast, w the
    // occluders' depth in units of surface thickness per unit of reversed
    // depth (0: Bend's shell of one unit)
    tuning: vec4<f32>,
    // Homogeneous unit light direction in pixels, view-depth change, near.
    ray: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;

// Must match WAVE_SIZE in contact_shadows.rs.
const WAVE_SIZE: i32 = 64;
// Shadow samples per pixel: the length of the shadow in pixels. This line
// and the next three are rewritten for a photo (`shader_source`): keep them
// as they are.
const SAMPLE_COUNT: i32 = 60;
// The first samples: one of them alone can shadow the pixel.
const HARD_SHADOW_SAMPLES: i32 = 4;
// The last samples fade out, softening the end of the shadow.
const FADE_OUT_SAMPLES: i32 = 8;
// Maximum shared blocks, and the constant baseline read count.
const READ_COUNT: i32 = 9;
const BASE_READ_COUNT: i32 = 2;
const MAX_REACH: i32 = 480;
const CONTACT_METRES: f32 = 0.1;
// Correctness reference: the bounded cooperative search is only a prefix.
// Continue to the visible ray endpoint; no physical or fixed pixel cutoff.
const FULL_VIEW_REFERENCE: bool = true;
// Reversed depth: the far plane at 0, the near plane at 1.
const FAR_DEPTH: f32 = 0.0;
const NEAR_DEPTH: f32 = 1.0;

var<workgroup> depth_data: array<f32, READ_COUNT * WAVE_SIZE>;
// Bounds of contiguous ray spans. A rejected span cannot change any of the
// four shadow minima; potential hits retain every original depth sample.
const SPAN: i32 = 32;
var<workgroup> depth_ranges: array<vec2<f32>, READ_COUNT * WAVE_SIZE / SPAN>;
// Several live invocations set this concurrently. Atomic writes avoid a
// workgroup data race before the uniform load's barrier (zero at the start).
var<workgroup> any_live: atomic<u32>;

fn load_depth(p: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depth_tex));
    if (any(p < vec2<i32>(0)) || any(p >= size)) {
        return FAR_DEPTH;
    }
    return textureLoad(depth_tex, p, 0);
}

// The pixel's side of the comparison: its own depth along the ray, made
// perspective-correct and scaled to units of surface thickness.
struct ShadowWindow {
    start: f32,
    scale: f32,
    z_sign: f32,
    // the pixel's distance to the light point, and -1 or 1 per sample
    // along the ray (pixel_distance, direction)
    distance: f32,
    direction: f32,
    // the deepest an occluder is taken to be, in units (1: Bend's shell)
    depth: f32,
};

// How far the sample `k` pixels along the ray is from shadowing the pixel:
// 0.75 or less is full shadow (with the contrast of 4), 1 or more lit.
// Bend takes |v - 1|, v being how far the sample's surface lies in front of
// the ray, in units: only a shell from 0.25 to 1.75 units in front shadows.
// A ray that goes under a hoof or a shoe and through its body, behind the
// visible faces, was then stopped only where it grazed their edges: the
// ground took the outline of the caster instead of its shadow (issue 988).
// The window now reaches back to the pixel's own depth when the ray recedes
// from the camera (a sun ahead), up to `depth`: a surface between the pixel
// and the ray is a body the ray went under. One nearer than the pixel keeps
// the shell, or each caster would darken the ground just behind it with a
// copy of its silhouette.
fn depth_delta(w: ShadowWindow, sample_depth: f32, k: i32) -> f32 {
    let v = (sample_depth * w.scale - w.start) * -w.z_sign;
    // When the ray approaches the camera, receding is non-positive at
    // every positive sample distance. Its clamp is exactly one: the shell
    // comparison needs no perspective divide and no depth-volume arithmetic.
    if (w.depth == 1.0 || w.start * w.direction * w.z_sign <= 0.0) {
        return abs(1.0 - v);
    }
    // the pixel's depth seen from the sample, in front of the ray by
    // start × (distance / distance_k - 1)
    let distance_k = max(w.distance + f32(k) * w.direction, 1.0);
    let receding = w.start * f32(k) * w.direction * w.z_sign / distance_k;
    return max(1.0 - v, v - clamp(receding, 1.0, w.depth));
}

// Project a physical ray without dividing by the light's view cosine:
// side lighting remains finite when its projected point is at infinity.
fn search_reach(pixel: vec2<f32>, depth: f32) -> i32 {
    if (depth <= FAR_DEPTH || depth >= NEAR_DEPTH || params.ray.w <= 0.0) {
        return SAMPLE_COUNT;
    }
    let centre = vec2<f32>(textureDimensions(depth_tex)) * 0.5;
    let projected = (pixel - centre) * params.ray.z - params.ray.xy;
    let end_depth = max(params.ray.w / depth + CONTACT_METRES * params.ray.z, params.ray.w);
    let physical_reach = CONTACT_METRES * max(abs(projected.x), abs(projected.y)) / end_depth;
    return i32(clamp(ceil(physical_reach), f32(SAMPLE_COUNT), f32(MAX_REACH)));
}

// Lower bound on depth_delta over a span. Taking the largest permitted
// receding depth is conservative even when that depth varies along the ray.
fn span_lower_bound(w: ShadowWindow, bounds: vec2<f32>, first: f32, last: f32) -> f32 {
    let values = (bounds * w.scale - w.start) * -w.z_sign;
    var limit = 1.0;
    if (w.depth > 1.0 && w.start * w.direction * w.z_sign > 0.0) {
        let distances = max(vec2<f32>(w.distance) + vec2<f32>(first, last) * w.direction, vec2<f32>(1.0));
        let receding = w.start * vec2<f32>(first, last) * w.direction * w.z_sign / distances;
        limit = clamp(max(receding.x, receding.y), 1.0, w.depth);
    }
    return max(1.0 - max(values.x, values.y), min(values.x, values.y) - limit);
}

// The major-axis ray moves one screen pixel per sample. Stop at the
// viewport edge or before crossing the projected light, never at a fixed
// physical distance. Missing off-screen geometry remains a shadow-map job.
fn visible_ray_reach(pixel: vec2<f32>, step: vec2<f32>, distance: f32, direction: f32) -> i32 {
    let size = vec2<f32>(textureDimensions(depth_tex));
    var remaining = vec2<f32>(max(size.x, size.y));
    if (step.x > 0.000001) { remaining.x = (size.x - 0.0001 - pixel.x) / step.x; }
    if (step.x < -0.000001) { remaining.x = (0.0 - pixel.x) / step.x; }
    if (step.y > 0.000001) { remaining.y = (size.y - 0.0001 - pixel.y) / step.y; }
    if (step.y < -0.000001) { remaining.y = (0.0 - pixel.y) / step.y; }
    var end = min(remaining.x, remaining.y);
    if (direction < 0.0) { end = min(end, distance - 0.0001); }
    return i32(clamp(floor(end), 0.0, max(size.x, size.y)));
}

fn isign(v: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(select(select(0, -1, v.x < 0), 1, v.x > 0), select(select(0, -1, v.y < 0), 1, v.y > 0));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    // ---- ComputeWavefrontExtents
    var xy = vec2<i32>(group.yz) * WAVE_SIZE + params.wave_offset.xy;
    let light_xy = floor(params.light.xy) + 0.5;
    let light_xy_fraction = params.light.xy - light_xy;
    let reverse_direction = params.light.w > 0.0;
    let sign_xy = isign(xy);
    let horizontal = abs(xy.x + sign_xy.y) < abs(xy.y - sign_xy.x);
    let axis = vec2<i32>(select(0, sign_xy.y, horizontal), select(-sign_xy.x, 0, horizontal));
    xy = axis * i32(group.x) + xy;
    let xy_f = vec2<f32>(xy);
    let x_axis_major = abs(xy_f.x) > abs(xy_f.y);
    let major_axis = select(xy_f.y, xy_f.x, x_axis_major);
    let major_axis_start = abs(major_axis);
    let major_axis_end = abs(major_axis) - f32(WAVE_SIZE);
    var ma_light_frac = select(light_xy_fraction.y, light_xy_fraction.x, x_axis_major);
    ma_light_frac = select(ma_light_frac, -ma_light_frac, major_axis > 0.0);
    let start_xy = xy_f + light_xy;
    let end_xy = mix(params.light.xy, start_xy, (major_axis_end + ma_light_frac) / (major_axis_start + ma_light_frac));
    let xy_delta = start_xy - end_xy;
    let thread_step = f32(tid ^ select(u32(WAVE_SIZE - 1), 0u, reverse_direction));
    var pixel_xy = mix(start_xy, end_xy, thread_step / f32(WAVE_SIZE));
    let pixel_distance = major_axis_start - thread_step + ma_light_frac;

    // ---- the depths along the ray
    let direction = -params.light.w;
    let z_sign = select(1.0, -1.0, NEAR_DEPTH > FAR_DEPTH);
    let write_xy = vec2<i32>(floor(pixel_xy));
    let sampling_depth = load_depth(write_xy);
    let reach = search_reach(vec2<f32>(write_xy) + 0.5, sampling_depth);
    // ---- early out of a wavefront with nothing but sky (DepthBounds 0, 1)
    let skip_pixel = sampling_depth >= NEAR_DEPTH || sampling_depth <= FAR_DEPTH;
    if (!skip_pixel) {
        atomicMax(&any_live, u32(reach));
    }
    let wave_reach = workgroupUniformLoad(&any_live);
    if (wave_reach == 0u) {
        return;
    }
    // Only wavefronts near enough to need a longer physical ray load more
    // than the original two blocks. Scalar state avoids per-thread arrays.
    let reads = i32(wave_reach) / WAVE_SIZE + 2;
    for (var i = 0; i < BASE_READ_COUNT; i++) {
        let read_xy = vec2<i32>(floor(pixel_xy));
        let minor_axis = select(pixel_xy.x, pixel_xy.y, x_axis_major);
        let bilinear = fract(minor_axis) - 0.5;
        let bias = select(-1, 1, bilinear > 0.0);
        let offset_xy = select(vec2<i32>(bias, 0), vec2<i32>(0, bias), x_axis_major);
        let d0 = load_depth(read_xy);
        let d1 = load_depth(read_xy + offset_xy);
        let use_point_filter = abs(d0 - d1) > abs(FAR_DEPTH - d0) * params.tuning.y;
        let shadow_depth = select(d0 + abs(d0 - d1) * z_sign, d0, use_point_filter);
        let distance = pixel_distance + f32(WAVE_SIZE * i) * direction;
        var stored_depth = (shadow_depth - params.light.z) / distance;
        if (i != 0) {
            stored_depth = select(1e10, stored_depth, distance > 0.0);
        }
        depth_data[i * WAVE_SIZE + i32(tid)] = stored_depth;
        pixel_xy += xy_delta * direction;
    }
    if (wave_reach > u32(SAMPLE_COUNT)) {
        for (var i = BASE_READ_COUNT; i < READ_COUNT; i++) {
            if (i < reads) {
                let read_xy = vec2<i32>(floor(pixel_xy));
                let minor_axis = select(pixel_xy.x, pixel_xy.y, x_axis_major);
                let bilinear = fract(minor_axis) - 0.5;
                let bias = select(-1, 1, bilinear > 0.0);
                let offset_xy = select(vec2<i32>(bias, 0), vec2<i32>(0, bias), x_axis_major);
                let d0 = load_depth(read_xy);
                let d1 = load_depth(read_xy + offset_xy);
                let use_point_filter = abs(d0 - d1) > abs(FAR_DEPTH - d0) * params.tuning.y;
                let shadow_depth = select(d0 + abs(d0 - d1) * z_sign, d0, use_point_filter);
                let distance = pixel_distance + f32(WAVE_SIZE * i) * direction;
                var stored_depth = (shadow_depth - params.light.z) / distance;
                if (i != 0) {
                    stored_depth = select(1e10, stored_depth, distance > 0.0);
                }
                depth_data[i * WAVE_SIZE + i32(tid)] = stored_depth;
            }
            pixel_xy += xy_delta * direction;
        }
    }
    workgroupBarrier();
    if (wave_reach > u32(SAMPLE_COUNT)) {
        if (i32(tid) < reads * WAVE_SIZE / SPAN) {
            let first = i32(tid) * SPAN;
            var bounds = vec2<f32>(1e30, -1e30);
            for (var j = 0; j < SPAN; j++) {
                let value = depth_data[first + j];
                bounds = vec2<f32>(min(bounds.x, value), max(bounds.y, value));
            }
            depth_ranges[tid] = bounds;
        }
        workgroupBarrier();
    }
    if (skip_pixel) {
        return;
    }

    let start_depth = (sampling_depth - params.light.z) / pixel_distance;
    let sample_index = i32(tid) + 1;
    var shadow_value = vec4<f32>(1.0);
    var hard_shadow = 1.0;
    // the shadowing window, in units of SurfaceThickness of the depth left
    // to the far plane (at least one pixel near the light point)
    let depth_scale = min(pixel_distance + direction, 1.0 / params.tuning.x) * pixel_distance / abs(FAR_DEPTH - sampling_depth);
    let window = ShadowWindow(
        start_depth * depth_scale,
        depth_scale,
        z_sign,
        pixel_distance,
        direction,
        max(1.0, params.tuning.w * sampling_depth),
    );
    // sample i lies i + 1 pixels along the ray
    for (var i = 0; i < HARD_SHADOW_SAMPLES; i++) {
        hard_shadow = min(hard_shadow, depth_delta(window, depth_data[sample_index + i], i + 1));
    }
    // the bulk, in four groups averaged at the end
    for (var i = HARD_SHADOW_SAMPLES; i < SAMPLE_COUNT - FADE_OUT_SAMPLES; i++) {
        shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta(window, depth_data[sample_index + i], i + 1));
    }
    for (var i = SAMPLE_COUNT - FADE_OUT_SAMPLES; i < SAMPLE_COUNT; i++) {
        let fade_out = f32(i + 1 - (SAMPLE_COUNT - FADE_OUT_SAMPLES)) / f32(FADE_OUT_SAMPLES + 1) * 0.75;
        shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta(window, depth_data[sample_index + i], i + 1) + fade_out);
    }
    // Keep distant shading identical; only close pixels visit the extension.
    // Samples remain contiguous so thin jewellery and straps cannot fall through.
    if (reach > SAMPLE_COUNT && hard_shadow > 0.75 && any(shadow_value > vec4<f32>(0.75))) {
        for (var i = (SAMPLE_COUNT - FADE_OUT_SAMPLES) / 4 * 4; i < reach;) {
            // ---- conservative span skip
            let block = (sample_index + i) / SPAN;
            let advance = min((block + 1) * SPAN - sample_index - i, reach - i) / 4 * 4;
            if (advance >= 4 && span_lower_bound(window, depth_ranges[block], f32(i + 1), f32(i + advance)) > max(max(shadow_value.x, shadow_value.y), max(shadow_value.z, shadow_value.w)) + 0.0001) {
                i += advance;
                continue;
            }
            // ---- end span skip
            let k = vec4<f32>(f32(i)) + vec4<f32>(1.0, 2.0, 3.0, 4.0);
            let sample_depths = vec4<f32>(
                depth_data[sample_index + i], depth_data[sample_index + i + 1],
                depth_data[sample_index + i + 2], depth_data[sample_index + i + 3]);
            let v = (sample_depths * window.scale - window.start) * -window.z_sign;
            var delta = abs(1.0 - v);
            if (window.depth > 1.0 && window.start * window.direction * window.z_sign > 0.0) {
                let distance_k = max(vec4<f32>(window.distance) + k * window.direction, vec4<f32>(1.0));
                let receding = window.start * k * window.direction * window.z_sign / distance_k;
                delta = max(1.0 - v, v - clamp(receding, vec4<f32>(1.0), vec4<f32>(window.depth)));
            }
            let fade = max(vec4<f32>(0.0), k - f32(reach - FADE_OUT_SAMPLES)) / f32(FADE_OUT_SAMPLES + 1) * 0.75;
            shadow_value = min(shadow_value, select(vec4<f32>(1.0), delta + fade, k <= vec4<f32>(f32(reach))));
            if (all(shadow_value <= vec4<f32>(0.75))) { break; }
            i += 4;
        }
    }
    // ---- full-view correctness reference
    if (FULL_VIEW_REFERENCE && hard_shadow > 0.75 && any(shadow_value > vec4<f32>(0.75))) {
        let origin = mix(start_xy, end_xy, thread_step / f32(WAVE_SIZE));
        let full_reach = visible_ray_reach(origin, xy_delta * direction / f32(WAVE_SIZE), pixel_distance, direction);
        let wave = Wave(start_xy, end_xy, xy_delta, x_axis_major, reverse_direction, major_axis_start, ma_light_frac, direction, z_sign);
        // Revisit the bounded prefix's fading tail without fading it. Its
        // end is no longer the end of the ray, including when it exceeds 480.
        for (var i = max(0, min(reach, full_reach) - FADE_OUT_SAMPLES) / 4 * 4; i < full_reach; i += 4) {
            let samples = vec4<f32>(
                stored_depth(wave, sample_index + i), stored_depth(wave, sample_index + i + 1),
                stored_depth(wave, sample_index + i + 2), stored_depth(wave, sample_index + i + 3));
            let delta = vec4<f32>(
                depth_delta(window, samples.x, i + 1), depth_delta(window, samples.y, i + 2),
                depth_delta(window, samples.z, i + 3), depth_delta(window, samples.w, i + 4));
            let valid = vec4<i32>(i) + vec4<i32>(1, 2, 3, 4) <= vec4<i32>(full_reach);
            shadow_value = min(shadow_value, select(vec4<f32>(1.0), delta, valid));
            if (all(shadow_value <= vec4<f32>(0.75))) { break; }
        }
    }
    // ---- end full-view correctness reference
    // a sample on the reference depth is full shadow; the contrast widens it
    let contrast = params.tuning.z;
    shadow_value = saturate(shadow_value * contrast + (1.0 - contrast));
    hard_shadow = saturate(hard_shadow * contrast + (1.0 - contrast));
    let result = min(hard_shadow, dot(shadow_value, vec4<f32>(0.25)));
    let size = vec2<i32>(textureDimensions(out_tex));
    if (all(write_xy >= vec2<i32>(0)) && all(write_xy < size)) {
        textureStore(out_tex, write_xy, vec4<f32>(1.0 - result, 0.0, 0.0, 0.0));
    }
}

// Every pixel lit, before the wavefronts (wgpu's clear_texture needs a
// device feature).
@compute @workgroup_size(8, 8, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_tex);
    if (id.x < size.x && id.y < size.y) {
        textureStore(out_tex, vec2<i32>(id.xy), vec4<f32>(0.0));
    }
}
