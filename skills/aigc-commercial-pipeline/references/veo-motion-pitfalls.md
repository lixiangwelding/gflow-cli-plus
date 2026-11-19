# Veo Commercial Motion Pitfalls & Surgical Fixes

Real-world failure modes observed during commercial hardware/product production on Google Flow (`veo-fast` `i2v` & `--end-frame`), paired with proven keyframe and prompt constraints.

---

## 1. Mid-Air Laser / Light-Beam Jumps (Surface Detachment)

### Symptom on 3-Frame QA Strip
- **Start (`0.4s`):** Laser line or optical projection sits cleanly against the wall/floor as drawn in the `nano2` start frame.
- **Mid / End (`2.5s – 4.6s`):** As the camera moves or talent interacts, the laser beam detaches from the wall plane and floats through mid-air like a sci-fi lightsaber or volumetric fog cone.

### Root Cause
Words like *"laser beam shooting across the room"*, *"projecting rays into space"*, or *"glowing hologram line"* bias Veo toward 3D volumetric air beams rather than surface-grazing optics.

### Fix (Keyframe + Prompt Constraint)
1. **Start/End Keyframe:** Render the laser strictly as a razor-thin contact line intersecting wall/tile edges, with zero smoky air haze.
2. **Prompt Constraint:** Explicitly lock both the emitter housing and the projected line onto the physical wall plane:
   > *"The device stays flush-mounted flat against the matte plaster wall; its thin crisp green alignment line stays strictly painted flat onto the wall surface with zero mid-air volumetric beam, zero floating rays, and zero fog."*

---

## 2. Foreground Camera Occlusion on Lateral Pans / Truck Moves

### Symptom on 3-Frame QA Strip
- **Start (`0.4s`):** Hero product and talent are clearly framed.
- **Mid / End:** During a lateral truck/pan (`pan right`, `orbit`, `slider move`), Veo invents an out-of-focus foreground wall edge, doorframe, curtain, or shoulder that wipes across 50–100% of the lens, ruining the usable duration of the shot.

### Root Cause
Lateral translation prompts (*"camera slides sideways past..."*, *"cinematic parallax pan"*) strongly trigger Veo's depth-parallax prior, causing it to hallucinate near-field occluders between camera and subject.

### Fix
1. **Replace wide lateral trucks with micro-dolly or pedestal moves** when the hero product must stay continuously legible.
2. **Add explicit clear-sightline negatives/constraints:**
   > *"Smooth subtle dolly-in with a completely unobstructed foreground sightline; nothing passes between the camera lens and the subject, no foreground doorframe, no pillar wipe, no out-of-focus foreground objects."*

---

## 3. Small Wall Switches / Outlets Warping or Stealing Motion in `i2v`

### Symptom on 3-Frame QA Strip
- Background light switches, power sockets, or small wall plates near the hero product shimmer, double-vision, sprout extra buttons, or attract unintended hand interactions during `i2v`.

### Root Cause
Veo allocates local motion attention to high-contrast rectangular micro-features on otherwise flat walls. Prompting *"ignore the wall switch"* in `i2v` rarely works once the pixels exist in `initial_frame`.

### Fix (Mandatory Pre-`i2v` Clean-Plate Pass)
- **Fix the still BEFORE calling `video i2v`:** Inspect every `Sxx_start.png`. If any small switch, outlet, or stray fixture sits on the wall, run a zero-credit `gflow image i2i` pass to remove it first:
  > *"Keep the exact product, framing, and lighting from the reference image, seamlessly remove the small wall light switch/outlet so the surrounding plaster wall is completely clean and uninterrupted."*

---

## 4. Melting, Spelling-Drift, or Wobbling 3D Chinese Title Cards

### Symptom on 3-Frame QA Strip
- **Start (`0.4s`):** Crisp Chinese brand logo and slogan from the `nano2` keyframe.
- **Mid / End:** Strokes swim, radicals merge or mutate into pseudo-characters, or 3D bevels bend like jelly during camera movement.

### Root Cause
Large camera rotations (`orbit around 3D text`), particle explosions *"forming text"*, or asking `t2v` to generate Chinese typography from scratch exceed video diffusion glyph stability.

### Fix
1. **Never use `t2v` for Chinese brand lockups.** Render the exact 3D metallic/glass Chinese title card in `nano2` (`image t2i` / `image i2i`) and verify every stroke at full resolution first.
2. **Animate via locked `i2v` with lighting-only motion + micro push-in (<3%):**
   > *"Animate the attached title card as the exact locked layout: all 3D Chinese characters and logo geometry remain 100% rigid, stationary, and rock-steady in place with zero text morphing or stroke deformation, while a single subtle warm specular light sweep glides smoothly across the brushed metal bevels."*
3. **Dual-frame lock (optional):** Pass the **same** approved title still (or a pure specular-shifted twin) as both `--initial-frame` and `--end-frame` when absolute end-hold stillness is required for packshots.

---

## 5. Quick 3-Frame Strip Triage Matrix

| Strip Observation (`0.4s` \| `mid` \| `end-0.35s`) | Diagnosis | Action |
|---|---|---|
| Frame 1 has wall clutter / bad product proportions | Bad `i2v` seed frame | Fix in `nano2` `image i2i` first (costs 0 credits) |
| Frame 1 sharp, Frame 2–3 product warps | Excessive camera rotation or fast hand crossing product | Switch to slower linear dolly + lock rigid body phrasing, or add `--end-frame` |
| Frame 3 darkened / wiped by blur | Foreground parallax occluder | Remove `"parallax"` / `"orbit"`; add `"unobstructed foreground sightline"` |
| Beam/reflection detaches into room air in Frame 2 | Volumetric light prior triggered | Constrain optics `"strictly flush on the wall surface, zero mid-air beam"` |
| Audio has random spoken gibberish on B-roll | Veo invented background dialogue | Add `"ambient room tone and subtle mechanical click only, no speech"` or mute stem (`volume=0`) in Stage 5 mix |
