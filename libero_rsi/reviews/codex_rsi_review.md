# LIBERO RSI stack review

**Date:** 2026-10-18 (America/New_York)  
**Requested via:** Codex CLI `gpt-6-astra` / `model_reasoning_effort=high`  
**Execution note:** `codex exec -m gpt-6-astra -c model_reasoning_effort="high"` was launched correctly, but ChatGPT backend returned persistent **HTTP 403** (`workspace routing discovery failed`). This document is a **local code-grounded fallback review** of the same scope, with findings tied to symbols/lines in the tree. Do **not** treat it as an Astra-generated Codex transcript.

**Scope read:**
- [`run_ep_fbpatch.py`](../run_ep_fbpatch.py)
- `rsi_playground_ee.py` (private bench; not in this snapshot)
- [`ee_pg_easy_v2_lib.json`](../libs/ee_pg_easy_v2_lib.json)
- [`validate.py`](../validate.py)
- [`dsl_libero.py`](../dsl_libero.py)

---

## 1. Executive summary

- Recent work (goal ownership, yaw preserve, multi-action stall feedback, identity escape, relook context, `ASTRA_EFFORT=xhigh`) is directionally right and mostly wired end-to-end.
- The biggest remaining failure mode is **control signal vs diagnosis gap**: neighbour collisions and bad grasp height often never light `descent_hit` / usable `descent_stalled`, so recoveries and slow-think stall menus do not fire when the video clearly shows a problem.
- **Lessons still do not affect Jev.** Only predicates / recoveries / `intent_instructions` change behaviour. Overnight video reflection that mostly writes prose lessons will look “successful” while changing nothing.
- **Relook vs PATCH** is better, but still fragile: noisy `grasp_yaw_deg` (≥15° from prior PER) counts as “explicit” and can wipe a working PATCH yaw; slow-think issue enum is missing the neighbour-block classes that the applicator already understands.
- Stack is **not Long-ready**: no joint/drawer/microwave intents, no multi-stage goal handoff, place targets only via single place keypoint, horizon/decision caps sized for short Object/Spatial.
- Do not restart playground until short-horizon stall→target-update is reliable on the failing cream-cheese / ketchup / butter inits.

---

## 2. Concrete bugs

### P0 — `descent_stalled` can miss real stalls when the grasp estimate is bad

**Where:** `ee_pg_easy_v2_lib.json` predicate `descent_stalled`; evaluated in `namespace()` → recoveries in `run_ep_fbpatch.py` ~L1252+.

**Expr (current):**
```text
prev == 'descend_to_grasp' and not closed_cmd and not contact
and dist(tcp, grasp) < 1.5*obj_h
and (n_same >= 2 or last_result == 'stalled' or last_result == 'timeout')
```

**Bug:** If Astra’s grasp point is wildly off (or TCP stopped far from that estimate after hitting a neighbour / table), `dist(tcp, grasp) >= 1.5*obj_h` makes the predicate **false** even when `last_result == 'stalled'`. Then `hp_descent_stalled_feedback` never fires. The sibling rule `repeated_contactless_descent_feedback` explicitly requires `not descent_stalled`, so a “should-be-stalled but predicate false” case can also miss that path depending on contact flags.

**Evidence from recent retunes:** ketchup / cream failures where stall feedback was sparse despite horizon stalls.

**Fix direction:** Split “servo reported stall/timeout after descend” (always true when `prev==descend` and `last_result in {stalled,timeout}`) from “near grasp but no contact”. Do not gate the former on `dist(tcp, grasp)`.

---

### P0 — Lessons are non-executive by design; reflection often only writes them

**Where:** Jev call site `run_ep_fbpatch.py` ~L1291–1299 (passes `observed_facts`, `predicate_meanings`, intent definitions — **no lessons**). Documented in `rsi_libero.py` / `rsi_libero_v2.py`: lessons “not shown to Jev”. Playground `rsi_playground_ee.py` still heavily prompts `add_lesson` / `revise_lesson`.

**Bug (product, not typo):** Video reflection can “succeed” (library grows) while behaviour is unchanged unless it also emits `add_recovery` / `add_predicate` / `modify_recovery` with `force_intent` / `stall_feedback` / `approach_change` / `shape`. Current `ee_pg_easy_v2_lib.json` has **14 lessons** and only **5 recoveries** (mostly human priors).

**Fix direction:** Make failure reflection **reject** lesson-only edits when the episode had stall/identity/close-refuse evidence; require at least one executable recovery/predicate op (playground prompt already hints this — enforce in apply gate).

---

### P1 — Slow-think `issue` enum ≠ applicator’s neighbour-block classes

**Where:**
- `SLOW_PROMPT` ~L809: `"ok" | "wrong_object" | "tip_push" | "empty_grasp" | "place_miss" | "other"`
- `_apply_episode_plan` ~L858–886: treats `blocked_by_neighbour`, `object_pushed_by_descent` as stall issues (preserve identity, force non-relook feedback)
- Fallback prompt ~L773 **does** include those failure classes

**Bug:** Mid-episode slow-think cannot honestly label neighbour block; it must say `other`. Identity-preserve / plan-normalize paths that key off the precise strings are weaker on the path that actually runs most often (async/sync slow-think), while fallback (rarer) has the right vocabulary.

**Fix:** Align `SLOW_PROMPT` issue enum with fallback `failure_class` (at least add `blocked_by_neighbour`, `object_pushed_by_descent`).

---

### P1 — Explicit-yaw wipe still possible after “preserve”

**Where:** `_new_keypoints_explicit_yaw` / `_restore_approach_pose` / `_commit_keypoints` ~L208–261; `YAW_EXPLICIT_EPS_DEG = 15`.

**Bug:** Preserve compares **new keypoint yaw vs prior PER yaw**, not vs current **commanded** PATCH yaw. If PER yaw stays ~0 and Astra relook returns a noisy `grasp_yaw_deg` of 20°, `explicit_new_yaw=True` → confirmed PATCH yaw (e.g. −45 from slow-think) is **not** restored. This reintroduces the “turned then snapped back” failure under perception noise.

**Fix:** Treat as explicit only if new yaw differs from **`_cmd_yaw_deg()`** (PATCH/RSHAPE/PER chain) by ≥ eps **and** Astra’s relook context acknowledges a reason; or require `|new - cmd| >= eps` with confidence, else keep PATCH.

---

### P1 — `descent_hit` / contact-gated clears are deaf to neighbour bumps without pad flags

**Where:** predicates `descent_hit`, recoveries `hp_descent_hit_clear`; stall menu uses `left_contact`/`right_contact`.

**Bug:** User videos show side collisions that often leave `contact` / unilateral pads at 0. Then clear/push paths never arm; policy only sees generic stall or empty relook.

**Fix:** Add a predicate on **servo stall + XY progress toward grasp + no bilateral contact** (geometry), and/or object-body displacement diagnostic already computed post-episode (`moved_most`) fed into reflection — for online control, prefer TCP stall + height below expected clear without requiring pad contact.

---

### P1 — `push_aside` is prompt-only for slow-think; not in `_choose_stall_feedback`

**Where:** `_choose_stall_feedback` ~L467–487; `_apply_stall_feedback` ~L508–589 (`else: unknown_action`); `push_aside()` ~L784–794.

**Bug:** Library/slow prompts advertise `push_aside`, but automatic stall feedback never selects it. Only Astra plan steps invoke `push_aside()`. If slow-think keeps returning relook/adjust, neighbours stay. Also `push_aside` forcibly `grip(1)` — risky if identity wrong.

**Fix:** When unilateral contact or `blocked_by_neighbour`, prefer `clear_reapproach` then allow one `push_aside` if Astra supplies `blocker_px`; keep push episode-only and identity-gated.

---

### P2 — Identity-bad plan rewriter drops stall steps but can still spend relook budget

**Where:** `_apply_episode_plan` ~L862–865: on `id_bad`, forces `plan = [relook] + …` and `_reset_patch`.

**Risk:** Correct for wrong object, but if Astra flickers `matched_language_goal=null` during a confirmed progressive grasp (`stall_issue` false because issue=`other`), identity is cleared and PATCH wiped (including yaw). Coupled with P1 enum gap, `other` during neighbour stall is dangerous.

**Fix:** If `goal_id_confirmed()` already True and issue in stall-like set (including `other` + `last_result in stalled/timeout`), do **not** clear identity on null mlg alone.

---

### P2 — Playground single-ep / retune runners skip video reflection

**Where:** `rsi_playground_ee.py` owns `call_video` updates; ad-hoc `run_ep_fbpatch.py` retunes do not.

**Bug:** Operational, not harness: retunes look like “Astra never reflected.” Expected if runner ≠ playground.

**Fix:** Shared post-episode hook: on failure or slow/fallback intervene, always call the same update path (or explicitly log `reflection=skipped:runner`).

---

### P2 — `validate.py` allows recoveries without proving predicates fire in-episode

Validator checks dims / required control keys (`force_intent` | `shape` | `approach_change` | `stall_feedback`) but cannot catch “trigger never true on real stalls.” Need a tiny offline fixture: synthetic `ns` with `last_result='stalled'` and far TCP → expect recovery trigger true after predicate fix.

---

## 3. Risky interactions

| Interaction | What happens | Risk |
|---|---|---|
| Relook keypoint refresh vs PATCH | `_commit_keypoints` resets position offsets; yaw/tilt restored unless “explicit” new yaw | Noisy yaw → wipe working approach (P1) |
| Slow-think `adjust` after keypoint commit | Keypoints first (relook priority), then adjust mutates PATCH | OK if adjust present; empty plan + keypoints alone can clear dz offsets |
| Stall + identity confirmed + empty relook | `_apply_episode_plan` skips empty relook and forces `_apply_stall_feedback` | Good; depends on `stallish` which needs issue/`last_result` |
| Recovery `stall_feedback` + `force_intent` | stall branch `break`s after preferring feedback’s `force_next_intent` | OK for current lib (all `move_above_object`); watch confirm_contact→close vs later gates |
| Identity refuse close → relook budget → escape | `_escape_identity_deadlock` multi-action + escape look | Better than noop; can still burn horizon if escape looks unmatched |
| Async slow-think vs sync | Async discarded after `SLOW_STALE_INTENTS`; sync on stage-fail / after close | Interventions can arrive one intent late |
| Lessons vs recoveries | Lessons stored, never in Jev prompt | Reflection theater (P0) |
| Fallback vs slow issue vocab | Fallback has neighbour classes; slow does not | Diagnosis skew |

---

## 4. Experience-library gaps (`ee_pg_easy_v2_lib.json` v `ee_pg_v13`)

- **14 lessons / 5 recoveries** — skew toward non-executive text; several lessons explicitly say interventions remain “untested / not established.”
- Human prior about clearing neighbours exists in lesson space more than in executable recoveries (no recovery trigger on “stalled + neighbour geometry”).
- `descent_stalled` distance gate (above) is the main executable hole.
- No recovery that says: *identity confirmed ∧ last_result stalled ∧ empty_close history → forbid close until bilateral contact* beyond intent_instructions prose.
- `max_uses` on stall recoveries (2–3) can exhaust mid-episode while Jev continues to pick `descend`/`close`.

---

## 5. Long-horizon gaps

- Intent menu is short pick-and-place only (`move_above_object` … `finish`); **no** open/close drawer, microwave, hinge, or toggle.
- Single language goal + single grasp/place keypoint — **no** stage pointer / todo (“object A done → rebind language goal B”).
- Identity gate is episode-global; after placing A, held-identity / matched-goal logic is not reset for object B.
- `HORIZON=600`, `MAX_DEC=60` — likely tight for LIBERO Long / `libero_10`.
- Place success depends on Astra `place_px`; no library intents for “stove”, “cabinet”, etc. beyond whatever place keypoint means.
- Joint progress cannot be expressed in current DSL predicates (TCP/objects only).

**Recommendation:** Do not start Long until short Object stalls reliably update the shared goal and neighbour-clear works on at least cream/ketchup fail inits.

---

## 6. Prioritized fixes

1. **P0 — Predicate: servo-stall after descend always true**  
   Add `descend_servo_stalled := prev=='descend_to_grasp' and last_result in {stalled,timeout} and not closed_cmd` (no distance gate). Point `hp_descent_stalled_feedback` at it.  
   *Impact:* recoveries actually fire on the failures you are watching.

2. **P0 — Enforce executable reflection**  
   On fail / intervene episodes, reject library updates that only touch lessons; require recovery/predicate op. Optionally distill one general `stall_feedback` recovery from successful yaw/raise evidence (relative, not `yaw=90`).  
   *Impact:* overnight learning stops being dead letters.

3. **P1 — Align slow-think issue enum + soften null-mlg wipe under stall**  
   Add neighbour-block issues; if already `goal_id_confirmed` and stallish, keep identity on null mlg.  
   *Impact:* fewer false identity resets; clearer push/clear plans.

4. **P1 — Explicit yaw vs commanded yaw**  
   Compare new keypoint yaw to `_cmd_yaw_deg()`, not only PER.  
   *Impact:* stops “turned then snapped back.”

5. **P1 — Geometry-based clear without pad contact**  
   Stall + below clear height + no bilateral contact → `clear_reapproach` / raise.  
   *Impact:* neighbour bumps that never set contact flags.

6. **P2 — Log `reflection=skipped` for non-playground runners; optional shared hook**

7. **Later — Long todo:** stage object, joint intents, per-stage identity reset, higher horizon — only after short-horizon P0/P1.

---

## 7. Suggested smoke tests (do not run playground)

1. Synthetic `ns`: `prev=descend_to_grasp`, `last_result=stalled`, `dist(tcp,grasp)=3*obj_h`, no contact → `descend_servo_stalled` true; recovery would select stall_feedback.
2. Unit: PATCH yaw=−45; relook keypoints yaw=0 (PER was 0) → after `_commit_keypoints`, PATCH yaw still −45.
3. Unit: PATCH yaw=−45; relook keypoints yaw=25 (noise) with preserve-vs-cmd rule → still −45 unless flagged explicit.
4. Slow-think fixture: `issue=blocked_by_neighbour`, `matched_language_goal=true`, plan=`[relook]` → applicator forces non-relook stall step.
5. Identity refuse ×3 with relook budget 0 → `_escape_identity_deadlock` sets non-`close_gripper` `force_next_intent` (no horizon noop-only path before escape).
6. Manual single-ep on cream `i1` / ketchup `i1` with `ASTRA_EFFORT=xhigh` after (1)+(4); check log for `stall_feedback` / `approach_pose_preserve` / no yaw snap-back.

---

## 中文摘要（给用户）

- 审阅请求是用 Codex **`gpt-6-astra` + effort `high`** 发起的，但盒子访问 ChatGPT 一直 **403**，所以这篇是**同范围的本地代码审阅**，不是 Astra 会话原文。
- 最大洞：下降卡住时，若抓取点估偏了，`descent_stalled` 可能一直为假，恢复规则根本打不着。
- 反思写再多 **lesson 也进不了 Jev**；能改行为的只有谓词/恢复规则——库里课很多、可执行恢复很少。
- 慢思考的问题标签里**没有「邻物挡住」**，和应用层已经认识的类别对不上，清障很难被正确触发。
- 保留 yaw 的逻辑仍可能被**噪声 yaw** 当成「明确新角度」而清掉，角度还会打回原形。
- 邻物碰撞经常**没有触点信号**，靠接触才触发的清障等于聋了。
- 长程还缺：多阶段目标、开关抽屉/微波等关节动作、更大步数预算——短程卡住没稳前不建议开 Long。
- 建议先改：放开「伺服 stalled 必真」谓词 → 强制失败反思产出可执行 recovery → 对齐慢思考邻物标签并按当前指令 yaw 做保留判断。

