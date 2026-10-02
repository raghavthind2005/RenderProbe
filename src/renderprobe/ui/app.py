"""Gradio UI for RenderProbe (core three tabs).

Tabs:
  * **Run**     - pick scene / probe / models / params / seeds, run, see the
                  per-condition results table and the accuracy / visual-gain /
                  oracle summary tables. Carries the API-key panel: a key entered
                  there is bound to that one run through ``core.credentials``, so it
                  never reaches ``os.environ`` and never leaks into another session.
  * **Sweep**   - accuracy-vs-factor curve (with bootstrap-CI band) per model.
  * **Inspect** - browse the individual wrong-or-errored cases: the scene image
                  next to the model's raw answer, its parse, and ground truth.

gradio (and matplotlib, for the sweep plot) are imported lazily inside the
builders so ``import renderprobe.ui`` works without them installed. All the
non-trivial logic lives in ``data.py`` and is unit-tested there; this module is
deliberately thin wiring.
"""
from __future__ import annotations

from renderprobe import renderers
from renderprobe.core import credentials
from renderprobe.core.registry import Registry
from renderprobe.core.runner import ExperimentConfig, run_experiment_collect
from renderprobe.ui import data as D

_DEFAULT_ANALYZERS = [
    "accuracy", "visual_gain", "decomposition", "oracle", "taxonomy",
    "calibration", "sweep_curve",
]

# Size of the fixed pool of param sliders the Run tab shows/hides per scene. Our
# scenes expose at most 2 knobs; 3 leaves headroom for a third-party scene.
_MAX_PARAMS = 3


def _provider_label(env: str) -> str:
    """Dropdown label for a credential variable: who serves it and what it costs."""
    info = D.provider_info(env)
    return f"{info['provider']} ({'free tier' if info['free'] else 'paid'})"


def _seed_list(text: str) -> list[int]:
    """Parse '0,1,2' (or '0 1 2') into a list of ints; default [0]."""
    parts = [p for chunk in text.split(",") for p in chunk.split()]
    seeds = [int(p) for p in parts if p.strip()]
    return seeds or [0]


def _parse_sweep_values(text: str, specs: list, name: str) -> list:
    """Parse a comma/space-separated sweep list, cast to the named param's type."""
    typ = next((s["type"] for s in specs if s["name"] == name), "float")
    out: list = []
    for tok in text.replace(",", " ").split():
        try:
            out.append(int(round(float(tok))) if typ == "int" else float(tok))
        except ValueError:
            pass
    return out


def _scene_to_image(scenes: list) -> dict[str, object]:
    """Map scene_id -> its first PIL image, for the inspect tab."""
    out: dict[str, object] = {}
    for s in scenes:
        if s.images:
            out[s.id] = s.images[0]
    return out


def build_app(registry: Registry | None = None):
    """Construct and return the gradio Blocks app (does not launch it)."""
    import gradio as gr

    if registry is None:
        registry = Registry()
        registry.autodiscover()

    scenes = registry.list_scenes()
    probes = registry.list_probes()
    opening_scene = D.default_scene(registry)
    key_envs = D.key_envs(registry)
    analyzers = [a for a in _DEFAULT_ANALYZERS if a in registry.list_analyzers()]

    def _update_controls(scene_name: str):
        """On scene change: reconfigure the slider pool (label/range/default/visibility)
        from the scene's params_schema, and repopulate the sweep-parameter choices."""
        specs = []
        try:
            specs = D.param_specs(registry.get_scene(scene_name))
        except Exception:
            specs = []
        updates = []
        for i in range(_MAX_PARAMS):
            if i < len(specs):
                sp = specs[i]
                updates.append(gr.update(
                    label=sp["name"], minimum=sp["min"], maximum=sp["max"],
                    step=sp["step"], value=sp["default"], visible=True))
            else:
                updates.append(gr.update(visible=False))
        sweep_update = gr.update(
            choices=["(none)"] + [s["name"] for s in specs], value="(none)")
        return (*updates, sweep_update)

    # Previews already drawn, keyed by what was asked for. A true-3D preview is a CPU
    # path trace of a few seconds, and dragging a dial back to a value already seen
    # should not pay for it twice. Small and bounded: this is a convenience, not a store.
    _preview_cache: dict[tuple, tuple] = {}
    _PREVIEW_CACHE_MAX = 8

    def _preview(scene_name: str, *args):
        """Render the scene at the current dial values + first seed, and show the
        image alongside the exact question and ground-truth the probe will use."""
        *slider_vals, seeds_text = args
        if not scene_name:
            return None, ""
        cache_key = (scene_name, tuple(slider_vals), seeds_text)
        if cache_key in _preview_cache:
            return _preview_cache[cache_key]
        try:
            gen = registry.get_scene(scene_name)
            specs = D.param_specs(gen)
            params = D.params_from_values(gen, list(slider_vals)[:len(specs)])
            seed = _seed_list(seeds_text)[0]
            with renderers.using_registry(registry):   # resolve --plugin-dir renderers too
                scene = gen.generate(params, seed)
            probe_name = D.compatible_probe(registry, scene_name, probes)
            probe = registry.get_probe(probe_name)
            question = probe.question(scene)
            gt = probe.ground_truth(scene)
            img = scene.images[0] if scene.images else None
            md = (
                f"**Question** _(probe: `{probe_name}`)_\n\n> {question}\n\n"
                f"**Ground truth:** `{gt}`  \n"
                f"**Params:** `{params}`, "
                f"**tier:** `{scene.meta.get('tier', 'n/a')}`, seed `{seed}`"
            )
            if len(_preview_cache) >= _PREVIEW_CACHE_MAX:
                _preview_cache.pop(next(iter(_preview_cache)))
            _preview_cache[cache_key] = (img, md)
            return img, md
        except Exception as exc:
            return None, f"Could not generate this scene: {exc}"

    # Telemetry is off on a page that has credential fields on it. gradio's analytics
    # send component metadata rather than values, but a tool that asks for somebody's
    # API key should not also be making network calls they did not ask for.
    with gr.Blocks(title="RenderProbe", analytics_enabled=False) as app:
        gr.Markdown(
            "# RenderProbe\n"
            "**Render scenes with known ground truth -> probe any VLM -> localize "
            "*where* it fails: perception, grounding, or reasoning.**\n\n"
            "The *oracle* condition hands the model the ground-truth primitives as "
            "text; the *report* condition asks it to read those primitives off the "
            "image. Together they split a failure three ways - **encoding** (can't "
            "see it), **grounding** (sees it but doesn't use it), or **reasoning** "
            "(fails even given the facts). `recovery = acc_oracle - acc_full` bounds "
            "the visual-pathway share; the report probe splits it (see the "
            "Decomposition table). Recovery carries a paired McNemar test + bootstrap "
            "CI. Keys are entered below and held for this browser session only. "
            "See docs/methodology.md."
        )

        # State carried between tabs.
        st_results = gr.State([])
        st_reports = gr.State([])
        st_images = gr.State({})

        with gr.Tab("Run"):
            with gr.Row():
                scene_dd = gr.Dropdown(
                    scenes, value=opening_scene, label="Scene"
                )
                probe_dd = gr.Dropdown(
                    probes,
                    value=(
                        D.compatible_probe(registry, opening_scene, probes)
                        if opening_scene else None
                    ),
                    label="Probe (auto-paired to the scene)",
                )
            runnable_now = D.default_model_selection(registry)
            with gr.Group():
                model_cbg = gr.CheckboxGroup(
                    D.model_choices(registry),
                    value=runnable_now,
                    label="Models to run",
                    info=("Only the models a key is present for. Each shows its "
                          "provider and what the last roster probe found."),
                )
                show_all_cb = gr.Checkbox(
                    value=False, label="Show models I have no key for")
                # One provider at a time rather than a field per provider. Six labeled
                # boxes and a six-row table said the same thing the model labels already
                # say, and pushed the scene preview off the bottom of the page.
                with gr.Accordion("API keys", open=not runnable_now):
                    with gr.Row():
                        provider_dd = gr.Dropdown(
                            choices=[(_provider_label(e), e) for e in key_envs],
                            value=key_envs[0] if key_envs else None,
                            label="Provider", scale=2)
                        key_box = gr.Textbox(
                            label="Key", type="password", scale=3,
                            placeholder="paste the key, then press Save")
                        save_btn = gr.Button("Save key", variant="secondary", scale=1)
                    provider_md = gr.Markdown(
                        D.provider_hint(key_envs[0]) if key_envs else "")
                    key_status_md = gr.Markdown(D.credential_markdown(registry))
                    gr.Markdown(
                        "A saved key is held in memory for your session only: never "
                        "written to disk, never logged, and never placed in the "
                        "server's environment, where another session's run would pick "
                        "it up. A provider left empty falls back to its environment "
                        "variable, which is what the CLI reads."
                    )
                # Session-scoped credentials: {env var -> key}. Bound around the run and
                # unbound after it, so a key never outlives the request that used it.
                st_keys = gr.State({})
            analyzer_cbg = gr.CheckboxGroup(
                analyzers, value=analyzers, label="Analyzers"
            )
            gr.Markdown(
                "**Scene controls** - drag the dials to shape the scene; the ranges "
                "come from the scene's own schema, and the preview below updates live."
            )
            _init_gen = registry.get_scene(opening_scene) if opening_scene else None
            init_specs = D.param_specs(_init_gen) if _init_gen else []
            param_sliders = []
            for i in range(_MAX_PARAMS):
                if i < len(init_specs):
                    sp = init_specs[i]
                    param_sliders.append(gr.Slider(
                        minimum=sp["min"], maximum=sp["max"], step=sp["step"],
                        value=sp["default"], label=sp["name"], visible=True))
                else:
                    param_sliders.append(gr.Slider(visible=False))
            with gr.Row():
                seeds_box = gr.Textbox(
                    value="0,1,2",
                    label="Seeds (comma-separated; preview uses the first)")
                elicit_cb = gr.Checkbox(
                    value=False, label="Elicit confidence (Calibration tab)")
                blind_cb = gr.Checkbox(
                    value=False,
                    label="Text-only baseline (Visual gain tab; doubles model calls)")
            with gr.Accordion(
                "Sweep one parameter (optional - enables the Sweep tab)", open=False
            ):
                with gr.Row():
                    sweep_dd = gr.Dropdown(
                        ["(none)"] + [s["name"] for s in init_specs],
                        value="(none)", label="Sweep parameter")
                    sweep_vals = gr.Textbox(
                        value="", label="Sweep values (e.g. 4,6,8)")
            gr.Markdown("### Scene preview - exactly what the model is shown")
            with gr.Row():
                preview_img = gr.Image(label="Scene", type="pil", height=300)
                preview_md = gr.Markdown()
            run_btn = gr.Button("Run experiment", variant="primary")
            status_md = gr.Markdown()
            results_df = gr.Dataframe(
                headers=D.RESULT_COLUMNS, label="Results (all conditions)", wrap=True
            )
            gr.Markdown("### Accuracy")
            acc_df = gr.Dataframe(headers=D.accuracy_rows(None)[0])
            gr.Markdown(
                "### Visual gain (acc_full - acc_blind)\n"
                "Empty unless *Text-only baseline* is enabled on the Run tab. It "
                "measures a model's prior over the answer set, so it earns its cost "
                "only where that set is small."
            )
            vg_df = gr.Dataframe(headers=D.visual_gain_rows(None)[0])
            gr.Markdown(
                "### Failure decomposition (three-way - the headline)\n"
                "Localizes each model's bottleneck: **encoding** (can't report the "
                "primitive from the image), **grounding/integration** (reports it but "
                "still fails the task), or **reasoning** (fails even given the facts "
                "as text). `recovery` bounds the visual-pathway share; `acc_report` "
                "splits it. See docs/methodology.md."
            )
            decomp_df = gr.Dataframe(headers=D.decomposition_rows(None)[0], wrap=True)
            gr.Markdown(
                "### Oracle - paired statistics behind `recovery` "
                "(recovery = acc_oracle - acc_full)"
            )
            orc_banner = gr.Markdown()
            orc_df = gr.Dataframe(headers=D.oracle_rows(None)[0], wrap=True)
            orc_note = gr.Markdown()
            gr.Markdown(
                "### Failure taxonomy\n"
                "*Why* answers were wrong, per condition. A category heavy under "
                "`full` but gone under `oracle` points to the visual pathway; one "
                "that persists under `oracle` points to reasoning."
            )
            tax_df = gr.Dataframe(headers=D.taxonomy_rows(None)[0], wrap=True)

            # On scene change: reconfigure the dials + sweep choices, auto-pair the
            # probe, then refresh the live preview (chained so it reads the new dials).
            _preview_inputs = [scene_dd, *param_sliders, seeds_box]
            scene_dd.change(
                _update_controls, inputs=scene_dd,
                outputs=[*param_sliders, sweep_dd],
            ).then(
                _preview, inputs=_preview_inputs, outputs=[preview_img, preview_md],
            )
            scene_dd.change(
                lambda s: D.compatible_probe(registry, s, probes),
                inputs=scene_dd, outputs=probe_dd,
            )
            # Any dial or the seed changing re-renders the preview live.
            for _sl in param_sliders:
                _sl.change(_preview, inputs=_preview_inputs,
                           outputs=[preview_img, preview_md])
            seeds_box.change(_preview, inputs=_preview_inputs,
                             outputs=[preview_img, preview_md])

        with gr.Tab("Sweep"):
            gr.Markdown(
                "Accuracy vs. the swept factor, per model (shaded = 95% bootstrap CI). "
                "Populated by the last run - sweep needs a params value set to a list."
            )
            sweep_plot = gr.Plot(label="Accuracy vs. factor")
            sweep_status = gr.Markdown()

        with gr.Tab("Calibration"):
            gr.Markdown(
                "**Does the model's stated confidence match its accuracy?** "
                "Expected Calibration Error (ECE) per condition, plus a reliability "
                "diagram (points on the dashed y=x line are well-calibrated; below it "
                "= over-confident). Enable *Elicit confidence* on the Run tab first - "
                "otherwise there are no confidences to score and ECE is null."
            )
            cal_df = gr.Dataframe(headers=D.calibration_rows(None)[0], wrap=True)
            reliability_plot = gr.Plot(label="Reliability diagram (full condition)")
            reliability_status = gr.Markdown()

        with gr.Tab("Inspect"):
            gr.Markdown("Browse individual wrong-or-errored cases from the last run.")
            with gr.Row():
                insp_cond = gr.Dropdown(
                    ["full", "oracle", "report", "blind"], value="full",
                    label="Condition"
                )
                case_dd = gr.Dropdown([], label="Failure case")
            with gr.Row():
                insp_img = gr.Image(label="Scene", type="pil")
                insp_detail = gr.Markdown()

        # ---- callbacks -----------------------------------------------------

        # Blank values for every Run-tab output, positionally matching the wired
        # outputs list below (used on early-return paths so the arity always lines
        # up): st_results, st_reports, st_images, status_md, results_df, acc_df,
        # vg_df, decomp_df, orc_df, orc_note, orc_banner, tax_df, cal_df.
        _empty_run = ([], [], {}, "", [], [], [], [], [], "", "", [], [])

        def _run(scene, probe, sel_models, sel_analyzers, s0, s1, s2, seeds_text,
                 elicit, blind, sweep_param, sweep_vals_text, session_keys):
            session_keys = session_keys or {}
            if not scene or not probe or not sel_models:
                msg = "Pick a scene, a probe, and at least one model."
                return (_empty_run[0], _empty_run[1], _empty_run[2], msg, *_empty_run[4:])
            # Refuse BEFORE the runner rather than after. A missing key used to come
            # back as a table of blank `model_error` rows under a success message,
            # which looks exactly like a model that answered badly.
            blocked = D.blocked_models(registry, sel_models, session_keys)
            if blocked:
                lines = "\n".join(
                    f"- **{b['name']}** needs `{b['api_key_env']}` "
                    f"({b['provider']}, {D.provider_info(b['api_key_env'])['url']})"
                    for b in blocked
                )
                msg = ("Nothing was run: no API key for "
                       f"{len(blocked)} of the selected models.\n\n{lines}\n\n"
                       "Enter the key under **API keys** above, or export the variable "
                       "and restart, or untick the model.")
                return (_empty_run[0], _empty_run[1], _empty_run[2], msg, *_empty_run[4:])
            gen = registry.get_scene(scene)
            specs = D.param_specs(gen)
            params = D.params_from_values(gen, [s0, s1, s2][:len(specs)])
            # optional sweep: turn one param into a list of values -> enables sweep_curve
            if sweep_param and sweep_param != "(none)" and sweep_vals_text.strip():
                vals = _parse_sweep_values(sweep_vals_text, specs, sweep_param)
                if vals:
                    params[sweep_param] = vals
            cfg = ExperimentConfig(
                scene_generator=scene,
                scene_params=params,
                seeds=_seed_list(seeds_text),
                probe=probe,
                models=sel_models,
                analyzers=sel_analyzers,
                elicit_confidence=bool(elicit),
                blind=bool(blind),
            )
            # Session keys are bound for the duration of this run only, and only in
            # this request's context - never in os.environ, where a concurrent visitor
            # would pick them up.
            with credentials.scope(session_keys):
                scene_objs, results, reports = run_experiment_collect(cfg, registry)
            names = {r.name for r in reports}
            skipped = [a for a in sel_analyzers if a not in names]
            msg = f"Ran {len(results)} probes over {len(scene_objs)} scenes."
            if skipped:
                msg += f" Skipped (gates unmet): {', '.join(skipped)}."
            errors = D.model_error_summary(results)
            if errors:
                msg += ("\n\nThe model call failed on some rows. These are excluded "
                        "from every accuracy rather than scored as wrong:\n\n"
                        + "\n".join(f"- {e}" for e in errors))
            orc = D.report_by_name(reports, "oracle")
            return (
                results,
                reports,
                _scene_to_image(scene_objs),
                msg,
                D.results_table(results),
                D.accuracy_rows(D.report_by_name(reports, "accuracy"))[1],
                D.visual_gain_rows(D.report_by_name(reports, "visual_gain"))[1],
                D.decomposition_rows(D.report_by_name(reports, "decomposition"))[1],
                D.oracle_rows(orc)[1],
                D.oracle_note(orc),
                D.oracle_suspect_banner(orc),
                D.taxonomy_rows(D.report_by_name(reports, "taxonomy"))[1],
                D.calibration_rows(D.report_by_name(reports, "calibration"))[1],
            )

        run_btn.click(
            _run,
            inputs=[
                scene_dd, probe_dd, model_cbg, analyzer_cbg,
                param_sliders[0], param_sliders[1], param_sliders[2],
                seeds_box, elicit_cb, blind_cb, sweep_dd, sweep_vals, st_keys,
            ],
            outputs=[
                st_results, st_reports, st_images, status_md, results_df,
                acc_df, vg_df, decomp_df, orc_df, orc_note, orc_banner, tax_df, cal_df,
            ],
        ).then(
            _render_sweep,
            inputs=st_reports,
            outputs=[sweep_plot, sweep_status],
        ).then(
            _render_reliability,
            inputs=st_reports,
            outputs=[reliability_plot, reliability_status],
        ).then(
            _refresh_cases,
            inputs=[st_results, insp_cond],
            outputs=case_dd,
        )

        insp_cond.change(_refresh_cases, inputs=[st_results, insp_cond], outputs=case_dd)
        case_dd.change(
            _show_case,
            inputs=[st_results, st_images, insp_cond, case_dd],
            outputs=[insp_img, insp_detail],
        )

        def _model_update(keys, show_all, selected):
            """Rebuild the model selector, keeping whatever is still selectable."""
            choices = D.model_choices(registry, keys, runnable_only=not show_all)
            valid = {v for _, v in choices}
            keep = [m for m in (selected or []) if m in valid]
            return gr.update(choices=choices,
                             value=keep or D.default_model_selection(registry, keys))

        def _save_key(env, key, keys, show_all, selected):
            """Store one provider's key for this session and clear the box.

            The box is cleared on save so a credential is not left sitting in the page
            for the rest of the session; what replaces it is a length and a source,
            which is enough to confirm the paste landed.
            """
            keys = dict(keys or {})
            if env:
                if key and key.strip():
                    keys[env] = key.strip()
                else:
                    keys.pop(env, None)          # empty box = fall back to the env var
            return (
                keys,
                gr.update(value=""),
                D.provider_hint(env, keys),
                D.credential_markdown(registry, keys),
                _model_update(keys, show_all, selected),
            )

        def _on_provider(env, keys):
            return D.provider_hint(env, keys or {})

        save_btn.click(
            _save_key,
            inputs=[provider_dd, key_box, st_keys, show_all_cb, model_cbg],
            outputs=[st_keys, key_box, provider_md, key_status_md, model_cbg],
        )
        key_box.submit(
            _save_key,
            inputs=[provider_dd, key_box, st_keys, show_all_cb, model_cbg],
            outputs=[st_keys, key_box, provider_md, key_status_md, model_cbg],
        )
        provider_dd.change(_on_provider, inputs=[provider_dd, st_keys],
                           outputs=provider_md)
        show_all_cb.change(_model_update, inputs=[st_keys, show_all_cb, model_cbg],
                           outputs=model_cbg)

        # Show the initial scene preview as soon as the app loads.
        app.load(_preview, inputs=_preview_inputs, outputs=[preview_img, preview_md])

    return app


# Tab-specific render helpers (kept module-level so they can be unit-smoke-tested)


def _render_sweep(reports):
    """Return (matplotlib figure | None, status markdown) for the sweep tab."""
    rep = D.report_by_name(reports, "sweep_curve")
    series = D.sweep_series(rep)
    if not series["series"]:
        return None, "No sweep available - set a params value to a list and re-run."
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, "matplotlib not installed - `pip install renderprobe[ui]`."

    fig, ax = plt.subplots(figsize=(6, 4))
    for model, s in series["series"].items():
        ax.plot(s["x"], s["y"], marker="o", label=model)
        ax.fill_between(s["x"], s["lo"], s["hi"], alpha=0.15)
    ax.set_xlabel(series["factor"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.set_title(f"Accuracy vs. {series['factor']}")
    fig.tight_layout()
    return fig, f"Factor: **{series['factor']}**"


def _render_reliability(reports):
    """Return (matplotlib figure | None, status markdown) for the calibration tab.

    Plots per-model reliability points (mean confidence vs. accuracy per bin) for
    the `full` condition, against the y=x diagonal. No confidences -> no figure.
    """
    rep = D.report_by_name(reports, "calibration")
    series = D.reliability_series(rep, condition="full", binning="equal_width")
    if not series["series"]:
        return None, (
            "No confidences to plot - tick **Elicit confidence** on the Run tab and "
            "re-run (the model must state a confidence for calibration to be scored)."
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, "matplotlib not installed - `pip install renderprobe[ui]`."

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    for model, s in series["series"].items():
        ax.plot(s["conf"], s["acc"], marker="o", label=model)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Reliability (full condition)")
    fig.tight_layout()
    return fig, "Points below the diagonal = over-confident; above = under-confident."


def _refresh_cases(results, condition):
    """Update the inspect-tab dropdown choices for a condition."""
    import gradio as gr

    cases = D.failure_cases(results, condition)
    choices = [D.case_label(c) for c in cases]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _show_case(results, images, condition, label):
    """Render the selected failure case: image + textual detail."""
    cases = D.failure_cases(results, condition)
    match = next((c for c in cases if D.case_label(c) == label), None)
    if match is None:
        return None, "Select a case."
    img = images.get(match["scene_id"])
    detail = (
        f"**scene_id:** {match['scene_id']}  \n"
        f"**model:** {match['model']}  \n"
        f"**condition:** {match['condition']}  \n"
        f"**failure category:** `{match['category']}`  \n"
        f"**ground truth:** `{match['gt']}`  \n"
        f"**parsed prediction:** `{match['pred']}`  \n"
        f"**error_flag:** `{match['error_flag']}`  \n"
    )
    if match.get("detail"):
        # A failed call has no output to show; what it has is a reason, and that is
        # the only thing on this panel worth reading when the run produced nothing.
        detail += f"**why the call failed:** {match['detail']}  \n"
    detail += f"\n**raw model output:**\n\n> {match['raw'] or '(empty)'}"
    return img, detail


def launch(share: bool = False, **kwargs) -> None:
    """Build and launch the app (entry point for `renderprobe ui`)."""
    build_app().launch(share=share, **kwargs)
