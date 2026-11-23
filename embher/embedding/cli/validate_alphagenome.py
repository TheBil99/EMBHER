#!/usr/bin/env python3
"""
Check that the instrumented capture forward reproduces the library's own
uninstrumented forward pass.

hk.intercept_methods wraps every module call, so the risk is that capturing
perturbs the computation (or that the mixed-precision policy differs). This
runs alphagenome_research's own create_model() apply_fn on the same sequence
and params, and compares its coverage-head predictions against the captured
ones element-wise.

Usage:
    python scripts/validate_alphagenome.py --weights_path $ALPHAGENOME_WEIGHTS
"""

import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights_path', default=os.environ.get('ALPHAGENOME_WEIGHTS'))
    p.add_argument('--window', type=int, default=102400)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    import jax
    import jax.numpy as jnp
    from alphagenome_research.model import dna_model as local_dna_model

    from embher.embedding.capture import alphagenome as agcap

    print("loading checkpoint ...")
    model = agcap.load_model(args.weights_path)

    rng = np.random.default_rng(args.seed)
    sequence = ''.join(rng.choice(list('ACGT'), size=args.window))

    # --- captured (instrumented) path -------------------------------------
    captured = agcap.run_forward(
        model, sequence,
        window_bp=args.window, win_start=0,
        crop_start=0, crop_end=args.window,
    )

    # --- reference (library) path -----------------------------------------
    _, apply_fn, _, _, _ = local_dna_model.create_model(
        model.metadata,
        num_splice_sites=local_dna_model.ModelSettings().num_splice_sites,
        splice_site_threshold=local_dna_model.ModelSettings().splice_site_threshold,
    )
    dna = jax.device_put(
        np.asarray(model.one_hot_encoder.encode(sequence))[np.newaxis], model.device)
    organism_index = jax.device_put(
        np.full((1,), local_dna_model.convert_to_organism_index(model.organism),
                dtype=np.int32),
        model.device)
    # _apply_fn(params, state, dna_sequence, organism_index) -> predictions
    predictions = apply_fn(model.params, model.state, dna, organism_index)

    # --- control: same policy, same jit, but NO interception ---------------
    # Any difference this path shows against the library is pure XLA/bf16
    # noise, not something hk.intercept_methods caused. Without it we cannot
    # tell a capture bug from bf16 rounding.
    import haiku as hk
    import jmp
    from alphagenome_research.model import model as ag_model

    settings = local_dna_model.ModelSettings()
    policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')

    @hk.transform_with_state
    def _plain(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(ag_model.AlphaGenome, policy):
            preds, _ = ag_model.AlphaGenome(
                model.metadata,
                num_splice_sites=settings.num_splice_sites,
                splice_site_threshold=settings.splice_site_threshold,
            )(dna_sequence, organism_index)
        return {f'{h}/{r}': preds[h][f'scaled_predictions_{r}bp']
                for h in agcap.COVERAGE_HEADS for r in (1, 128)
                if f'scaled_predictions_{r}bp' in preds[h]}

    control, _ = jax.jit(_plain.apply)(
        model.params, model.state, None, dna, organism_index)

    print(f"\n{'tensor':38s} {'max|Δ|':>12s} {'rel':>10s} {'ctrl rel':>10s}  status")
    worst = 0.0
    worst_control = 0.0
    for head in agcap.COVERAGE_HEADS:
        for res in (1, 128):
            key = f'scaled_predictions_{res}bp'
            if key not in predictions[head]:
                continue
            name = f'predictions/{head}/res_{res}'
            ref = np.asarray(jnp.asarray(predictions[head][key][0], jnp.float32))
            got = captured[name][0].astype(np.float32)
            if got.shape != ref.shape:
                print(f"{name:38s} SHAPE MISMATCH {got.shape} vs {ref.shape}")
                worst = np.inf
                continue
            delta = float(np.abs(got - ref).max())
            scale = float(np.abs(ref).max()) or 1.0
            ctrl = np.asarray(jnp.asarray(control[f'{head}/{res}'][0], jnp.float32))
            ctrl_rel = float(np.abs(ctrl - ref).max()) / scale
            worst = max(worst, delta / scale)
            worst_control = max(worst_control, ctrl_rel)
            # The capture is only suspect if it disagrees with the library by
            # noticeably more than the uninstrumented control already does.
            ok = delta / scale <= max(3 * ctrl_rel, 1e-6)
            print(f"{name:38s} {delta:12.6g} {delta/scale:10.3g} {ctrl_rel:10.3g}  "
                  f"{'OK' if ok else 'MISMATCH'}")

    print(f"\nworst capture-vs-library relative difference: {worst:.3g}")
    print(f"worst control-vs-library relative difference: {worst_control:.3g}")
    print("  (control = identical model/policy, no interception; its spread is"
          " the bf16/XLA noise floor)")
    print("PASS" if worst <= max(3 * worst_control, 1e-6) else "FAIL")


if __name__ == '__main__':
    main()
