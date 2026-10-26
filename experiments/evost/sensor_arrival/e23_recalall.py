"""E23 (2026-10-25): g41's hand-over (e21_handover2.py, unchanged) with one more arm, the recalibration control of
Section 6.5 of the paper.

At the hand-over (stream window EVENT = 1,024) the two arms compared in the paper change two things at once,
  s_prof0_r   new rows refitted,  the NEW sensors' calibrators restarted;
  a_prof0_r   every row refitted, EVERY sensor's calibrator restarted;
so the established sensors' gain of a_prof0_r over s_prof0_r cannot be attributed to refitting their identities. The new arm
  r_prof0_r   the table of s_prof0_r (new rows refitted; every base forecast identical to s_prof0_r's) and EVERY
              sensor's calibrator restarted on the post-event forecasts of that table, exactly as a_prof0_r restarts them
              (same windows, exact ridge, the arm's own ridge and forgetting factor)
separates them: r_prof0_r vs s_prof0_r isolates rebuilding the established sensors' calibrators, and a_prof0_r vs r_prof0_r
compares joint identity refitting with new-only refitting when both restart every calibrator.

Arms (only those the comparison needs): same, konly, s_prof0_r, a_prof0_r, r_prof0_r. Every fit draws its window order from a
generator seeded per run (7919 * seed + year), so leaving out g41's other arms changes none of these arms; the smoke checks this
against g41's reference values on the same GPU type.
"""
import json
from pathlib import Path
import sys
import torch
import e3_confirm as e3
import e10_refresh as e10
import e16_budget as e16
import e17_coldstart as c17
import e18_handover as e18
import e21_handover2 as e21

H = e21.H
e21.RESTART['sr:'] = ('all', 'p:')                                       # r_: s_'s post-event table, every calibrator restarted


class RecalBundle(e21.Handover2Bundle):
    def base(self, key, raw, first, count, device):
        variant = key[3]
        if isinstance(variant, str) and variant.startswith('sr:'):          # r_: every forecast is s_'s
            return super().base(key[:3] + ('s:' + variant[3:],), raw, first, count, device)
        return super().base(key, raw, first, count, device)


def arms_of(B, seed):
    B.seed = seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arm = lambda name, ckpt, variant, **kw: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant, **kw)
    arms = [arm('same', B.year, 'full')]
    n_new = int((~B.known(B.old, device)).sum())
    B.info.update(sensors=B.n, new_sensors=n_new, cold_arms=bool(n_new >= c17.MIN_NEW and n_new < B.n), history=H, event=e18.EVENT,
                  released=e18.RELEASED, tail=e18.TAIL, select_windows=int(e3.SELECT_WINDOWS), half=e21.HALF, gap=e21.GAP,
                  runner='e23_recalall', arms_note='same, konly, s_prof0_r, a_prof0_r, r_prof0_r')
    if B.info['cold_arms']:
        arms += [arm('konly', B.old, 'konly'),
                 arm('s_prof0_r', B.old, 's:prof0', history=H, restart=True),
                 arm('a_prof0_r', B.old, 'sa:prof0', history=H, restart=True),
                 arm('r_prof0_r', B.old, 'sr:prof0', history=H, restart=True)]
    B.arm_names = [a['name'] for a in arms]
    return arms


e3.Bundle = RecalBundle
e3.arms_of = arms_of
e3.make_chunks = e21.make_chunks

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e16.set_budget(e16.FULL)
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    B = e10.RefreshBundle.last
    if B.info['cold_arms'] and '--max-windows' not in sys.argv:
        assert e18.EVENT >= e3.SELECT_WINDOWS, 'the selection windows must end before the event'
    (out / 'e10_info.json').write_text(json.dumps(B.info, indent=2) + '\n')
