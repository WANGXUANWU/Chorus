"""Record the frozen Chorus configuration with a hash of its source (configs/chorus_frozen.json)."""
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, 'configs', 'chorus_frozen.json')
c = json.load(open(p, encoding='utf-8'))
files = sorted(os.path.relpath(f, ROOT).replace(os.sep, '/') for f in glob.glob(os.path.join(ROOT, 'src', 'chorus', '*.py')))
h = hashlib.sha256()
for f in files:
    h.update(open(os.path.join(ROOT, f), 'rb').read())
c.update({
    'frozen': '2026-10-22 (amendments 14 to 18 and 21)',
    'strategy_rule': ('stage 1 from full sufficient statistics (one fused row-panel pass forming X^T X / n and X^T Y / n, '
                      'then the certified Gram path) when the Gram share of the ledger holds 8 D^2 bytes and its '
                      'transients and 1.25 times the predicted statistics time is below the predicted working-set '
                      'time; otherwise the working-set engine (amendment 21)'),
    'evidence_strategy': ('results/v2/strategy_invariance.csv: 36 tuning cases, Beijing and 16 DepMap dependencies, '
                          '664 responses with identical stage-1 levels, picks and supports under both strategies'),
    'startup_check': ('ledger.preflight refuses a budget before any data are loaded when the resident set, the '
                      'start-up allowance of 180 MB, the planned inputs and 1.1 times the fixed arrays exceed M minus '
                      'the margin (amendment 18); fitting code and selection unchanged'),
    'refit': ('pooled relaxed refit of every candidate on its own support: phi in {0, .25, .5, .75, 1}, one phi per '
              'candidate chosen by the validation error summed over responses in units of each validation variance; '
              'least squares from the stage-2 Gram matrix (amendment 17)'),
    'evidence_refit': ('results/pilot/acc/*/Chorus3 (36 tuning cases): FEE 0.0633 to 0.0602, lower in 36 of 36 cases; '
                       'supports and picks identical to the frozen rule'),
    'source_files': files,
    'source_sha256': h.hexdigest(),
    'source_hash_method': 'sha256 of the concatenated bytes of source_files in the listed order',
})
json.dump(c, open(p, 'w', encoding='utf-8'), indent=1)
print(c['source_sha256'], len(files))
