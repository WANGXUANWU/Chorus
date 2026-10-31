"""Tables of the 30 Oct pre-submission revision, derived from the approved a25 tables.

Author comment: the tables carried a red shade and an underline, which looked poor, so only the first place stays,
in bold black. Row labels lose their bold, so bold has one meaning everywhere. Table 1 is ranked again, because the
old table marked every completed Chorus run: bold now goes to the best completed run of each column, which gives
adelie the two smallest designs for time. The dagger keeps its meaning. Captions state the unit of the +- values
and of the sizes, and Table 4 names its main-grid column "Main" as the text does.
Reads manuscript/tables/*_a25*.tex; writes the same tables under the suffix a26.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'manuscript', 'tables')
SHADE, BOLD_SYS = '\\cellcolor{best}', '\\textbf{\\sys{}}'


def read(name):
    return open(os.path.join(TAB, name + '.tex'), encoding='utf-8').read()


def write(name, text):
    open(os.path.join(TAB, name + '.tex'), 'w', encoding='utf-8').write(text)
    print(name)


def black(text):
    """Drop the shade and the underline, and the bold of the row labels."""
    text = text.replace(SHADE, '').replace(BOLD_SYS, '\\sys{}')
    text = text.replace('\\textbf{Frozen configuration}', 'Frozen configuration')
    return re.sub(r'\\underline\{([^{}]*)\}', r'\1', text)


def rank_ooc(text):
    """Bold the smallest completed value of each column (a dagger marks a stopped run)."""
    head, body = text.split('\\midrule\n')
    body, tail = body.split('\\bottomrule')
    rows = [r.rstrip('\\\n').split(' & ') for r in body.strip().split('\n')]
    rows = [[re.sub(r'\\textbf\{([^{}]*)\}', r'\1', c.strip()) for c in r] for r in rows]
    for j in range(1, len(rows[0])):
        done = [float(r[j]) for r in rows if 'dagger' not in r[j]]
        for r in rows:
            if 'dagger' not in r[j] and float(r[j]) == min(done):
                r[j] = '\\textbf{' + r[j] + '}'
    return head + '\\midrule\n' + ''.join(' & '.join(r) + '\\\\\n' for r in rows) + '\\bottomrule' + tail


def sub(text, old, new):
    assert old in text, old
    return text.replace(old, new)


def main():
    t = rank_ooc(black(read('T1_ooc_a25')))
    write('T1_ooc_a26', sub(t, 'to 400{,}000 rows. $\\dagger$', 'to 400{,}000 rows. Bold: best completed run. $\\dagger$'))

    t = black(read('T3_accuracy_a25_compact'))
    write('T3_accuracy_a26_compact', sub(t, 'Bold on a shaded cell: best. Underline: second best. $\\dagger$: not '
                                            'significantly different from the best.',
                                         'Bold: best. $\\dagger$: not significantly different from the best. Size: '
                                         'size error.'))

    t = black(read('T4_real_a25'))
    write('T4_real_a26', sub(t, 'selects. Bold on a shaded cell: best. Underline: second best. $\\dagger$',
                             'selects, mean $\\pm$ standard error over responses, with the median model size. Bold: '
                             'best. $\\dagger$'))

    t = read('T5_attribution_a25')
    t = sub(t, 'and rise in FEE, positive', 'and mean rise in FEE per case, positive')
    write('T5_attribution_a26', sub(t, ' & All & ', ' & Main & '))

    t = black(read('T6_sensitivity_a25'))
    write('T6_sensitivity_a26', sub(t, 'one choice varied at a time. Bold on a shaded cell: best setting. Underline: '
                                       'second best.',
                                    'one choice varied at a time, mean $\\pm$ standard error over cases. Bold: best '
                                    'setting.'))


if __name__ == '__main__':
    main()
