"""Build a Word (.docx) version of the LaTeX report.

Run from report/ after `latexmk -pdf main.tex` (needs main.aux and main.bbl):
    python tools/make_docx.py
TikZ figures are rendered to PNG with pdflatex; figure, table, equation, section and citation numbers are taken
from the LaTeX build, so the Word file matches the PDF. Output: MedLift3D_Report.docx
"""
import glob
import os
import re
import shutil
import subprocess

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'docx_build')
FIGS = os.path.join(BUILD, 'figs')
OUT = os.path.join(ROOT, 'MedLift3D_Report.docx')


# ----------------------------------------------------------------- helpers
def group(s, i):
    """Return (content, end) of the balanced {...} group starting at s[i] == '{'."""
    assert s[i] == '{', s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '{' and s[j - 1] != '\\':
            depth += 1
        elif s[j] == '}' and s[j - 1] != '\\':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError('unbalanced group')


def skip_ws(s, i):
    while i < len(s) and s[i] in ' \t\n':
        i += 1
    return i


def replace_cmd(s, name, nargs, fn, opt=False):
    """Replace \\name[opt]{a1}..{an} by fn(opt, args)."""
    out, i, pat = [], 0, '\\' + name
    while True:
        j = s.find(pat, i)
        if j < 0:
            out.append(s[i:])
            return ''.join(out)
        k = j + len(pat)
        if k < len(s) and (s[k].isalpha()):
            out.append(s[i:k]); i = k; continue
        o = None
        if opt and s[k:k + 1] == '[':
            e = s.index(']', k); o = s[k + 1:e]; k = e + 1
        args = []
        for _ in range(nargs):
            k = skip_ws(s, k)
            a, k = group(s, k)
            args.append(a)
        out.append(s[i:j]); out.append(fn(o, args)); i = k


# ----------------------------------------------------------------- 1. render TikZ figures
def render_tikz():
    os.makedirs(FIGS, exist_ok=True)
    pre = open(os.path.join(ROOT, 'main.tex')).read().split('\\begin{document}')[0]
    pre = pre.replace('\\input{figures/tikzstyles}', '\\input{%s/figures/tikzstyles}' % ROOT)
    for f in sorted(glob.glob(os.path.join(ROOT, 'figures', 'fig_*.tex'))):
        name = os.path.splitext(os.path.basename(f))[0]
        png = os.path.join(FIGS, name + '.png')
        if os.path.exists(png) and os.path.getmtime(png) > os.path.getmtime(f):
            continue
        body = ('\\begin{ganttchart}' in open(f).read())
        tex = os.path.join(FIGS, name + '_t.tex')
        open(tex, 'w').write(pre + '\\pagestyle{empty}\n\\begin{document}\n\\noindent\\resizebox{%s}{!}{\\input{%s}}\n\\end{document}\n'
                             % ('\\linewidth', f))
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-output-directory', FIGS, tex],
                       stdout=subprocess.DEVNULL, check=True)
        subprocess.run(['pdftoppm', '-r', '300', '-png', '-singlefile', tex[:-4] + '.pdf', png[:-4]], check=True)
        im = Image.open(png).convert('RGB')
        bb = im.convert('L').point(lambda v: 255 if v < 250 else 0).getbbox()
        im = im.crop((max(bb[0] - 10, 0), max(bb[1] - 10, 0), bb[2] + 10, bb[3] + 10))
        if body:
            im = im.rotate(0, expand=True)
        im.save(png, dpi=(300, 300))
        for ext in ('aux', 'log', 'pdf', 'tex'):
            p = tex[:-4] + '.' + ext
            if os.path.exists(p):
                os.remove(p)
        print('rendered', name)


# ----------------------------------------------------------------- 2. numbers from the LaTeX build
def read_aux():
    aux = open(os.path.join(ROOT, 'main.aux')).read()
    labels = {k: v for k, v in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', aux)}
    cites = {k: int(v) for k, v in re.findall(r'\\bibcite\{([^}]+)\}\{\{(\d+)\}', aux)}
    return labels, cites


NAMES = {'fig': ('Figure', 'Figures'), 'tab': ('Table', 'Tables'), 'ch': ('Chapter', 'Chapters'),
         'sec': ('Section', 'Sections'), 'eq': ('Eq.', 'Eqs.')}


def cref(keys, labels):
    keys = [k.strip() for k in keys.split(',')]
    kind = keys[0].split(':')[0]
    nums = [labels.get(k, '??') for k in keys]
    if kind == 'eq':
        nums = ['(%s)' % n for n in nums]
    one, many = NAMES.get(kind, ('', ''))
    if len(nums) == 1:
        return '%s %s' % (one, nums[0])
    return '%s %s and %s' % (many, ', '.join(nums[:-1]), nums[-1])


def cite(keys, cites):
    ns = sorted({cites[k.strip()] for k in keys.split(',')})
    parts, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        parts.append(str(ns[i]) if j == i else ('%d, %d' % (ns[i], ns[j]) if j == i + 1 else '%d--%d' % (ns[i], ns[j])))
        i = j + 1
    return '[%s]' % ', '.join(parts)


# ----------------------------------------------------------------- 3. LaTeX -> pandoc-friendly LaTeX
def siunitx(s):
    def num(o, a):
        x = a[0]
        if re.fullmatch(r'\d{4,}', x):
            x = '{:,}'.format(int(x))
        return x
    s = replace_cmd(s, 'num', 1, num)
    s = replace_cmd(s, 'SI', 2, lambda o, a: '%s %s' % (a[0], a[1].replace('^3', '³').replace('^2', '²')))
    s = replace_cmd(s, 'ang', 1, lambda o, a: a[0] + '°')
    return s


def table_spec(spec):
    spec = re.sub(r'>\{[^{}]*(\{[^{}]*\})?[^{}]*\}', '', spec)
    spec = re.sub(r'\bX\b', 'p{6cm}', spec)
    return spec


STATE = {'ch': 0, 'sec': 0, 'sub': 0}


def convert(src, labels, cites):
    s = src
    s = re.sub(r'(?m)^\s*%.*$', '', s)                       # comments
    s = s.replace('\\FloatBarrier', '')
    s = re.sub(r'\\begin\{adjustbox\}\{[^}]*\}|\\end\{adjustbox\}', '', s)
    s = s.replace('sidewaysfigure', 'figure')
    s = replace_cmd(s, 'resultfig', 3, lambda o, a: '\\begin{figure}[htbp]\\centering\\includegraphics[width=%s]{figures/results/%s}\\caption{%s}\\label{%s}\\end{figure}'
                    % (o or '0.85\\linewidth', a[0], a[1], a[2]), opt=True)
    s = replace_cmd(s, 'resizebox', 3, lambda o, a: a[2].replace('\\input{figures/', '\\includegraphics[width=%s]{FIGS/' % (
        '0.95\\linewidth' if 'textheight' in a[0] else a[0])) + '')
    s = re.sub(r'\{FIGS/([a-z_0-9]+)\}', lambda m: '{%s/%s.png}' % (FIGS, m.group(1)), s)
    s = s.replace('figures/results/', os.path.join(ROOT, 'figures/results/'))
    # tables
    s = re.sub(r'\\begin\{tabularx\}\{[^}]*\}', '\\\\begin{tabular}', s)
    s = s.replace('\\end{tabularx}', '\\end{tabular}')

    def tab_env(m):
        return '\\begin{tabular}{%s}' % table_spec(m.group(1))
    s = re.sub(r'\\begin\{tabular\}\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', tab_env, s)

    def longtable(m):
        spec, body = m.group(1), m.group(2)
        cap = re.search(r'\\caption\{(.*?)\}\\label\{(.*?)\}\\\\', body)
        head, rest = body.split('\\endfirsthead')
        head = head[cap.end():] if cap else head
        rest = rest.split('\\endhead', 1)[1]
        out = '\\begin{tabular}{%s}%s%s\\end{tabular}' % (table_spec(spec), head, rest)
        if cap:
            out = '\\begin{table}\\caption{%s}\\label{%s}%s\\end{table}' % (cap.group(1), cap.group(2), out)
        return out
    s = re.sub(r'\\begin\{longtable\}\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}(.*?)\\end\{longtable\}', longtable, s, flags=re.S)
    s = re.sub(r'\\cmidrule(\([a-z]*\))?\{[^}]*\}', '', s)
    s = replace_cmd(s, 'makecell', 1, lambda o, a: a[0].replace('\\\\', ' '))
    s = s.replace('\\textbullet~', 'BULLETMARK').replace('\\newline', 'LINEBREAKMARK')
    s = s.replace('\\cmark', '✓').replace('\\xmark', '✗')
    s = siunitx(s)
    # floats: caption numbers, unwrap to paragraphs
    def float_env(m):
        kind, body = m.group(1), m.group(2)
        lab = re.search(r'\\label\{([^}]*)\}', body)
        num = labels.get(lab.group(1), '') if lab else ''
        body = re.sub(r'\\label\{[^}]*\}', '', body)
        caps = []
        def take(o, a):
            caps.append(a[0]); return ''
        body = replace_cmd(body, 'caption', 1, take, opt=True)
        main_cap = caps[-1] if caps else ''
        subs = caps[:-1]
        word = 'Figure' if kind == 'figure' else 'Table'
        capline = '\n\n\\textit{\\textbf{%s %s:} %s}\n\n' % (word, num, main_cap)
        if kind == 'figure':
            # subfigures: image followed by its own label (a), (b), ...
            parts = re.findall(r'\\includegraphics(\[[^\]]*\])?\{([^}]*)\}', body)
            out = ''
            for n, (o, f) in enumerate(parts):
                out += '\n\n\\includegraphics%s{%s}\n\n' % (o, f)
                if len(parts) > 1:
                    letter = '(%s)' % 'abcdefgh'[n]
                    out += '\\textit{%s %s}\n\n' % (letter, subs[n] if n < len(subs) and subs[n] else '')
            return out + capline
        body = body.replace('\\centering', '').replace('\\small', '')
        return capline + body + '\n\n'
    s = re.sub(r'\\begin\{(figure|table)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}', float_env, s, flags=re.S)
    s = re.sub(r'\\begin\{subfigure\}\{[^}]*\}|\\end\{subfigure\}|\\hfill', '', s)
    # cross references and citations
    s = replace_cmd(s, 'Cref', 1, lambda o, a: cref(a[0], labels))
    s = replace_cmd(s, 'cref', 1, lambda o, a: cref(a[0], labels))
    s = replace_cmd(s, 'cite', 1, lambda o, a: cite(a[0], cites))
    s = s.replace('~[', ' [').replace('][', ', ')
    # numbered headings
    state = STATE

    def heading(m):
        cmd, star, title = m.group(1), m.group(2), m.group(3)
        if star:
            return '\\%s*{%s}' % (cmd, title)
        if cmd == 'chapter':
            state['ch'] += 1; state['sec'] = state['sub'] = 0
            return '\\chapter*{Chapter %d: %s}' % (state['ch'], title)
        if cmd == 'section':
            state['sec'] += 1; state['sub'] = 0
            return '\\section*{%d.%d %s}' % (state['ch'], state['sec'], title)
        state['sub'] += 1
        return '\\subsection*{%d.%d.%d %s}' % (state['ch'], state['sec'], state['sub'], title)
    s = re.sub(r'\\(chapter|section|subsection)(\*?)\{((?:[^{}]|\{[^{}]*\})*)\}', heading, s)
    s = re.sub(r'\\label\{[^}]*\}', '', s)
    s = replace_cmd(s, 'paragraph', 1, lambda o, a: '\\textbf{%s} ' % a[0])
    # equations: add their numbers
    eqn = {'ch': 0, 'n': 0}
    out, pos = [], 0
    for m in re.finditer(r'\\chapter\*\{Chapter (\d+)|\\begin\{(equation|align)\}(.*?)\\end\{\2\}', s, flags=re.S):
        out.append(s[pos:m.start()]); pos = m.end()
        if m.group(1):
            eqn['ch'] = int(m.group(1)); eqn['n'] = 0; out.append(m.group(0)); continue
        body = m.group(3)
        if m.group(2) == 'equation':
            eqn['n'] += 1
            out.append('\\begin{equation}%s\\qquad(%d.%d)\\end{equation}' % (body, eqn['ch'], eqn['n']))
        else:
            lines = body.split('\\\\')
            new = []
            for ln in lines:
                eqn['n'] += 1
                new.append(ln.rstrip() + ' \\qquad (%d.%d)' % (eqn['ch'], eqn['n']))
            out.append('\\begin{align}%s\\end{align}' % ' \\\\ '.join(new))
    out.append(s[pos:])
    s = ''.join(out)
    return s


def bibliography(cites):
    bbl = open(os.path.join(ROOT, 'main.bbl')).read()
    items = re.split(r'\\bibitem\[', bbl)[1:]
    entries = []
    for it in items:
        it = it.split('\\end{thebibliography}')[0]
        # drop the optional label and key: [...]{key}
        depth, k = 1, 0
        while depth:
            depth += {'[': 1, ']': -1}.get(it[k], 0); k += 1
        key_end = it.index('}', k)
        text = it[key_end + 1:]
        text = text.replace('\\newblock', ' ').replace('\\penalty0', '').replace('\\natexlab', '')
        text = re.sub(r'\\urlprefix', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        entries.append(text)
    return '\\chapter*{References}\n\\begin{enumerate}\n' + '\n'.join('\\item ' + e for e in entries) + '\n\\end{enumerate}\n'


# ----------------------------------------------------------------- 4. assemble, pandoc, style
def main():
    if os.path.isdir(BUILD):
        for f in glob.glob(os.path.join(BUILD, '*.tex')):
            os.remove(f)
    render_tikz()
    labels, cites = read_aux()
    title = r"""
\begin{center}
\textbf{MedLift-3D: Beyond the Missing Wedge: Calibrated 3D Lung Nodule Mass and Growth Assessment from Low-Dose Chest Tomosynthesis}

By

Jubair Ahmed (011222136)

Foyez Ahammed Naeem (011222011)

Md Motasim Billah (011222016)

Submitted in partial fulfilment of the requirements of the degree of Bachelor of Science in Computer Science and Engineering

October 30, 2026

Department of Computer Science and Engineering

United International University
\end{center}
"""
    parts = [open(os.path.join(ROOT, 'frontmatter', f)).read() for f in ('abstract.tex', 'acknowledgements.tex')]
    parts += [open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, 'chapters', 'ch*.tex')))]
    body = '\n\n'.join(convert(p, labels, cites) for p in parts)
    body = body.replace('\\addcontentsline{toc}{chapter}{Abstract}', '').replace('\\addcontentsline{toc}{chapter}{Acknowledgements}', '')
    body = body.replace('\\chapter*{Acknowledgements}', '\\chapter*{Acknowledgements}', 1)
    body = body.replace('\\chapter*{Chapter 1:', 'TOCMARK\n\n\\chapter*{Chapter 1:', 1)
    doc = '\\documentclass{report}\n\\usepackage{amsmath}\n\\begin{document}\n' + title + body + bibliography(cites) + '\\end{document}\n'
    tex = os.path.join(BUILD, 'report_for_word.tex')
    open(tex, 'w').write(doc)
    ref = os.path.join(BUILD, 'reference.docx')
    make_reference(ref)
    subprocess.run(['pandoc', tex, '-f', 'latex', '-o', OUT, '--reference-doc', ref, '--resource-path', ROOT], check=True)
    style(OUT)
    print('wrote', OUT)


def make_reference(path):
    import docx
    from docx.shared import Pt, RGBColor
    data = subprocess.run(['pandoc', '-o', path, '--print-default-data-file', 'reference.docx'], check=True)
    d = docx.Document(path)
    for name in ('Normal', 'Body Text', 'First Paragraph', 'Compact'):
        if name in [s.name for s in d.styles]:
            st = d.styles[name]
            st.font.name = 'Times New Roman'; st.font.size = Pt(12)
            st.paragraph_format.line_spacing = 1.5
            st.paragraph_format.space_after = Pt(6)
    for lvl, size in ((1, 20), (2, 16), (3, 13), (4, 12)):
        st = next(x for x in d.styles if x.name == 'Heading %d' % lvl)
        st.font.name = 'Times New Roman'; st.font.size = Pt(size); st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(18 if lvl == 1 else 12)
        if lvl == 1:
            st.paragraph_format.page_break_before = True
    from docx.oxml.ns import qn
    for st in d.styles:
        rpr = st.element.rPr
        if rpr is not None and rpr.rFonts is not None:
            for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
                rpr.rFonts.attrib.pop(qn(a), None)
            rpr.rFonts.set(qn('w:ascii'), 'Times New Roman'); rpr.rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    d.save(path)


def style(path):
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt
    d = docx.Document(path)
    sec = d.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)   # A4
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(1)
    width = sec.page_width - sec.left_margin - sec.right_margin
    paras = d.paragraphs
    for p in paras:
        txt = p.text.strip()
        if p._p.xpath('.//pic:pic'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif re.match(r'^(Figure|Table) \d+\.\d+:', txt) or re.match(r'^\([a-h]\)', txt):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.size = Pt(10)
        if txt == 'TOCMARK':
            p.text = ''
            h = p.insert_paragraph_before('Table of Contents')
            h.style = next(x for x in d.styles if x.name == 'Heading 1')
            run = p.add_run()
            for tag, attr in (('w:fldChar', {'w:fldCharType': 'begin'}), ('w:instrText', None),
                              ('w:fldChar', {'w:fldCharType': 'separate'}), ('w:t', None), ('w:fldChar', {'w:fldCharType': 'end'})):
                el = OxmlElement(tag)
                if attr:
                    for k, v in attr.items():
                        el.set(qn(k), v)
                if tag == 'w:instrText':
                    el.set(qn('xml:space'), 'preserve'); el.text = 'TOC \\o "1-3" \\h \\z \\u'
                if tag == 'w:t':
                    el.text = 'Right-click and choose Update Field to build the table of contents.'
                run._r.append(el)
    for shp in d.inline_shapes:
        if shp.width > width:
            r = width / shp.width
            shp.width = int(shp.width * r); shp.height = int(shp.height * r)
    # title page: centred, larger title, page break after
    for p in paras[:14]:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if paras and paras[0].runs:
        for r in paras[0].runs:
            r.font.size = Pt(18)
    # tables: rebuild bullet lists in cells, widths from content, booktabs-like borders
    for t in d.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in list(cell.paragraphs):
                    if 'LINEBREAKMARK' in p.text or 'BULLETMARK' in p.text:
                        items = [x.strip().replace('BULLETMARK', '\u2022 ') for x in p.text.split('LINEBREAKMARK')]
                        for r in p.runs[1:]:
                            r._r.getparent().remove(r._r)
                        p.runs[0].text = items[0]
                        anchor = p
                        for it in items[1:]:
                            np_ = OxmlElement('w:p'); anchor._p.addnext(np_)
                            from docx.text.paragraph import Paragraph
                            anchor = Paragraph(np_, p._parent); anchor.add_run(it)
        ncol = len(t.columns)
        lens = [0] * ncol
        for row in t.rows:
            for k, cell in enumerate(row.cells[:ncol]):
                txt = cell.text
                lens[k] = max(lens[k], min(len(txt), 60), max((len(w) for w in txt.split()), default=0) * 1.6)
        wts = [max(x, 6) for x in lens]
        tot = sum(wts)
        for row in t.rows:
            for k, cell in enumerate(row.cells[:ncol]):
                cell.width = int(width * wts[k] / tot)
        grid = t._tbl.tblGrid
        if grid is not None:
            for k, gc in enumerate(grid.findall(qn('w:gridCol'))):
                if k < ncol:
                    gc.set(qn('w:w'), str(int(width / 635 * wts[k] / tot)))
        tblPr = t._tbl.tblPr
        borders = OxmlElement('w:tblBorders')
        for edge in ('top', 'bottom', 'insideH'):
            e = OxmlElement('w:' + edge)
            e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '6' if edge == 'insideH' else '12')
            e.set(qn('w:color'), '808080' if edge == 'insideH' else '000000')
            borders.append(e)
        tblPr.append(borders)
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(2)
                    for r in p.runs:
                        r.font.size = Pt(9.5)
        for cell in t.rows[0].cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.bold = True
    # ask Word to refresh the table of contents on opening
    settings = d.settings.element
    upd = OxmlElement('w:updateFields'); upd.set(qn('w:val'), 'true'); settings.append(upd)
    d.save(path)


if __name__ == '__main__':
    main()
