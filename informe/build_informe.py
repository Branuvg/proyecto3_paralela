"""Genera informe/Proyecto3_Informe.docx a partir de docs/*.md y ../template.docx (caratula)."""
import re, struct, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT.parent / "template.docx"
OUT = ROOT / "informe" / "Proyecto3_Informe.docx"
DOCS = ROOT / "docs"
FILES = ["01_analisis_solucion_secuencial", "02_medicion_tiempo_secuencial", "03_oportunidades_paralelizacion",
         "04_diseno_algoritmo_paralelo", "05_implementacion_cuda", "06_analisis_y_conclusiones"]
REPO = "https://github.com/Branuvg/proyecto3_paralela"

FONT = '<w:rFonts w:ascii="Calibri" w:eastAsia="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
MONO = '<w:rFonts w:ascii="Consolas" w:eastAsia="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>'


def rpr(b=False, i=False, code=False, sz=22):
    f = MONO if code else FONT
    s = sz - 2 if code else sz
    return ("<w:rPr>" + f + ("<w:b/><w:bCs/>" if b else "") + ("<w:i/><w:iCs/>" if i else "") +
            '<w:color w:val="000000"/><w:sz w:val="%d"/><w:szCs w:val="%d"/></w:rPr>' % (s, s))


def run(t, **k):
    return '<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>' % (rpr(**k), escape(t))


TOK = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)")


def inline(text, b=False, sz=22):
    out = []
    for part in TOK.split(text):
        if not part:
            continue
        if part.startswith("**"):
            out.append(run(part[2:-2], b=True, sz=sz))
        elif part.startswith("`"):
            out.append(run(part[1:-1], code=True, b=b, sz=sz))
        elif part.startswith("*") and len(part) > 2:
            out.append(run(part[1:-1], i=True, b=b, sz=sz))
        else:
            out.append(run(part, b=b, sz=sz))
    return "".join(out)


def para(runs, jc=None, before=0, after=120, keep=False, style=None, num=None, ind=None):
    p = ""
    if style:
        p += '<w:pStyle w:val="%s"/>' % style
    if keep:
        p += "<w:keepNext/>"
    if num is not None:
        p += '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="%d"/></w:numPr>' % num
    p += '<w:spacing w:before="%d" w:after="%d"/>' % (before, after)
    if ind:
        p += ind
    if jc:
        p += '<w:jc w:val="%s"/>' % jc
    return "<w:p><w:pPr>%s</w:pPr>%s</w:p>" % (p, runs)


def strip_num(t):
    return re.sub(r"^\d+(\.\d+)*\.?\s+", "", t)


BORD = "".join('<w:%s w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>' % s
               for s in ("top", "left", "bottom", "right", "insideH", "insideV"))


def table(rows):
    ncol = len(rows[0])
    w = [max(6, min(40, max(len(re.sub(r"[`*]", "", r[c])) for r in rows if c < len(r)))) for c in range(ncol)]
    tot = sum(w)
    cw = [max(700, int(9360 * x / tot)) for x in w]
    cw[-1] += 9360 - sum(cw)
    x = ('<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/><w:tblBorders>%s</w:tblBorders><w:tblLayout w:type="fixed"/>'
         '<w:tblCellMar><w:left w:w="70" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>' % BORD)
    x += "".join('<w:gridCol w:w="%d"/>' % c for c in cw) + "</w:tblGrid>"
    for ri, r in enumerate(rows):
        x += "<w:tr><w:trPr><w:cantSplit/>" + ("<w:tblHeader/>" if ri == 0 else "") + "</w:trPr>"
        for ci in range(ncol):
            t = r[ci] if ci < len(r) else ""
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if ri == 0 else ""
            x += ('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s</w:tcPr>' % (cw[ci], shade) +
                  para(inline(t, b=(ri == 0), sz=18), after=20, before=20) + "</w:tc>")
        x += "</w:tr>"
    return x + "</w:tbl>" + para("", after=60)


images = []


def image(alt, path):
    data = (DOCS / path).read_bytes()
    wpx, hpx = struct.unpack(">II", data[16:24])
    cx = int(5486400 * (1 if wpx >= 900 else wpx / 900))
    cy = int(cx * hpx / wpx)
    n = len(images) + 1
    rid = "rIdImg%d" % n
    name = "image%d.png" % n
    images.append((rid, name, data))
    a = escape(alt, {'"': "&quot;"})
    d = ('<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="%d" cy="%d"/>'
         '<wp:docPr id="%d" name="Figura %d" descr="%s"/>'
         '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
         '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="%d" name="%s"/><pic:cNvPicPr/></pic:nvPicPr>'
         '<pic:blipFill><a:blip r:embed="%s"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
         '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
         '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>') % (cx, cy, 100 + n, n, a, n, name, rid, cx, cy)
    return para(d, jc="center", after=120)


def convert(md):
    lines = md.splitlines()
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                out.append(para(run(lines[i] or " ", code=True), after=0, ind='<w:ind w:left="360"/>'))
                i += 1
            i += 1
            out.append(para("", after=60))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1))
            t = strip_num(m.group(2))
            if lvl == 1:
                out.append(para(run(t, b=True, sz=24), style="Heading1", before=360, after=120, keep=True))
            else:
                out.append(para(inline(t, b=True), before=200, after=80, keep=True))
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(table(rows))
            continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)", ln)
        if m:
            out.append(image(m.group(1), m.group(2)))
            i += 1
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", ln)
        if m:
            num = 2 if m.group(2)[0].isdigit() else 1
            out.append(para(inline(m.group(3)), num=num, after=60))
            i += 1
            continue
        if ln.startswith(">"):
            out.append(para(inline(ln.lstrip("> ").strip()), ind='<w:ind w:left="360"/>'))
            i += 1
            continue
        buf = [ln.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|!\[|>|\s*([-*]|\d+\.)\s)", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        out.append(para(inline(" ".join(buf))))
    return "".join(out)


NUMBERING = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="hybridMultilevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/></w:rPr></w:lvl></w:abstractNum>'
    '<w:abstractNum w:abstractNumId="2"><w:multiLevelType w:val="hybridMultilevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>'
    '<w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num><w:num w:numId="2"><w:abstractNumId w:val="2"/></w:num></w:numbering>')


def cover_fix(x):
    x = x.replace("<w:t>Titulo</w:t>", "<w:t>Mejorando algoritmos con CUDA</w:t>")
    ps = list(re.finditer(r"<w:p[ >].*?</w:p>", x, flags=re.S))
    p22, p27 = ps[22], ps[27]
    assert "Gabriel Bran" in p22.group(0) and "Repo:" in p27.group(0)

    def cp(txt):
        return ('<w:r><w:rPr>%s<w:color w:val="000000"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
                '<w:t xml:space="preserve">%s</w:t></w:r>' % (FONT, escape(txt)))
    names = "<w:r><w:br/></w:r>".join(cp(t) for t in ["Iris Ayala, 23965", "Gabriel Bran, 23590", "Anggie Quezada, 23643"])
    new22 = '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>%s</w:p>' % names
    new27 = '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>%s</w:p>' % cp("Repo: " + REPO)
    return x[:p22.start()] + new22 + x[p22.end():p27.start()] + new27 + x[p27.end():]


def main():
    z = zipfile.ZipFile(TPL)
    doc = cover_fix(z.read("word/document.xml").decode("utf8"))
    body = para(run("Proyecto 3: Mejorando algoritmos con CUDA", b=True, sz=28), jc="center", after=200, keep=True)
    for f in FILES:
        body += convert((DOCS / (f + ".md")).read_text(encoding="utf8"))
    pb = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
    k = doc.rindex("<w:sectPr")
    doc = doc[:k] + pb + body + doc[k:]
    rels = z.read("word/_rels/document.xml.rels").decode("utf8")
    add = '<Relationship Id="rIdNum1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
    add += "".join('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/%s"/>' % (r, n)
                   for r, n, _ in images)
    rels = rels.replace("</Relationships>", add + "</Relationships>")
    ct = z.read("[Content_Types].xml").decode("utf8")
    extra = '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
    if 'Extension="png"' not in ct:
        extra = '<Default Extension="png" ContentType="image/png"/>' + extra
    ct = ct.replace("</Types>", extra + "</Types>")
    styles = z.read("word/styles.xml").decode("utf8")
    if 'w:styleId="Heading1"' not in styles:
        styles = styles.replace("</w:styles>", '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:outlineLvl w:val="0"/></w:pPr></w:style></w:styles>')
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as o:
        for n in z.namelist():
            data = z.read(n)
            if n == "word/document.xml":
                data = doc.encode("utf8")
            elif n == "word/_rels/document.xml.rels":
                data = rels.encode("utf8")
            elif n == "[Content_Types].xml":
                data = ct.encode("utf8")
            elif n == "word/styles.xml":
                data = styles.encode("utf8")
            o.writestr(n, data)
        o.writestr("word/numbering.xml", NUMBERING)
        for _, n, d in images:
            o.writestr("word/media/" + n, d)
    print("OK", OUT, len(images), "imagenes")


main()
