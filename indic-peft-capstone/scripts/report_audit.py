from pathlib import Path
import zipfile, re
ROOT=Path('/home/ubuntu/indic-peft-comparison/reports')
for p in sorted(ROOT.glob('*.docx')):
    with zipfile.ZipFile(p) as z:
        xml=z.read('word/document.xml').decode('utf-8','ignore')
        app=z.read('docProps/app.xml').decode('utf-8','ignore') if 'docProps/app.xml' in z.namelist() else ''
    paras=[]
    for m in re.finditer(r'<w:p(?: [^>]*)?>(.*?)</w:p>',xml,re.S):
        body=m.group(1)
        texts=re.findall(r'<w:t[^>]*>(.*?)</w:t>',body,re.S)
        text=''.join(texts).replace('&amp;','&').replace('&lt;','<').replace('&gt;','>')
        style=re.search(r'<w:pStyle[^>]*w:val="([^"]+)"',body)
        if text.strip(): paras.append((style.group(1) if style else '', text.strip()))
    tables=len(re.findall(r'<w:tbl(?: |>)',xml))
    print('\nFILE',p.name,'paragraphs',len(paras),'tables',tables,'pages_text',re.findall(r'<Pages>(.*?)</Pages>',app))
    for style,text in paras[:250]:
        if style.lower().startswith(('heading','title','subtitle')) or re.match(r'^(chapter|[0-9]+\.?[0-9]*\s)',text,re.I):
            print(style, text[:200])
