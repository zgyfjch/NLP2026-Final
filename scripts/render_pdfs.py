from pathlib import Path
import json
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'tmp/pdf_qa';out.mkdir(parents=True,exist_ok=True)
paths=list((ROOT/'NLP2026_TeamXX_Demo').rglob('*.pdf'))+list((ROOT/'reports').glob('*.pdf'))
inventory=[]
for path in paths:
    doc=pdfium.PdfDocument(path); dest=out/path.stem;dest.mkdir(exist_ok=True)
    thumbs=[]
    for i,page in enumerate(doc):
        img=page.render(scale=1.4).to_pil();img.save(dest/f'page-{i+1}.png')
        thumb=img.copy();thumb.thumbnail((420,580));thumbs.append(thumb)
    sheet=Image.new('RGB',(840,620*((len(thumbs)+1)//2)),'#d9dfe5');draw=ImageDraw.Draw(sheet)
    for i,im in enumerate(thumbs):
        x=(i%2)*420;y=(i//2)*620;sheet.paste(im,(x,y+25));draw.text((x+10,y+5),f'Page {i+1}',fill='black')
    sheet.save(out/f'{path.stem}_contact.png')
    item=dict(file=str(path),pages=len(doc));inventory.append(item);print(item)
(out/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
