"""Materialize the two Page HTML embeds from the saved aggregate history."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def build():
    data=json.loads((ROOT/'docs/research/experiment_progress.json').read_text())
    template=(ROOT/'scripts/progress_visualization.html').read_text()
    output=ROOT/'data/experiments/progress_page'
    output.mkdir(parents=True,exist_ok=True)
    for mode,key in (('development','development'),('public','organizer')):
        encoded=json.dumps(data[key],separators=(',',':'),ensure_ascii=False).replace('</','<\\/')
        html=template.replace('__MODE__',json.dumps(mode)).replace('__DATA__',encoded)
        assert len(html.encode())<256*1024
        (output/(mode+'.html')).write_text(html)
        print(mode,len(html.encode()),len(encoded.encode()))
if __name__=='__main__':build()
