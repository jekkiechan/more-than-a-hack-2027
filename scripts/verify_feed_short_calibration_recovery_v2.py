"""Run the unchanged independent grid verifier against the recovered readout."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import feed_short_calibration_readout_v2 as recovery
from scripts import verify_feed_short_calibration_v1 as original
p=recovery.p
read,sha,write=p.read,p.sha,p.write

def mapped(path):return recovery.RESULT if Path(path)==recovery.ORIGINAL else path

def receipt(path,value):
    value.update(aggregate_path=str(recovery.RESULT.relative_to(ROOT)),preserved_partial_aggregate_sha256=sha(recovery.ORIGINAL),recovery_amendment_sha256=sha(recovery.AMENDMENT),verifier_adapter_sha256=sha(Path(__file__)))
    write(path,value)

try:
    p.read=lambda path:read(mapped(path))
    p.sha=lambda path:sha(mapped(path))
    p.write=receipt
    original.main()
finally:p.read,p.sha,p.write=read,sha,write
