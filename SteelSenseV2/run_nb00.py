import nbformat, time, sys
from nbclient import NotebookClient
SRC = "00_Dataset_Integrity_Audit.ipynb"
t = time.perf_counter()
nb = nbformat.read(SRC, as_version=4)
NotebookClient(nb, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": "."}}).execute()
print(f"EXECUTED OK in {time.perf_counter()-t:.0f}s")
nbformat.write(nb, "00_Dataset_Integrity_Audit.executed.ipynb")
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            print(f"CELL {i} ERROR:", o["ename"], o["evalue"])
        elif o.get("output_type") == "stream":
            txt = o["text"].strip()
            if any(k in txt for k in ("TOTAL:", "Contaminated:", "Whole dataset",
                                      "Classes retaining", "must be 0")):
                print(txt[:500]); print("-" * 60)
