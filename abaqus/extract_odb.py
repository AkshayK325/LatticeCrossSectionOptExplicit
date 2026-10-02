"""Run with Abaqus Python (Abaqus 2024 = Python 3):

    abaqus python extract_odb.py A1_cantilever.odb [more.odb ...]

Writes next to each odb:
  <job>_history.csv   all history output, long format: step, region, variable, time, value
                      (time = total time across steps)
  <job>_field_U.csv   nodal displacement U of every field frame: step, frame_time, instance, node, U1, U2, U3
Copy the CSVs (the odb is optional) into abaqus/csv/ of the repository.
"""
import csv
import os
import sys

from odbAccess import openOdb


def extract(path):
    odb = openOdb(path, readOnly=True)
    stem = os.path.splitext(path)[0]
    n_hist = 0
    with open(stem + '_history.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'region', 'variable', 'time', 'value'])
        for sname in odb.steps.keys():
            step = odb.steps[sname]
            t0 = step.totalTime
            for rname in step.historyRegions.keys():
                reg = step.historyRegions[rname]
                for vname in reg.historyOutputs.keys():
                    for t, val in reg.historyOutputs[vname].data:
                        w.writerow([sname, rname, vname, repr(t0 + t), repr(val)])
                        n_hist += 1
    n_field = 0
    with open(stem + '_field_U.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'frame_time', 'instance', 'node', 'U1', 'U2', 'U3'])
        for sname in odb.steps.keys():
            step = odb.steps[sname]
            for fr in step.frames:
                if 'U' not in fr.fieldOutputs.keys():
                    continue
                for v in fr.fieldOutputs['U'].values:
                    inst = v.instance.name if v.instance is not None else ''
                    w.writerow([sname, repr(step.totalTime + fr.frameValue), inst, v.nodeLabel] +
                               [repr(float(x)) for x in v.data])
                    n_field += 1
    odb.close()
    print('%s: %d history rows, %d field rows' % (path, n_hist, n_field))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        extract(p)
