import pylidc as pl
import numpy as np
import matplotlib.pyplot as plt

plt.interactive(False)

'''
Feature              Meaning                    # 
-                    -                          - 
Subtlety           | Fairly Subtle            | 3 
Internalstructure  | Soft Tissue              | 1 
Calcification      | Absent                   | 6 
Sphericity         | Ovoid/Round              | 4 
Margin             | Medium Margin            | 3 
Lobulation         | No Lobulation            | 1 
Spiculation        | No Spiculation           | 1 
Texture            | Solid                    | 5 
Malignancy         | Moderately Unlikely      | 2 
'''

def getNoduleSize(nodule):
    # take largest dimension over all annotations
    bb = 0
    for ann in nodule:
        bb = max(bb, max(ann.bbox_dimensions()))
    return bb


# ----- Main -----

nodSize  = []
pixSpace = []
for scan in pl.query(pl.Scan).all():
# cycle 1018 scans
    nods = scan.cluster_annotations()
    if len(nods) > 0:
        print("Scan of patient {}: {} nodules.".format(scan.patient_id,len(nods)))
        nodSize  = nodSize  + [getNoduleSize(nod) for nod in nods]
        pixSpace = pixSpace + [scan.pixel_spacing]

size = np.array(nodSize)
np.save('NoduleSize.npy',size)

pixSpace = np.array(pixSpace)
np.save('PixSpace.npy',pixSpace)

plt.figure(1)
plt.hist(size)
plt.title("Nodule Size. Total of {} nodules found.".format(len(size)))

plt.figure(2)
plt.hist(pixSpace)
plt.title("Pixel Spacing. Total of {} scans.".format(len(pixSpace)))

plt.show()