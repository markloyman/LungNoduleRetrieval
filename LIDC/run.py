import numpy as np
import pylidc

#scan = pylidc.query(pylidc.Scan).filter(pylidc.Scan.patient_id == 'LIDC-IDRI-0340').first()
# should be [4,4]

scan = pylidc.query(pylidc.Scan).filter(pylidc.Scan.patient_id == 'LIDC-IDRI-0867').first()


#print([len(a) for a in scan.cluster_annotations()])
print([len(a) for a in scan.cluster_annotations(metric='jaccard', tol=0.95, tol_limit=0.7)])

print(np.vstack([a.bbox()[0] for a in [scan.annotations[i] for i in [0,3,4,7]] ]))
print(np.vstack([a.bbox()[1] for a in [scan.annotations[i] for i in [0,3,4,7]] ]))
print(np.vstack([a.bbox()[2] for a in [scan.annotations[i] for i in [0,3,4,7]] ]))

print('-'*10)

print(np.vstack([a.bbox()[0] for a in [scan.annotations[i] for i in [1,2,5,6]] ]))
print(np.vstack([a.bbox()[1] for a in [scan.annotations[i] for i in [1,2,5,6]] ]))
print(np.vstack([a.bbox()[2] for a in [scan.annotations[i] for i in [1,2,5,6]] ]))