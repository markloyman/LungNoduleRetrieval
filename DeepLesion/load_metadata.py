import numpy as np
import csv
from typing import List, Dict


class DeepLesionMetaEntry:
    def __init__(self, dict_row):
        self.Path = {
            "FileName": dict_row['File_name'],
            "Patient": dict_row['Patient_index'],
            "Study": dict_row['Study_index'],
            "Series": dict_row['Series_ID'],
            "DataGroup": dict_row['Train_Val_Test'],
            "Noisy": bool(int(dict_row['Possibly_noisy'])),
        }
        self.Patient = {
            "ID": dict_row['Patient_index'],
            "Gender": dict_row['Patient_gender'],
            'Age': dict_row['Patient_age'],
        }
        self.BoundingBox = self.as_floats(dict_row['Bounding_boxes'])
        self.LesionType = int(dict_row['Coarse_lesion_type'])
        self.Window = self.as_floats(dict_row['DICOM_windows'])
        self.ImageSize = self.as_floats(dict_row['Image_size'])

        self.Slice = {
            'Key': int(dict_row['Key_slice_index']),
            'Range': self.as_floats(dict_row['Slice_range']),
        }
        self.DiameterPx = self.as_floats(dict_row['Lesion_diameters_Pixel_'])
        self.MeasurementCoord = self.as_floats(dict_row['Measurement_coordinates'])
        self.NormalizedLesionLocation = self.as_floats(dict_row['Normalized_lesion_location'])
        self.SpacingMmPx = self.as_floats(dict_row['Spacing_mm_px_'])

    @staticmethod
    def as_floats(text: str) -> List[float]:
        return [float(aa) for aa in text.split(',')]

    def get_bb_size_in_mm(self):
        dx, dy = self.BoundingBox[2] - self.BoundingBox[0], self.BoundingBox[3] - self.BoundingBox[1]
        dl = np.maximum(dx, dy)
        return self.SpacingMmPx[0] * dl

    def get_center(self):
        x = self.BoundingBox[0] + (self.BoundingBox[2] - self.BoundingBox[0]) // 2
        y = self.BoundingBox[1] + (self.BoundingBox[3] - self.BoundingBox[1]) // 2
        return x, y

def load_meta(filename: str, index_field: str, limit=0) -> Dict[str, DeepLesionMetaEntry]:
    meta = dict()
    with open(filename) as f:
        records = csv.DictReader(f)
        for row in records:
            if (limit > 0) and (records.line_num == limit):
                break
            meta[row[index_field][:-4]] = DeepLesionMetaEntry(row)
    return meta


if __name__ == "__main__":
    metadata = load_meta(filename="DL_info.csv", index_field='File_name', limit=20)
    print(type(metadata))
    print(len(metadata))


