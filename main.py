from ultralytics import YOLO
import numpy as np
import math


def build_th_vectors(detections, th_id):
    shape = [
        (0, 4), (1, 2), (2, 2), (7, 2), (8, 5),
        (10, 1), (15, 1), (19, 4), (20, 2), (23, 4),
        (25, 2), (28, 3), (30, 1), (31, 4), (32, 3),
        (33, 1), (35, 3), (36, 2), (38, 2), (40, 5), (42, 4)
    ]

    th_x, th_y = next(((x, y)
                      for x, y, cls in detections if cls == th_id), (-1, -1))

    by_cls = {}
    for x, y, cls in detections:
        if cls not in {cls for cls, _ in shape}:
            continue
        angle = math.atan2(x - th_x, y - th_y)
        by_cls.setdefault(cls, []).append((math.sin(angle), math.cos(angle)))

    def rotate(items, degrees):
        if degrees == 90:
            return [(cos, -sin) for sin, cos in items]
        if degrees == 180:
            return [(-sin, -cos) for sin, cos in items]
        if degrees == 270:
            return [(-cos, sin) for sin, cos in items]
        return items

    def flatten(pairs):
        return [value for pair in pairs for value in pair]

    result = []
    for degree in (0, 90, 180, 270):
        vector = []
        for cls, count in shape:
            items = by_cls.get(cls, [])[:count]
            items += [(0.0, 0.0)] * max(0, count - len(items))
            vector.extend(
                flatten(sorted(rotate(items, degree), key=lambda x: (x[0], x[1]))))
        result.append(vector)

    return result


model = YOLO("./Dragonsight_1.1.pt")
image1 = "./validation/181_home.jpg"
image2 = "./validation/181_war_scout_180.jpg"


base1 = model(image1)[0]
base2 = model(image2)[0]

th_id = next((k for k, v in model.names.items() if v == 'Town Hall'), None)

centers1 = base1.boxes.xywhn[:, :2].cpu().numpy()  # [x_center, y_center]
classes1 = base1.boxes.cls.cpu().numpy()

centers2 = base2.boxes.xywhn[:, :2].cpu().numpy()  # [x_center, y_center]
classes2 = base2.boxes.cls.cpu().numpy()

detections1 = np.hstack((centers1, classes1.reshape(-1, 1)))
detections2 = np.hstack((centers2, classes2.reshape(-1, 1)))

v1 = build_th_vectors(detections1, th_id)[0]
v2_list = build_th_vectors(detections2, th_id)

list_cosine_similarities = []
for v2 in v2_list:
    cosine_similarity = np.dot(
        v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    list_cosine_similarities.append(cosine_similarity)
print(f"Cosine Similarity: {max(list_cosine_similarities):.4f}")
