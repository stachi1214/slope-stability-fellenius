import numpy as np


def create_slope_geometry(H: float, m: float):
    """
    単純斜面の地表面座標を作成する。

    座標系：
    - 法先を原点 (0, 0)
    - x軸：水平右向き
    - y軸：鉛直上向き
    - 斜面勾配：1:m

    地表面：
    - 前方地盤
    - 斜面
    - 背後地盤
    """

    x_front = -3 * H
    y_front = 0.0

    x_toe = 0.0
    y_toe = 0.0

    x_crest = m * H
    y_crest = H

    x_back = x_crest + 3 * H
    y_back = H

    x_ground = [x_front, x_toe, x_crest, x_back]
    y_ground = [y_front, y_toe, y_crest, y_back]

    return x_ground, y_ground

def create_circle_geometry(xc: float, yc: float, R: float, n_points: int = 400):
    """
    円の座標を作成する。
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    x_circle = xc + R * np.cos(theta)
    y_circle = yc + R * np.sin(theta)

    return x_circle, y_circle

def line_segment_circle_intersections(p1, p2, xc: float, yc: float, R: float, tol: float = 1.0e-10):
    """
    線分 p1-p2 と円の交点を求める。

    Parameters
    ----------
    p1, p2 : tuple
        線分の端点 (x, y)
    xc, yc : float
        円の中心座標
    R : float
        円の半径
    tol : float
        数値誤差判定用の許容値

    Returns
    -------
    intersections : list of tuple
        線分上に存在する交点 [(x, y), ...]
    """

    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    # 線分を媒介変数 t で表す：
    # x = x1 + t dx
    # y = y1 + t dy
    # ただし 0 <= t <= 1 が線分上
    fx = x1 - xc
    fy = y1 - yc

    a = dx**2 + dy**2
    b = 2.0 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - R**2

    discriminant = b**2 - 4.0 * a * c

    intersections = []

    # 判別式が負なら交点なし
    if discriminant < -tol:
        return intersections

    # 判別式がほぼゼロなら接している
    if abs(discriminant) <= tol:
        t = -b / (2.0 * a)

        if -tol <= t <= 1.0 + tol:
            x = x1 + t * dx
            y = y1 + t * dy
            intersections.append((x, y))

        return intersections

    # 判別式が正なら2交点
    sqrt_discriminant = np.sqrt(discriminant)

    t1 = (-b - sqrt_discriminant) / (2.0 * a)
    t2 = (-b + sqrt_discriminant) / (2.0 * a)

    for t in [t1, t2]:
        if -tol <= t <= 1.0 + tol:
            x = x1 + t * dx
            y = y1 + t * dy
            intersections.append((x, y))

    return intersections


def find_ground_circle_intersections(H: float, m: float, xc: float, yc: float, R: float):
    """
    地表面とすべり円の交点を求める。

    現在は単純斜面を対象とする：
    - 法先 (0, 0)
    - 法肩 (mH, H)
    - 背後地盤 (mH + H, H)

    Returns
    -------
    intersections : list of dict
        交点情報のリスト。
        各要素は
        {
            "x": x座標,
            "y": y座標,
            "segment": 線分名
        }
    """

    x_ground, y_ground = create_slope_geometry(H, m)

    points = list(zip(x_ground, y_ground))

    segment_names = [
        "front_ground",
        "slope_face",
        "back_ground"
    ]

    intersections = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        pts = line_segment_circle_intersections(p1, p2, xc, yc, R)

        for x, y in pts:
            intersections.append({
                "x": x,
                "y": y,
                "segment": segment_names[i]
            })

    # 重複点を除去する
    # 例えば、法肩ちょうどで交わる場合、2つの線分から同じ交点が検出される可能性がある
    unique_intersections = []

    for p in intersections:
        is_duplicate = False

        for q in unique_intersections:
            if abs(p["x"] - q["x"]) < 1.0e-8 and abs(p["y"] - q["y"]) < 1.0e-8:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_intersections.append(p)

    # x座標の小さい順に並べる
    unique_intersections.sort(key=lambda p: p["x"])

    return unique_intersections

def ground_y_at_x(x: float, H: float, m: float):
    """
    指定した x 座標における地表面の y 座標を返す。

    地表面：
    - 前方地盤：y = 0
    - 斜面：y = x / m
    - 背後地盤：y = H
    """

    x_toe = 0.0
    x_crest = m * H

    if x <= x_toe:
        return 0.0
    elif x <= x_crest:
        return x / m
    else:
        return H


def circle_lower_y_at_x(x: float, xc: float, yc: float, R: float):
    """
    指定した x 座標における円の下側の y 座標を返す。

    円の式：
        (x - xc)^2 + (y - yc)^2 = R^2

    下側の円弧：
        y = yc - sqrt(R^2 - (x - xc)^2)
    """

    value = R**2 - (x - xc)**2

    if value < 0:
        return None

    return yc - np.sqrt(value)


def create_slice_boundaries(
    H: float,
    m: float,
    xc: float,
    yc: float,
    R: float,
    intersections,
    n_slices: int
):
    """
    すべり土塊を水平スライスに分割するための境界線データを作成する。

    Parameters
    ----------
    H : float
        斜面高さ
    m : float
        斜面勾配 1:m
    xc, yc : float
        円中心座標
    R : float
        円半径
    intersections : list
        地表面と円の交点リスト
    n_slices : int
        スライス数

    Returns
    -------
    slice_boundaries : list of dict
        各スライス境界線の情報
    """

    if len(intersections) != 2:
        return []

    # x座標の小さい方を左、大きい方を右とする
    p_left = intersections[0]
    p_right = intersections[1]

    x_left = p_left["x"]
    x_right = p_right["x"]

    if x_right <= x_left:
        return []

    x_values = np.linspace(x_left, x_right, n_slices + 1)

    slice_boundaries = []

    for x in x_values:
        y_ground = ground_y_at_x(x, H, m)
        y_circle = circle_lower_y_at_x(x, xc, yc, R)

        if y_circle is None:
            continue

        slice_boundaries.append({
            "x": x,
            "y_ground": y_ground,
            "y_circle": y_circle
        })

    return slice_boundaries

def create_slices(
    H: float,
    m: float,
    xc: float,
    yc: float,
    R: float,
    intersections,
    n_slices: int,
    gamma: float
):
    """
    スライス分割に基づき、各スライスの面積と重量を計算する。

    ここでは、すべり円弧を各スライス内で直線近似し、
    台形として面積を計算する。

    Parameters
    ----------
    H : float
        斜面高さ
    m : float
        斜面勾配 1:m
    xc, yc : float
        円中心座標
    R : float
        円半径
    intersections : list
        地表面と円の交点リスト
    n_slices : int
        スライス数
    gamma : float
        単位体積重量 kN/m3

    Returns
    -------
    slices : list of dict
        各スライスの情報
    """

    if len(intersections) != 2:
        return []

    # 念のため x 座標で並べる
    intersections_sorted = sorted(intersections, key=lambda p: p["x"])

    x_left_all = intersections_sorted[0]["x"]
    x_right_all = intersections_sorted[1]["x"]

    if x_right_all <= x_left_all:
        return []

    x_values = np.linspace(x_left_all, x_right_all, n_slices + 1)

    slices = []

    for i in range(n_slices):
        x_left = x_values[i]
        x_right = x_values[i + 1]

        y_ground_left = ground_y_at_x(x_left, H, m)
        y_ground_right = ground_y_at_x(x_right, H, m)

        y_circle_left = circle_lower_y_at_x(x_left, xc, yc, R)
        y_circle_right = circle_lower_y_at_x(x_right, xc, yc, R)

        if y_circle_left is None or y_circle_right is None:
            continue

        h_left = y_ground_left - y_circle_left
        h_right = y_ground_right - y_circle_right

        # 高さが負になる場合は、すべり円が地表面より上にある可能性がある
        if h_left < 0 or h_right < 0:
            continue

        width = x_right - x_left

        area = 0.5 * (h_left + h_right) * width
        weight = gamma * area

        x_mid = 0.5 * (x_left + x_right)

        slices.append({
            "i": i + 1,
            "x_left": x_left,
            "x_right": x_right,
            "x_mid": x_mid,
            "width": width,
            "y_ground_left": y_ground_left,
            "y_ground_right": y_ground_right,
            "y_circle_left": y_circle_left,
            "y_circle_right": y_circle_right,
            "h_left": h_left,
            "h_right": h_right,
            "area": area,
            "weight": weight
        })

    return slices