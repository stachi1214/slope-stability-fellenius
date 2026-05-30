import numpy as np


def calculate_fellenius_factor_of_safety(
    slices,
    c: float,
    phi_deg: float
):
    """
    フェルニウス法により円弧すべりの安全率を計算する。

    Parameters
    ----------
    slices : list of dict
        geometry.py の create_slices() で作成したスライス情報
    c : float
        有効粘着力 c' [kPa = kN/m2]
    phi_deg : float
        有効摩擦角 phi' [degree]

    Returns
    -------
    result : dict
        {
            "Fs": 安全率,
            "sum_resisting": 抵抗項の総和,
            "sum_driving": 滑動項の総和,
            "slices": 計算項目を追加したスライス情報
        }
    """

    if len(slices) == 0:
        return {
            "Fs": None,
            "sum_resisting": 0.0,
            "sum_driving": 0.0,
            "slices": []
        }

    phi_rad = np.deg2rad(phi_deg)

    calculated_slices = []

    sum_resisting = 0.0
    sum_driving = 0.0

    for s in slices:
        width = s["x_right"] - s["x_left"]
        dy_base = s["y_circle_right"] - s["y_circle_left"]

        # 底面を直線近似した長さ
        base_length = np.sqrt(width**2 + dy_base**2)

        # 底面角 alpha
        alpha_rad = np.arctan2(dy_base, width)
        alpha_deg = np.rad2deg(alpha_rad)

        W = s["weight"]

        # 抵抗項
        cohesion_resistance = c * base_length
        friction_resistance = W * np.cos(alpha_rad) * np.tan(phi_rad)
        resisting = cohesion_resistance + friction_resistance

        # 滑動項
        driving = W * np.sin(alpha_rad)

        sum_resisting += resisting
        sum_driving += driving

        s_new = s.copy()
        s_new.update({
            "base_length": base_length,
            "alpha_deg": alpha_deg,
            "alpha_rad": alpha_rad,
            "cohesion_resistance": cohesion_resistance,
            "friction_resistance": friction_resistance,
            "resisting": resisting,
            "driving": driving
        })

        calculated_slices.append(s_new)

    if abs(sum_driving) < 1.0e-12:
        Fs = None
    else:
        Fs = sum_resisting / sum_driving

    return {
        "Fs": Fs,
        "sum_resisting": sum_resisting,
        "sum_driving": sum_driving,
        "slices": calculated_slices
    }