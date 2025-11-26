import numpy as np
from typing import Optional, Tuple

# =========================
# One-Euro Filter (核心)
# =========================
class OneEuro:
    """
    经典 One-Euro Filter
    - f_min: 静止时的截止频率(Hz)，越小越稳
    - beta: 速度增敏系数，越大越跟手
    - f_d: 导数滤波频率(Hz)，平滑速度估计
    """
    def __init__(self, f_min=1.2, beta=0.04, f_d=4.0, x0=None, dx0=0.0):
        self.f_min, self.beta, self.f_d = float(f_min), float(beta), float(f_d)
        self.x_hat = None if x0 is None else float(x0)
        self.dx_hat = float(dx0)

    @staticmethod
    def _alpha(f_c, T):
        f_c = max(1e-6, float(f_c))
        T   = max(1e-6, float(T))
        tau = 1.0 / (2.0 * np.pi * f_c)
        return 1.0 / (1.0 + tau / T)

    def reset(self, x0=None, dx0=0.0):
        self.x_hat = None if x0 is None else float(x0)
        self.dx_hat = float(dx0)

    def update(self, x, dt):
        """x: 本帧原始值；dt: 本帧间隔(秒)"""
        x = float(x)
        if self.x_hat is None:
            self.x_hat, self.dx_hat = x, 0.0
            return x

        # 1) 估计并平滑导数
        dx_raw = (x - self.x_hat) / max(1e-6, dt)
        a_d = self._alpha(self.f_d, dt)
        self.dx_hat = a_d * self.dx_hat + (1 - a_d) * dx_raw

        # 2) 根据速度自适应截止频率
        f_c = self.f_min + self.beta * abs(self.dx_hat)
        a = self._alpha(f_c, dt)

        # 3) 主滤波
        self.x_hat = a * self.x_hat + (1 - a) * x
        return self.x_hat


# =========================================
# 姿态角计算（集成 One-Euro 滤波器）
# =========================================
class CalculatePostureAngles:
    def __init__(self,
                 key_points_3d: dict,
                 smoothing_factor: float = 0.9,   # 仅保留作元数据记录/后备
                 oneeuro_params: Optional[dict] = None):
        """
        初始化姿态角度计算器（仅3D模式，内置 One-Euro 滤波）

        Args:
            key_points_3d (dict): 3D关键点 {name: (x,y,z) 或 None}
            smoothing_factor (float): 仅记录不再使用（历史兼容）
            oneeuro_params (dict): 为三个角分别传参：
                {
                  "forward":  {"f_min":1.2, "beta":0.05, "f_d":4.0},
                  "shoulder": {"f_min":1.0, "beta":0.04, "f_d":4.0},
                  "roll":     {"f_min":1.4, "beta":0.06, "f_d":4.0},
                }
        """
        self.key_points_3d = key_points_3d
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))

        # 默认滤波参数（30FPS 场景可作为好起点）
        defaults = {
            "forward":  {"f_min":1.2, "beta":0.05, "f_d":4.0},
            "shoulder": {"f_min":1.0, "beta":0.04, "f_d":4.0},
            "roll":     {"f_min":1.4, "beta":0.06, "f_d":4.0},
        }
        oneeuro_params = oneeuro_params or {}

        def _mk(name):
            p = {**defaults[name], **oneeuro_params.get(name, {})}
            return OneEuro(**p)

        # 为三个角各放一个 One-Euro 实例
        self.filters = {
            "forward": _mk("forward"),
            "shoulder": _mk("shoulder"),
            "roll": _mk("roll"),
        }

        self.angles = {
            "head_forward_angle": None,   # 颈部前倾(负=前倾, 正=后仰) [deg]
            "shoulder_tilt_angle": None,  # 肩膀倾斜(右高为正) [deg]
            "head_roll_angle": None,      # 头部侧倾(右高为正) [deg]
        }

        self._name_map = {
            "forward":  "head_forward_angle",
            "shoulder": "shoulder_tilt_angle",
            "roll":     "head_roll_angle",
        }

        self.calculation_metadata = {
            "head_tilt_used_3d": False,
            "total_3d_points": 0,
            "missing_3d_points": [],
            "smoothing_factor": self.smoothing_factor,  # 仅记录
            "filter": "one-euro",
        }

    # ---------- 对外主入口 ----------
    def get_posture_angles(self, dt: float = None):
        """
        计算三种姿态角并进行 One-Euro 平滑

        参数:
            dt: 本帧时间间隔(秒)。如果为 None，使用默认值 1/30 秒 (30 FPS)
                警告：强烈建议传入实际 dt 以获得最佳滤波效果

        返回:
            dict: {"head_forward_angle": float or None, ...}
        """
        # 向后兼容：如果未传入 dt，使用默认值并记录警告
        if dt is None:
            dt = 1.0 / 30.0  # 假设 30 FPS
            if not hasattr(self, '_dt_warning_logged'):
                import logging
                logging.getLogger(__name__).warning(
                    "get_posture_angles() 未传入 dt 参数，使用默认值 1/30s。"
                    "建议传入实际帧间隔以获得最佳 One-Euro 滤波效果。"
                )
                self._dt_warning_logged = True
        self.calculation_metadata.update({
            "total_3d_points": 0,
            "missing_3d_points": [],
            "dt": float(dt),
        })
        self._analyze_3d_data_availability()

        # 独立计算每个角（失败不影响其他）
        try:
            self._calc_head_forward(dt)
        except Exception as e:
            self.calculation_metadata.setdefault("calculation_errors", []).append(f"head_forward: {e}")

        try:
            self._calc_head_tilt(dt)
        except Exception as e:
            self.calculation_metadata.setdefault("calculation_errors", []).append(f"head_tilt: {e}")

        try:
            self._calc_shoulder_tilt(dt)
        except Exception as e:
            self.calculation_metadata.setdefault("calculation_errors", []).append(f"shoulder_tilt: {e}")

        return self.angles

    def get_metadata(self):
        """获取调试元数据（副本）"""
        return self.calculation_metadata.copy()

    # ---------- 内部工具 ----------
    def _analyze_3d_data_availability(self):
        for key, value in self.key_points_3d.items():
            if key.startswith("_"):  # 元数据字段（如 _head_orientation）
                continue
            if value is not None:
                self.calculation_metadata["total_3d_points"] += 1
            else:
                self.calculation_metadata["missing_3d_points"].append(key)

    def _set_angle_with_filter(self, name: str, raw_value: Optional[float], dt: float,
                               clip_range: Optional[Tuple[float, float]] = None):
        """
        将原始角 raw_value 经过 One-Euro 平滑后写入 self.angles。
        name: "forward"/"shoulder"/"roll"
        clip_range: (min_deg, max_deg) 可选限幅
        """
        key = self._name_map[name]

        if raw_value is None or not np.isfinite(raw_value):
            # 本帧缺失则不更新滤波器（保留上一平滑值）
            return

        if clip_range is not None:
            raw_value = float(np.clip(raw_value, clip_range[0], clip_range[1]))

        if self.angles[key] is None:
            # 首帧：初始化滤波状态，直接赋值更稳
            self.filters[name].reset(x0=float(raw_value))
            self.angles[key] = float(raw_value)
        else:
            self.angles[key] = float(self.filters[name].update(float(raw_value), dt))

        # 可选：统一保留 6 位小数
        self.angles[key] = float(np.round(self.angles[key], 6))

    # ---------- 角度计算（含平滑） ----------
    def _calc_head_forward(self, dt: float):
        """
        颈椎前倾/后仰角：
        - 优先使用 _head_orientation.forward（过滤器融合出的统一前向矢量）
        - 回退：肩中心 -> (双眼中点 或 鼻子) 的颈部向量在 YZ 平面上的偏转
            约定：前倾为负，后仰为正
        """
        # 优先使用统一前向向量
        if "_head_orientation" in self.key_points_3d:
            orientation = self.key_points_3d["_head_orientation"]
            if orientation and "forward" in orientation:
                fwd = np.asarray(orientation["forward"], dtype=float)  # 期望已是单位向量
                # 肩膀必须存在，否则无法判断符号（参考姿态）
                if (self.key_points_3d.get("left_shoulder") is None or
                    self.key_points_3d.get("right_shoulder") is None):
                    return
                vertical = np.array([0.0, 1.0, 0.0], dtype=float)
                dot = float(np.clip(np.dot(fwd, vertical), -1.0, 1.0))
                angle_from_vertical = float(np.degrees(np.arccos(abs(dot))))
                cervical_angle = -angle_from_vertical if fwd[2] > 0 else angle_from_vertical
                cervical_angle = float(np.clip(cervical_angle, -180.0, 180.0))

                # 写入（One-Euro 平滑）
                self._set_angle_with_filter("forward", cervical_angle, dt, clip_range=(-180.0, 180.0))

                # 记录质量信息（可选）
                q = orientation.get("quality", {})
                self.calculation_metadata["head_forward_used_filter"] = True
                self.calculation_metadata["head_forward_mode"] = orientation.get("mode", "unknown")
                self.calculation_metadata["head_forward_state"] = orientation.get("state", "unknown")
                self.calculation_metadata["head_forward_eye_nose_angle"] = q.get("eye_nose_angle")
                self.calculation_metadata["head_forward_nose_weight"] = q.get("nose_weight")
                self.calculation_metadata["head_forward_eye_weight"] = q.get("eye_weight")
                self.calculation_metadata["head_forward_continuity"] = q.get("continuity_angle")
                return

        # 回退：基于颈部向量（肩中心 -> 头部参考点）
        if (self.key_points_3d.get("left_shoulder") is None or
            self.key_points_3d.get("right_shoulder") is None):
            return

        head_ref = None
        if (self.key_points_3d.get("left_eye_center") is not None and
            self.key_points_3d.get("right_eye_center") is not None):
            head_ref = (np.array(self.key_points_3d["left_eye_center"], dtype=float) +
                        np.array(self.key_points_3d["right_eye_center"], dtype=float)) / 2.0
            self.calculation_metadata["head_forward_used_nose"] = False
        elif self.key_points_3d.get("nose") is not None:
            head_ref = np.array(self.key_points_3d["nose"], dtype=float)
            self.calculation_metadata["head_forward_used_nose"] = True
        else:
            return

        shoulder_center = (np.array(self.key_points_3d["left_shoulder"], dtype=float) +
                           np.array(self.key_points_3d["right_shoulder"], dtype=float)) / 2.0
        neck_vec = head_ref - shoulder_center
        neck_yz = np.array([0.0, neck_vec[1], neck_vec[2]], dtype=float)
        mag = float(np.linalg.norm(neck_yz))
        if mag < 1e-2:  # 过小无法稳定计算
            return

        neck_yz_n = neck_yz / mag
        vertical = np.array([0.0, 1.0, 0.0], dtype=float)
        dot = float(np.clip(np.dot(neck_yz_n, vertical), -1.0, 1.0))
        base_angle = float(np.degrees(np.arccos(abs(dot))))
        # 规定：Z<0 视作前倾为正；这里沿用你原代码的符号方案：
        angle_from_vertical = base_angle if neck_yz[2] < 0 else -base_angle
        cervical_angle = float(np.clip(angle_from_vertical, -180.0, 180.0))

        self._set_angle_with_filter("forward", cervical_angle, dt, clip_range=(-180.0, 180.0))

    def _calc_shoulder_tilt(self, dt: float):
        """
        肩膀倾斜角：左右肩连线与水平面的夹角
        正值=右肩高，负值=左肩高
        """
        if (self.key_points_3d.get("left_shoulder") is None or
            self.key_points_3d.get("right_shoulder") is None):
            return

        L = np.array(self.key_points_3d["left_shoulder"], dtype=float)
        R = np.array(self.key_points_3d["right_shoulder"], dtype=float)
        v = R - L
        horiz = np.linalg.norm([v[0], v[2]])
        if horiz <= 0.03:  # 水平分量太小，不可靠
            return

        tilt = float(np.degrees(np.arctan2(v[1], horiz)))
        tilt = float(np.clip(tilt, -40.0, 40.0))
        self._set_angle_with_filter("shoulder", tilt, dt, clip_range=(-40.0, 40.0))

    def _calc_head_tilt(self, dt: float):
        """
        头部侧倾角（roll）：双眼连线与水平面的夹角
        正值=右眼高，负值=左眼高
        """
        if (self.key_points_3d.get("left_eye_center") is None or
            self.key_points_3d.get("right_eye_center") is None):
            return

        L = np.array(self.key_points_3d["left_eye_center"], dtype=float)
        R = np.array(self.key_points_3d["right_eye_center"], dtype=float)
        v = R - L
        horiz = np.linalg.norm([v[0], v[2]])
        if horiz <= 0.02:
            return

        roll = float(np.degrees(np.arctan2(v[1], horiz)))
        roll = float(np.clip(roll, -90.0, 90.0))
        self.calculation_metadata["head_tilt_used_3d"] = True
        self._set_angle_with_filter("roll", roll, dt, clip_range=(-90.0, 90.0))
