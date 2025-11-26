"""
Head orientation filtering and fusion utilities.

This module normalizes head orientation derived from different keypoint
sources (eyes, nose) within the camera coordinate frame, and produces a
single, temporally-stable forward direction for downstream posture
classification.
"""
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.logger import logger


EPS = 1e-8
DEG = np.pi / 180.0


def _norm(vec: np.ndarray) -> np.ndarray:
    """Return a unit vector (avoiding division by zero)."""
    vec = np.asarray(vec, dtype=float)
    length = float(np.linalg.norm(vec))
    return vec / (length if length > EPS else EPS)


def _reorthonormalize(R: np.ndarray) -> np.ndarray:
    """Project to nearest proper rotation (det=+1) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0.0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Matrix logarithm on SO(3) -> axis-angle vector (robust near pi)."""
    R = np.asarray(R, dtype=float)
    # Clamp trace to legal range to mitigate drift
    tr = float(np.clip(np.trace(R), -1.0, 3.0))
    cos_theta = float(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-6:
        return np.zeros(3, dtype=float)

    # Handle neighborhood of pi where sin(theta) ~ 0
    if np.pi - theta < 1e-6:
        ax = np.sqrt(max((R[0, 0] + 1.0) * 0.5, 0.0))
        ay = np.sqrt(max((R[1, 1] + 1.0) * 0.5, 0.0))
        az = np.sqrt(max((R[2, 2] + 1.0) * 0.5, 0.0))
        axis = np.array([ax, ay, az], dtype=float)
        # sign disambiguation from off-diagonals
        if R[0, 1] < 0: axis[1] = -axis[1]
        if R[0, 2] < 0: axis[2] = -axis[2]
        n = float(np.linalg.norm(axis))
        if n < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis /= n
        return axis * theta

    omega_hat = (R - R.T) / (2.0 * np.sin(theta))
    w = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]], dtype=float) * theta
    return w


def _exp_so3(w: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> rotation matrix (Rodrigues)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-6:
        return np.eye(3, dtype=float)
    k = w / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=float
    )
    R = np.eye(3, dtype=float) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return _reorthonormalize(R)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (in degrees) between two vectors."""
    v1 = _norm(v1)
    v2 = _norm(v2)
    cos_theta = np.clip(float(np.dot(v1, v2)), -1.0, 1.0)
    return float(np.arccos(cos_theta) / DEG)


def slerp_R(Ra: np.ndarray, Rb: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two rotation matrices."""
    t = float(np.clip(t, 0.0, 1.0))
    dR = Ra.T @ Rb
    omega = _log_so3(dR)
    R = Ra @ _exp_so3(t * omega)
    return _reorthonormalize(R)


class HeadOrientationFusion:
    """
    Fuse head orientations coming from eyes / nose observations into one
    canonical orientation with confidence-aware arbitration.
    """

    def __init__(
        self,
        tau_agree_deg: float = 12.0,
        tau_conflict_deg: float = 25.0,
        bias_learn_deg: float = 8.0,
        bias_beta: float = 0.02,
        temporal_alpha: float = 0.5,
    ):
        self.tau_agree = float(tau_agree_deg)
        self.tau_conflict = float(tau_conflict_deg)
        self.tau_bias = float(bias_learn_deg)
        self.bias_beta = float(bias_beta)
        self.temporal_alpha = float(temporal_alpha)

        self.R_prev: Optional[np.ndarray] = None
        self.R_bias: np.ndarray = np.eye(3, dtype=float)  # nose -> eyes

    def _learn_bias(self, Re: np.ndarray, Rn: np.ndarray, delta_deg: float) -> None:
        """Online estimation of nose->eyes bias when the two agree."""
        if delta_deg > self.tau_bias:
            return
        target = Re @ Rn.T
        correction = _log_so3(self.R_bias.T @ target)
        self.R_bias = _reorthonormalize(self.R_bias @ _exp_so3(self.bias_beta * correction))

    def _conf_weight(self, value: Optional[float], default: float = 0.6) -> float:
        if value is None:
            return default
        try:
            val = float(value)
        except (TypeError, ValueError):
            return default
        return float(np.clip(val, 0.05, 1.0))

    def step(
        self,
        Re: Optional[np.ndarray] = None,
        Rn: Optional[np.ndarray] = None,
        conf_eye: Optional[float] = None,
        conf_nose: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
        """
        Fuse rotations derived from eyes/nose observations.

        Returns:
            fused rotation (3x3) and diagnostic information.
        """
        info: Dict[str, Optional[float]] = {
            "state": "unknown",
            "delta_deg": None,
            "we": None,
            "wn": None,
            "blend_t": None,  # actual nose slerp weight used in this step
            "dominant": None,
            "bias_angle": float(np.linalg.norm(_log_so3(self.R_bias)) / DEG),
            "continuity_angle": None,
        }

        have_e = Re is not None
        have_n = Rn is not None

        if not have_e and not have_n:
            if self.R_prev is None:
                return np.eye(3, dtype=float), info
            info["state"] = "hold_prev"
            info["dominant"] = "history"
            info["continuity_angle"] = 0.0
            return self.R_prev, info

        if have_e:
            Re = np.asarray(Re, dtype=float)
        if have_n:
            Rn = np.asarray(Rn, dtype=float)

        R_candidate: np.ndarray
        dominant: Optional[str] = None

        if have_e and not have_n:
            info["state"] = "eyes_only"
            dominant = "eyes"
            R_candidate = Re
            info["blend_t"] = 0.0
        elif have_n and not have_e:
            info["state"] = "nose_only"
            dominant = "nose"
            R_candidate = _reorthonormalize(self.R_bias @ Rn)
            info["blend_t"] = 1.0
        else:
            # Both observations are available.
            Rn_corr = _reorthonormalize(self.R_bias @ Rn)
            delta = _angle_between(Re[:, 2], Rn_corr[:, 2])
            info["delta_deg"] = delta

            # Bias learning then re-apply
            self._learn_bias(Re, Rn, delta)
            Rn_corr = _reorthonormalize(self.R_bias @ Rn)
            delta = _angle_between(Re[:, 2], Rn_corr[:, 2])
            info["delta_deg"] = delta

            we = self._conf_weight(conf_eye)
            wn = self._conf_weight(conf_nose)
            info["we"], info["wn"] = we, wn
            weight_nose = wn / (we + wn + EPS)

            if delta <= self.tau_agree:
                info["state"] = "agree_blend"
                dominant = "blend"
                R_candidate = slerp_R(Re, Rn_corr, weight_nose)
                info["blend_t"] = weight_nose
            elif delta <= self.tau_conflict:
                info["state"] = "mild_conflict_blend"
                if we >= wn:
                    weight_nose *= 0.5
                    dominant = "eyes"
                else:
                    weight_nose = 0.5 + 0.5 * weight_nose
                    dominant = "nose"
                R_candidate = slerp_R(Re, Rn_corr, float(np.clip(weight_nose, 0.0, 1.0)))
                info["blend_t"] = float(np.clip(weight_nose, 0.0, 1.0))
            else:
                info["state"] = "conflict_hold"
                dominant = "history"
                if self.R_prev is None:
                    R_candidate = Re if we >= wn else Rn_corr
                    dominant = "eyes" if we >= wn else "nose"
                else:
                    target = Re if we >= wn else Rn_corr
                    R_candidate = slerp_R(self.R_prev, target, self.temporal_alpha)
                info["blend_t"] = None

        # Temporal smoothing with history.
        if self.R_prev is None:
            self.R_prev = _reorthonormalize(R_candidate)
            info["continuity_angle"] = 0.0
        else:
            continuity = _angle_between(self.R_prev[:, 2], R_candidate[:, 2])
            R_candidate = slerp_R(self.R_prev, R_candidate, self.temporal_alpha)
            info["continuity_angle"] = continuity
            self.R_prev = _reorthonormalize(R_candidate)

        info["dominant"] = dominant
        return self.R_prev, info


class HeadOrientationFilter:
    """
    Compute head orientation from 3D keypoints and provide a fused,
    temporally-stable forward vector.
    """

    def __init__(
        self,
        eyes_missing_hysteresis: int = 3,
        eyes_recover_hysteresis: int = 2,
        slerp_alpha: float = 0.3,
    ):
        self.eyes_missing_hysteresis = int(eyes_missing_hysteresis)
        self.eyes_recover_hysteresis = int(eyes_recover_hysteresis)

        self.miss_eye_frames = 0
        self.recover_eye_frames = 0

        self.fusion = HeadOrientationFusion(temporal_alpha=float(slerp_alpha))

    def update(
        self, keypoints_3d: Dict[str, Optional[np.ndarray]], keypoints_conf: Optional[Dict]
    ) -> Optional[Dict]:
        left_shoulder = keypoints_3d.get("left_shoulder")
        right_shoulder = keypoints_3d.get("right_shoulder")
        if left_shoulder is None or right_shoulder is None:
            return None

        left_eye = keypoints_3d.get("left_eye_center")
        right_eye = keypoints_3d.get("right_eye_center")
        nose = keypoints_3d.get("nose")

        eyes_available = left_eye is not None and right_eye is not None
        nose_available = nose is not None

        # hysteresis counters
        if eyes_available:
            self.miss_eye_frames = 0
            self.recover_eye_frames += 1
        else:
            self.miss_eye_frames += 1
            self.recover_eye_frames = 0

        # nominal mode (used only for reporting fallback preference)
        mode = "eyes"
        if not eyes_available and nose_available:
            if self.miss_eye_frames >= self.eyes_missing_hysteresis:
                mode = "nose"
        elif eyes_available and nose_available:
            if self.recover_eye_frames < self.eyes_recover_hysteresis:
                mode = "nose"
        elif not nose_available and not eyes_available:
            return None

        # build frames
        R_eyes = None
        if eyes_available:
            R_eyes = self._build_head_frame_eyes(
                left_eye, right_eye, left_shoulder, right_shoulder, nose
            )
            if R_eyes is not None:
                R_eyes = self._enforce_axes(R_eyes, left_shoulder, right_shoulder, nose)

        R_nose = None
        if nose_available:
            R_nose = self._build_head_frame_nose(nose, left_shoulder, right_shoulder)
            if R_nose is not None:
                R_nose = self._enforce_axes(R_nose, left_shoulder, right_shoulder, nose)

        if R_eyes is None and R_nose is None:
            return None

        conf_eye = self._estimate_eye_weight(
            left_eye, right_eye, left_shoulder, right_shoulder, keypoints_conf
        )
        conf_nose = self._estimate_nose_weight(nose, left_shoulder, right_shoulder)

        R_fused, fusion_info = self.fusion.step(
            Re=R_eyes, Rn=R_nose, conf_eye=conf_eye, conf_nose=conf_nose
        )

        forward = -R_fused[:, 2]

        # simplify mode label for output
        mode_simple = "blend"
        dominant = fusion_info.get("dominant")
        state = fusion_info.get("state", "unknown")
        if dominant in ("eyes", "nose", "history"):
            mode_simple = dominant if dominant != "history" else mode
        elif state == "eyes_only":
            mode_simple = "eyes"
        elif state == "nose_only":
            mode_simple = "nose"
        elif state == "conflict_hold":
            mode_simple = mode

        quality = {
            "eye_nose_angle": fusion_info.get("delta_deg"),
            "nose_weight": fusion_info.get("wn"),
            "eye_weight": fusion_info.get("we"),
            "blend_t": fusion_info.get("blend_t"),
            "continuity_angle": fusion_info.get("continuity_angle"),
            "bias_angle": fusion_info.get("bias_angle"),
            "nose_used": mode_simple == "nose",
            "state": state,
        }

        return {
            "rotation": R_fused,
            "forward": forward,
            "mode": mode_simple,
            "state": state,
            "quality": quality,
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _estimate_eye_weight(
        self,
        left_eye: Optional[np.ndarray],
        right_eye: Optional[np.ndarray],
        left_shoulder: Optional[np.ndarray],
        right_shoulder: Optional[np.ndarray],
        quality_flags: Optional[Dict],
    ) -> float:
        if left_eye is None or right_eye is None:
            return 0.0

        base = 0.7
        if quality_flags:
            if quality_flags.get("left_eye_fallback") or quality_flags.get("right_eye_fallback"):
                base *= 0.4

        if left_shoulder is not None and right_shoulder is not None:
            shoulder_span = float(np.linalg.norm(right_shoulder - left_shoulder))
            eye_span = float(np.linalg.norm(right_eye - left_eye))
            if shoulder_span > 1e-6:
                ratio = np.clip(eye_span / shoulder_span, 0.25, 1.5)
                base *= float(ratio)

        return float(np.clip(base, 0.1, 1.0))

    def _estimate_nose_weight(
        self,
        nose: Optional[np.ndarray],
        left_shoulder: Optional[np.ndarray],
        right_shoulder: Optional[np.ndarray],
    ) -> float:
        if nose is None or left_shoulder is None or right_shoulder is None:
            return 0.0

        base = 0.6
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        distance = float(np.linalg.norm(nose - shoulder_center))
        if distance > 1e-6:
            base *= float(np.clip(distance / 0.18, 0.5, 1.4))  # 0.18m: typical head center->nose scale
        return float(np.clip(base, 0.1, 1.0))

    def _build_head_frame_eyes(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        left_shoulder: np.ndarray,
        right_shoulder: np.ndarray,
        nose: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        # X axis: left -> right
        x_axis = right_eye - left_eye
        if np.linalg.norm(x_axis) < 1e-6:
            return None
        x_axis = _norm(x_axis)

        eye_center = 0.5 * (left_eye + right_eye)
        z_axis = None

        # Prefer nose vector projected to X-orthogonal plane (Z is backward)
        if nose is not None:
            v_n = nose - eye_center
            if np.linalg.norm(v_n) > 1e-6:
                v_n = _norm(v_n)
                v_n_perp = v_n - np.dot(v_n, x_axis) * x_axis
                if np.linalg.norm(v_n_perp) > 1e-6:
                    z_axis = -_norm(v_n_perp)

        # Fallback: use "up guess" from eye_center -> shoulder_center (more stable than shoulder_vec)
        if z_axis is None:
            sh_center = 0.5 * (left_shoulder + right_shoulder)
            up_guess = eye_center - sh_center
            if np.linalg.norm(up_guess) < 1e-6:
                up_guess = np.array([0.0, 1.0, 0.0], dtype=float)  # final fallback
            else:
                up_guess = _norm(up_guess)
            z_temp = np.cross(x_axis, up_guess)
            if np.linalg.norm(z_temp) < 1e-6:
                # Extreme degeneracy: try another canonical up
                up_guess = np.array([0.0, 0.0, 1.0], dtype=float)
                z_temp = np.cross(x_axis, up_guess)
                if np.linalg.norm(z_temp) < 1e-6:
                    return None
            z_axis = _norm(z_temp)

        y_axis = np.cross(z_axis, x_axis)
        if np.linalg.norm(y_axis) < 1e-6:
            return None
        y_axis = _norm(y_axis)
        z_axis = _norm(np.cross(x_axis, y_axis))

        return _reorthonormalize(np.column_stack([x_axis, y_axis, z_axis]))

    def _build_head_frame_nose(
        self,
        nose: np.ndarray,
        left_shoulder: np.ndarray,
        right_shoulder: np.ndarray,
    ) -> Optional[np.ndarray]:
        # X axis: left -> right (shoulders)
        x_axis = right_shoulder - left_shoulder
        if np.linalg.norm(x_axis) < 1e-6:
            return None
        x_axis = _norm(x_axis)

        sh_center = 0.5 * (left_shoulder + right_shoulder)
        v_n = nose - sh_center
        if np.linalg.norm(v_n) < 1e-6:
            return None
        v_n = _norm(v_n)
        v_n_perp = v_n - np.dot(v_n, x_axis) * x_axis
        if np.linalg.norm(v_n_perp) < 1e-6:
            return None
        z_axis = -_norm(v_n_perp)  # backward

        y_axis = np.cross(z_axis, x_axis)
        if np.linalg.norm(y_axis) < 1e-6:
            return None
        y_axis = _norm(y_axis)
        z_axis = _norm(np.cross(x_axis, y_axis))

        return _reorthonormalize(np.column_stack([x_axis, y_axis, z_axis]))

    def _enforce_axes(
        self,
        R: np.ndarray,
        left_shoulder: np.ndarray,
        right_shoulder: np.ndarray,
        nose: Optional[np.ndarray],
    ) -> np.ndarray:
        """Ensure X aligns with shoulders and Z opposes nose direction."""
        R = np.asarray(R, dtype=float)
        shoulder_vec = _norm(right_shoulder - left_shoulder)

        # Rule 1: X must align with shoulder (left->right). Flip X & Z if violated (keep det=+1).
        if float(np.dot(R[:, 0], shoulder_vec)) < 0.0:
            R[:, 0] = -R[:, 0]
            R[:, 2] = -R[:, 2]

        # Rule 2: Z must be backward (oppose nose "forward" direction) if nose is available.
        if nose is not None:
            head_center = 0.5 * (left_shoulder + right_shoulder)
            nose_vec = nose - head_center
            if np.linalg.norm(nose_vec) > 1e-6:
                nose_vec = _norm(nose_vec)
                if float(np.dot(R[:, 2], nose_vec)) > 0.0:
                    # flip Y & Z (keep right-handed)
                    R[:, 1] = -R[:, 1]
                    R[:, 2] = -R[:, 2]

        R = _reorthonormalize(R)
        return R
