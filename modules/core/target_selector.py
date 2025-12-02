"""
Target Selection Module - Nearest Person Selector
==================================================

Selects the primary target person based on shoulder midpoint distance,
with hysteresis to prevent rapid switching between targets.

Features:
- Distance-based selection (shoulder midpoint to camera)
- Confidence filtering
- Switch hysteresis (time-based or frame-based)
- Lost target tolerance
- Current target bonus (sticky selection)
- Optional center bias
"""
from __future__ import annotations

import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .logger import logger


class TargetState(Enum):
    """Target selection state machine"""
    NO_TARGET = "no_target"
    TRACKING = "tracking"
    CANDIDATE = "candidate"  # Potential switch, waiting for hysteresis


@dataclass
class CandidateInfo:
    """Tracks a candidate for potential switch"""
    person_id: int
    first_seen_ms: int
    consecutive_frames: int = 1


@dataclass
class TargetSelectorConfig:
    """Configuration for target selection"""
    # Distance range (meters)
    distance_min_m: float = 0.3
    distance_max_m: float = 1.5

    # Confidence threshold
    confidence_min: float = 0.5

    # Switch hysteresis (use time OR frames)
    switch_time_ms: int = 1000      # Time threshold for switching
    switch_frames: int = 8          # Frame threshold (fallback)
    use_time_threshold: bool = True # True=time, False=frames

    # Lost target tolerance
    lost_time_ms: int = 500
    lost_frames: int = 5

    # Distance epsilon for "same distance" comparison
    epsilon_m: float = 0.08

    # Current target bonus (subtracted from distance)
    current_bonus_m: float = 0.03

    # Center bias
    center_bias: bool = True
    center_weight: float = 0.02  # Weight for center deviation penalty

    # Logging
    log_switches: bool = True
    log_candidates: bool = False  # Debug level


class NearestPersonSelector:
    """
    Selects the nearest valid person as the primary target.

    Uses shoulder midpoint distance with hysteresis to prevent
    rapid switching between multiple people.

    Usage:
        selector = NearestPersonSelector(config)

        # Each frame:
        target = selector.select(candidates, frame_size, timestamp_ms)
        if target:
            process_target(target)
    """

    def __init__(self, config: Optional[TargetSelectorConfig] = None):
        """
        Initialize selector.

        Args:
            config: Selection configuration. Uses defaults if None.
        """
        self.config = config or TargetSelectorConfig()

        # State
        self.state = TargetState.NO_TARGET
        self.current_target_id: Optional[int] = None
        self.current_target_distance: Optional[float] = None

        # Candidate tracking (for switch hysteresis)
        self.candidate: Optional[CandidateInfo] = None

        # Lost target tracking
        self.last_seen_ms: int = 0
        self.lost_frames: int = 0

        # Statistics
        self.switch_count: int = 0
        self.total_frames: int = 0

        logger.info(
            f"NearestPersonSelector initialized: "
            f"distance=[{self.config.distance_min_m}, {self.config.distance_max_m}]m, "
            f"switch_time={self.config.switch_time_ms}ms, "
            f"epsilon={self.config.epsilon_m}m"
        )

    def select(
        self,
        candidates: List[Dict[str, Any]],
        frame_size: Optional[Tuple[int, int]] = None,
        timestamp_ms: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Select the primary target from candidates.

        Args:
            candidates: List of person detections, each containing:
                - person_id: int
                - shoulder_midpoint_3d: (x, y, z) in meters, or None
                - shoulder_confidence: float [0, 1]
                - bbox_center_2d: (cx, cy) optional
                - bbox: dict with x1, y1, x2, y2
                - depth_valid: bool
                - face_bbox: optional dict
            frame_size: (width, height) for center bias calculation
            timestamp_ms: Current timestamp in milliseconds

        Returns:
            Selected target dict, or None if no valid target
        """
        self.total_frames += 1

        # Default timestamp
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Filter and score candidates
        valid_candidates = self._filter_candidates(candidates)
        scored = self._score_candidates(valid_candidates, frame_size)

        if not scored:
            return self._handle_no_candidates(timestamp_ms)

        # Sort by score (lower is better - distance based)
        scored.sort(key=lambda x: x[1])
        best_candidate, best_score = scored[0]
        best_id = best_candidate['person_id']
        best_distance = best_candidate.get('_distance', None)

        # Log candidates at debug level
        if self.config.log_candidates and len(scored) > 1:
            logger.debug(
                f"[TargetSelector] {len(scored)} candidates: "
                + ", ".join(f"id={c['person_id']} d={c.get('_distance', 0):.2f}m" for c, _ in scored[:3])
            )

        # State machine logic
        if self.state == TargetState.NO_TARGET:
            return self._select_new_target(best_candidate, best_distance, timestamp_ms)

        elif self.state == TargetState.TRACKING:
            # Check if current target is still in candidates
            current_in_candidates = any(
                c['person_id'] == self.current_target_id for c, _ in scored
            )

            if current_in_candidates:
                # Update current target info
                current_candidate = next(
                    c for c, _ in scored if c['person_id'] == self.current_target_id
                )
                self.current_target_distance = current_candidate.get('_distance')
                self.last_seen_ms = timestamp_ms
                self.lost_frames = 0

                # Check if someone else is significantly closer
                if best_id != self.current_target_id:
                    return self._check_switch_candidate(
                        current_candidate, best_candidate, best_distance, timestamp_ms
                    )

                # Continue tracking current
                self.candidate = None
                return current_candidate
            else:
                # Current target lost
                return self._handle_target_lost(best_candidate, best_distance, timestamp_ms)

        elif self.state == TargetState.CANDIDATE:
            # Waiting for switch hysteresis
            return self._process_candidate_state(scored, timestamp_ms)

        return None

    def _filter_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates by validity, distance range, and confidence."""
        valid = []

        for c in candidates:
            # Must have shoulder midpoint
            midpoint = c.get('shoulder_midpoint_3d')
            if midpoint is None:
                continue

            # Check depth validity
            if not c.get('depth_valid', True):
                continue

            # Calculate distance (Euclidean 3D distance from camera origin)
            if isinstance(midpoint, (list, tuple)) and len(midpoint) >= 3:
                x, y, z = midpoint[0], midpoint[1], midpoint[2]
                # Use Euclidean distance: sqrt(x^2 + y^2 + z^2)
                # This properly penalizes off-axis people
                distance = (x**2 + y**2 + z**2) ** 0.5
            else:
                continue

            # Check distance range
            if not (self.config.distance_min_m <= distance <= self.config.distance_max_m):
                continue

            # Check confidence
            confidence = c.get('shoulder_confidence', 1.0)
            if confidence < self.config.confidence_min:
                continue

            # Store computed distance
            c['_distance'] = distance
            valid.append(c)

        return valid

    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        frame_size: Optional[Tuple[int, int]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Score candidates (lower is better)."""
        scored = []

        for c in candidates:
            distance = c['_distance']
            score = distance

            # Current target bonus (reduce score = prefer current)
            if c['person_id'] == self.current_target_id:
                score -= self.config.current_bonus_m

            # Low confidence penalty
            confidence = c.get('shoulder_confidence', 1.0)
            if confidence < 0.7:
                score += 0.05 * (0.7 - confidence)  # Small penalty

            # Center bias (optional)
            if self.config.center_bias and frame_size:
                bbox_center = c.get('bbox_center_2d')
                if bbox_center:
                    frame_cx = frame_size[0] / 2
                    frame_cy = frame_size[1] / 2
                    # Normalized deviation from center
                    dx = abs(bbox_center[0] - frame_cx) / frame_size[0]
                    dy = abs(bbox_center[1] - frame_cy) / frame_size[1]
                    center_penalty = (dx + dy) * self.config.center_weight
                    score += center_penalty

            scored.append((c, score))

        return scored

    def _select_new_target(
        self,
        candidate: Dict[str, Any],
        distance: float,
        timestamp_ms: int
    ) -> Dict[str, Any]:
        """Select a new target when none exists."""
        self.current_target_id = candidate['person_id']
        self.current_target_distance = distance
        self.last_seen_ms = timestamp_ms
        self.lost_frames = 0
        self.state = TargetState.TRACKING
        self.candidate = None

        if self.config.log_switches:
            logger.info(
                f"[TargetSelector] New target: id={self.current_target_id}, "
                f"distance={distance:.2f}m"
            )

        return candidate

    def _check_switch_candidate(
        self,
        current: Dict[str, Any],
        best: Dict[str, Any],
        best_distance: float,
        timestamp_ms: int
    ) -> Dict[str, Any]:
        """Check if we should switch to a closer candidate."""
        current_distance = current.get('_distance', float('inf'))

        # Check if best is significantly closer
        distance_diff = current_distance - best_distance

        if distance_diff < self.config.epsilon_m:
            # Not significantly closer, keep current
            self.candidate = None
            return current

        # Best is closer - start or continue candidate tracking
        best_id = best['person_id']

        if self.candidate and self.candidate.person_id == best_id:
            # Continue tracking this candidate
            self.candidate.consecutive_frames += 1
            elapsed_ms = timestamp_ms - self.candidate.first_seen_ms

            # Check if hysteresis threshold reached
            should_switch = False
            if self.config.use_time_threshold:
                should_switch = elapsed_ms >= self.config.switch_time_ms
            else:
                should_switch = self.candidate.consecutive_frames >= self.config.switch_frames

            if should_switch:
                # Switch to new target
                self.switch_count += 1
                old_id = self.current_target_id
                self.current_target_id = best_id
                self.current_target_distance = best_distance
                self.last_seen_ms = timestamp_ms
                self.candidate = None
                self.state = TargetState.TRACKING

                if self.config.log_switches:
                    logger.info(
                        f"[TargetSelector] Switch: {old_id} -> {best_id}, "
                        f"distance {current_distance:.2f}m -> {best_distance:.2f}m"
                    )

                return best
        else:
            # New candidate
            self.candidate = CandidateInfo(
                person_id=best_id,
                first_seen_ms=timestamp_ms
            )
            self.state = TargetState.CANDIDATE

        # Still tracking current while evaluating candidate
        return current

    def _handle_target_lost(
        self,
        best: Dict[str, Any],
        best_distance: float,
        timestamp_ms: int
    ) -> Optional[Dict[str, Any]]:
        """Handle when current target is lost."""
        self.lost_frames += 1
        elapsed_ms = timestamp_ms - self.last_seen_ms

        # Check lost tolerance
        should_switch = False
        if self.config.use_time_threshold:
            should_switch = elapsed_ms >= self.config.lost_time_ms
        else:
            should_switch = self.lost_frames >= self.config.lost_frames

        if should_switch:
            # Lost too long, switch to best available
            old_id = self.current_target_id

            if self.config.log_switches:
                logger.warning(
                    f"[TargetSelector] Lost target {old_id} "
                    f"(frames={self.lost_frames}, elapsed={elapsed_ms}ms), "
                    f"switching to {best['person_id']}"
                )

            return self._select_new_target(best, best_distance, timestamp_ms)

        # Within tolerance, return None to indicate temporary loss
        # But keep state as TRACKING for quick recovery
        return None

    def _handle_no_candidates(self, timestamp_ms: int) -> None:
        """Handle when no valid candidates exist."""
        if self.state != TargetState.NO_TARGET:
            self.lost_frames += 1
            elapsed_ms = timestamp_ms - self.last_seen_ms

            should_reset = False
            if self.config.use_time_threshold:
                should_reset = elapsed_ms >= self.config.lost_time_ms
            else:
                should_reset = self.lost_frames >= self.config.lost_frames

            if should_reset:
                if self.config.log_switches:
                    logger.warning(
                        f"[TargetSelector] No candidates, reset from tracking {self.current_target_id}"
                    )
                self.state = TargetState.NO_TARGET
                self.current_target_id = None
                self.current_target_distance = None
                self.candidate = None

        return None

    def _process_candidate_state(
        self,
        scored: List[Tuple[Dict[str, Any], float]],
        timestamp_ms: int
    ) -> Optional[Dict[str, Any]]:
        """Process when in CANDIDATE state."""
        # Find current target in list
        current_candidate = None
        for c, _ in scored:
            if c['person_id'] == self.current_target_id:
                current_candidate = c
                break

        if current_candidate is None:
            # Current lost while evaluating candidate
            best, _ = scored[0]
            return self._handle_target_lost(best, best['_distance'], timestamp_ms)

        # Check if candidate is still best
        best, _ = scored[0]
        if self.candidate and best['person_id'] == self.candidate.person_id:
            # Candidate still best, continue evaluation
            return self._check_switch_candidate(
                current_candidate, best, best['_distance'], timestamp_ms
            )
        else:
            # Different best, reset candidate
            self.candidate = None
            self.state = TargetState.TRACKING
            return current_candidate

    def reset(self):
        """Reset selector state."""
        self.state = TargetState.NO_TARGET
        self.current_target_id = None
        self.current_target_distance = None
        self.candidate = None
        self.lost_frames = 0
        logger.info("[TargetSelector] Reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get selector statistics."""
        return {
            'state': self.state.value,
            'current_target_id': self.current_target_id,
            'current_distance': self.current_target_distance,
            'switch_count': self.switch_count,
            'total_frames': self.total_frames
        }
