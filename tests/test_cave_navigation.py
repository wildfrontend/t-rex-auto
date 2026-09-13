from dino_bot.cave_navigation import (
    DEFAULT_SWIPE_VECTORS,
    DONE,
    RESCAN,
    STUCK,
    SWIPE,
    CaveNavigator,
)


def test_calibrated_sequence_moves_camera_down_then_left() -> None:
    navigator = CaveNavigator()
    first = navigator.next_step(cave_visible=False)
    assert first.kind == SWIPE
    assert first.vector == DEFAULT_SWIPE_VECTORS[0]

    navigator.on_swipe_result(moved=True)
    second = navigator.next_step(cave_visible=False)
    assert second.kind == SWIPE
    assert second.vector == DEFAULT_SWIPE_VECTORS[1]


def test_cave_detection_short_circuits_without_a_tap() -> None:
    step = CaveNavigator().next_step(cave_visible=True)
    assert step.kind == DONE
    assert step.vector is None


def test_dropped_swipe_is_retried_but_successful_swipe_is_not_repeated() -> None:
    navigator = CaveNavigator(max_swipe_failures=1)
    first = navigator.next_step(cave_visible=False)
    navigator.on_swipe_result(moved=False)
    assert navigator.next_step(cave_visible=False) == first
    navigator.on_swipe_result(moved=True)
    assert navigator.next_step(cave_visible=False).vector == DEFAULT_SWIPE_VECTORS[1]


def test_missing_cave_gets_bounded_rescans_then_reports_stuck() -> None:
    navigator = CaveNavigator(max_rescans=2)
    for _ in DEFAULT_SWIPE_VECTORS:
        assert navigator.next_step(cave_visible=False).kind == SWIPE
        navigator.on_swipe_result(moved=True)
    assert navigator.next_step(cave_visible=False).kind == RESCAN
    assert navigator.next_step(cave_visible=False).kind == RESCAN
    assert navigator.next_step(cave_visible=False).kind == STUCK


def test_swipe_vectors_scale_with_frame_width() -> None:
    step = CaveNavigator().next_step(cave_visible=False, frame_width=600)
    assert step.vector == (300, 700, 300, 400)
