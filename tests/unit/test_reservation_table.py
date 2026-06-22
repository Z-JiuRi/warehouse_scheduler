"""Unit tests for reservation table."""

import pytest
from app.tools.reservation_table import ReservationTable
from app.domain.path_models import TimedPosition


class TestReservationTable:
    @pytest.fixture
    def table(self):
        return ReservationTable()

    def test_vertex_reservation(self, table):
        table.reserve_vertex(0, 0, 5, "R1")
        assert table.is_vertex_reserved(0, 0, 5)
        assert not table.is_vertex_reserved(0, 0, 6)
        assert not table.is_vertex_reserved(1, 0, 5)

    def test_edge_reservation(self, table):
        table.reserve_edge(0, 0, 1, 0, 5, "R1")
        assert table.is_edge_reserved(0, 0, 1, 0, 5)
        assert not table.is_edge_reserved(0, 0, 1, 0, 6)
        # Reverse edge should NOT be reserved
        assert not table.is_edge_reserved(1, 0, 0, 0, 5)

    def test_reserve_path(self, table):
        path = [
            TimedPosition(0, 0, 0),
            TimedPosition(1, 0, 1),
            TimedPosition(2, 0, 2),
        ]
        table.reserve_path(path, "R1")
        assert table.is_vertex_reserved(0, 0, 0)
        assert table.is_vertex_reserved(1, 0, 1)
        assert table.is_vertex_reserved(2, 0, 2)
        # Edge from (0,0) to (1,0) at t=0
        assert table.is_edge_reserved(0, 0, 1, 0, 0)

    def test_reserve_path_endpoint_padding(self, table):
        path = [TimedPosition(0, 0, 0), TimedPosition(1, 0, 1)]
        table.reserve_path(path, "R1", max_t=5)
        # Endpoint (1,0) should be reserved for t=2,3,4,5
        assert table.is_vertex_reserved(1, 0, 2)
        assert table.is_vertex_reserved(1, 0, 5)
        assert not table.is_vertex_reserved(1, 0, 6)

    def test_move_safe_no_reservation(self, table):
        assert table.is_move_safe(0, 0, 1, 0, 0, "R1")

    def test_move_safe_vertex_blocked(self, table):
        table.reserve_vertex(1, 0, 1, "R2")
        assert not table.is_move_safe(0, 0, 1, 0, 0, "R1")

    def test_move_safe_edge_blocked(self, table):
        table.reserve_edge(0, 0, 1, 0, 0, "R2")
        assert not table.is_move_safe(0, 0, 1, 0, 0, "R1")

    def test_move_safe_swap_conflict(self, table):
        # R2 is moving from (1,0) to (0,0) at t=0
        table.reserve_edge(1, 0, 0, 0, 0, "R2")
        # R1 wants to move from (0,0) to (1,0) at t=0 -> swap!
        assert not table.is_move_safe(0, 0, 1, 0, 0, "R1")

    def test_wait_safe(self, table):
        table.reserve_vertex(0, 0, 1, "R2")
        assert not table.is_wait_safe(0, 0, 0, "R1")
        assert table.is_wait_safe(0, 0, 0, "R2")  # same robot OK
        assert table.is_wait_safe(1, 0, 0, "R1")

    def test_clear(self, table):
        table.reserve_vertex(0, 0, 0, "R1")
        table.clear()
        assert not table.is_vertex_reserved(0, 0, 0)

    def test_get_vertex_owner(self, table):
        table.reserve_vertex(0, 0, 0, "R1")
        assert table.get_vertex_owner(0, 0, 0) == "R1"
        assert table.get_vertex_owner(1, 0, 0) is None
