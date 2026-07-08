"""
Dashboard subpackage.

Exposes a read-only Server-Sent Events stream of every event-bus event
so an external TUI (see ``dashboard-tui/`` at the repo root) can observe
Roberta while she is running.

The dashboard is fully optional. When disabled in config, the service is
never started and has zero runtime cost.
"""

from robot.dashboard.service import DashboardService

__all__ = ["DashboardService"]
