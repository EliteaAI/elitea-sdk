# import all available tools
from .tickets_tool import FetchTicketsTool, CreateTicketTool
from .backend_reports_tool import GetReportsTool, GetReportByIDTool, CreateExcelReportTool, AddTagToReportTool
from .backend_tests_tool import GetTestsTool, GetTestByIDTool, RunTestByIDTool, CreateBackendTestTool
from .ui_reports_tool import GetUIReportsTool, GetUIReportByIDTool, GetUITestsTool
from .run_ui_test_tool import RunUITestTool
from .update_ui_test_schedule_tool import UpdateUITestScheduleTool
from .create_ui_excel_report_tool import CreateUIExcelReportTool
from .create_ui_test_tool import CreateUITestTool
from .cancel_ui_test_tool import CancelUITestTool

__all__ = [
    {"name": "get_ticket_list", "group": "read", "tool": FetchTicketsTool},
    {"name": "create_ticket", "group": "write", "tool": CreateTicketTool},
    {"name": "get_reports", "group": "read", "tool": GetReportsTool},
    {"name": "get_report_by_id", "group": "read", "tool": GetReportByIDTool},
    {"name": "add_tag_to_report", "group": "write", "tool": AddTagToReportTool},
    {"name": "create_excel_report", "group": "write", "tool": CreateExcelReportTool},
    {"name": "get_tests", "group": "read", "tool": GetTestsTool},
    {"name": "get_test_by_id", "group": "read", "tool": GetTestByIDTool},
    {"name": "run_test_by_id", "group": "execute", "tool": RunTestByIDTool},
    {"name": "create_backend_test", "group": "write", "tool": CreateBackendTestTool},
    {"name": "get_ui_reports", "group": "read", "tool": GetUIReportsTool},
    {"name": "get_ui_report_by_id", "group": "read", "tool": GetUIReportByIDTool},
    {"name": "get_ui_tests", "group": "read", "tool": GetUITestsTool},
    {"name": "run_ui_test", "group": "execute", "tool": RunUITestTool},
    {"name": "update_ui_test_schedule", "group": "write", "tool": UpdateUITestScheduleTool},
    {"name": "create_ui_excel_report", "group": "write", "tool": CreateUIExcelReportTool},
    {"name": "create_ui_test", "group": "write", "tool": CreateUITestTool},
    {"name": "cancel_ui_test", "group": "execute", "tool": CancelUITestTool}
]
