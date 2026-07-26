"""CSV and PDF reporting for persisted inspection history."""

import csv
import html
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from industrial_vision_inspector.storage import InspectionRecord

CSV_FIELDS = ("id", "timestamp", "image_path", "result", "confidence", "notes")


@dataclass(frozen=True)
class ReportData:
    """Aggregated inspection data consumed by the single PDF template."""

    start_time: datetime
    end_time: datetime
    total_count: int
    pass_count: int
    fail_count: int
    defect_rate: float
    examples: tuple[InspectionRecord, ...]


def write_inspections_csv(
    records: Iterable[InspectionRecord], output_path: str | Path
) -> Path:
    """Write inspection records in their supplied order and return the output path."""
    path = Path(output_path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.id,
                    "timestamp": _utc_timestamp(record),
                    "image_path": record.image_path,
                    "result": record.result,
                    "confidence": f"{record.confidence:.6f}",
                    "notes": record.notes or "",
                }
            )
    return path


def prepare_report_data(
    records: Iterable[InspectionRecord], *, max_examples: int = 4
) -> ReportData:
    """Aggregate records and choose deterministic examples with existing images."""
    if max_examples < 0:
        raise ValueError("max_examples cannot be negative")
    record_list = list(records)
    if not record_list:
        raise ValueError("cannot create a report without inspections")

    pass_count = sum(record.result == "pass" for record in record_list)
    fail_count = len(record_list) - pass_count
    available = [
        record for record in record_list if Path(record.image_path).is_file()
    ]

    examples: list[InspectionRecord] = []
    for outcome in ("fail", "pass"):
        match = next(
            (record for record in available if record.result == outcome), None
        )
        if match is not None and len(examples) < max_examples:
            examples.append(match)
    for record in available:
        if len(examples) >= max_examples:
            break
        if record not in examples:
            examples.append(record)

    return ReportData(
        start_time=min(record.timestamp for record in record_list),
        end_time=max(record.timestamp for record in record_list),
        total_count=len(record_list),
        pass_count=pass_count,
        fail_count=fail_count,
        defect_rate=fail_count / len(record_list),
        examples=tuple(examples),
    )


def write_inspection_pdf(
    report: ReportData,
    output_path: str | Path,
    *,
    generated_at: datetime | None = None,
) -> Path:
    """Render one fixed inspection-summary PDF template."""
    path = Path(output_path)
    generated = generated_at or datetime.now(timezone.utc)
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=36,
        title="Industrial Vision Inspection Report",
        author="Industrial Vision Inspector",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "InspectionTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#243B53"),
        spaceAfter=8,
    )
    center_style = ParagraphStyle(
        "Centered",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        leading=14,
    )

    story = [
        Paragraph("Industrial Vision Inspection Report", title_style),
        Paragraph(
            f"Generated {_format_utc(generated)}<br/>"
            f"Inspection range: {_format_utc(report.start_time)} to "
            f"{_format_utc(report.end_time)}",
            center_style,
        ),
        Spacer(1, 18),
        _summary_table(report),
        Spacer(1, 18),
        _defect_chart(report),
    ]
    if len(report.examples) >= 3:
        story.append(PageBreak())
    else:
        story.append(Spacer(1, 18))
    story.append(Paragraph("Example inspections", styles["Heading2"]))
    if report.examples:
        story.append(_example_table(report.examples, center_style))
    else:
        story.append(Paragraph("No readable example images were available.", styles["BodyText"]))

    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return path


def _summary_table(report: ReportData) -> Table:
    table = Table(
        [
            ["Total", "Pass", "Fail", "Defect rate"],
            [
                str(report.total_count),
                str(report.pass_count),
                str(report.fail_count),
                f"{report.defect_rate:.1%}",
            ],
        ],
        colWidths=[1.55 * inch] * 4,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334E68")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F0F4F8")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
            ]
        )
    )
    return table


def _defect_chart(report: ReportData) -> Drawing:
    drawing = Drawing(460, 190)
    drawing.add(
        String(
            230,
            176,
            f"Inspection outcomes - defect rate {report.defect_rate:.1%}",
            textAnchor="middle",
            fontName="Helvetica-Bold",
            fontSize=12,
            fillColor=colors.HexColor("#243B53"),
        )
    )
    chart = Pie()
    chart.x = 70
    chart.y = 22
    chart.width = 135
    chart.height = 135
    chart.data = [report.pass_count, report.fail_count]
    chart.slices[0].fillColor = colors.HexColor("#2F855A")
    chart.slices[1].fillColor = colors.HexColor("#C53030")
    chart.slices.strokeColor = colors.white
    drawing.add(chart)

    legend = Legend()
    legend.x = 270
    legend.y = 115
    legend.fontName = "Helvetica"
    legend.fontSize = 10
    legend.dx = 12
    legend.dy = 12
    legend.deltay = 24
    legend.colorNamePairs = [
        (colors.HexColor("#2F855A"), f"Pass: {report.pass_count}"),
        (colors.HexColor("#C53030"), f"Fail: {report.fail_count}"),
    ]
    drawing.add(legend)
    return drawing


def _example_table(
    records: tuple[InspectionRecord, ...], label_style: ParagraphStyle
) -> Table:
    cells = []
    for record in records:
        image_path = Path(record.image_path)
        reader = ImageReader(str(image_path))
        width, height = reader.getSize()
        scale = min(2.25 * inch / width, 2.05 * inch / height)
        image = Image(str(image_path), width=width * scale, height=height * scale)
        label = Paragraph(
            f"<b>{record.result.upper()}</b> - {record.confidence:.1%}<br/>"
            f"{html.escape(image_path.name)}",
            label_style,
        )
        cells.append([image, Spacer(1, 6), label])

    rows = []
    for index in range(0, len(cells), 2):
        row = cells[index : index + 2]
        if len(row) == 1:
            row.append("")
        rows.append(row)
    table = Table(rows, colWidths=[3.25 * inch, 3.25 * inch])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E2EC")),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _draw_footer(canvas: Canvas, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.line(42, 27, A4[0] - 42, 27)
    canvas.setFillColor(colors.HexColor("#627D98"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(42, 16, "Industrial Vision Inspector")
    canvas.drawRightString(A4[0] - 42, 16, f"Page {document.page}")
    canvas.restoreState()


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _utc_timestamp(record: InspectionRecord) -> str:
    return (
        record.timestamp.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
