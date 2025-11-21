import csv
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# -----------------------------
# READ CSV & ANALYZE DATA
# -----------------------------
def read_and_analyze_csv(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    # assumes there is a numeric column named "value"
    values = [float(row["value"]) for row in rows]

    summary = {
        "count": len(values),
        "total": sum(values),
        "average": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }

    # build table data for PDF (header + rows)
    headers = list(rows[0].keys())
    table_data = [headers]
    for row in rows:
        table_data.append([row[h] for h in headers])

    return summary, table_data


# -----------------------------
# GENERATE PDF REPORT
# -----------------------------
def generate_pdf_report(output_pdf_path, csv_filename, summary, table_data):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_pdf_path)

    story = []

    # Title
    story.append(Paragraph("Automated Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Meta info
    story.append(Paragraph(f"Source file: {csv_filename}", styles["Normal"]))
    story.append(
        Paragraph(
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    # Summary section
    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(f"Records: {summary['count']}", styles["Normal"]))
    story.append(Paragraph(f"Total value: {summary['total']:.2f}", styles["Normal"]))
    story.append(Paragraph(f"Average value: {summary['average']:.2f}", styles["Normal"]))
    story.append(Paragraph(f"Minimum value: {summary['min']:.2f}", styles["Normal"]))
    story.append(Paragraph(f"Maximum value: {summary['max']:.2f}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Table with raw data
    story.append(Paragraph("Data", styles["Heading2"]))

    table = Table(table_data)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.append(table)

    # Build PDF
    doc.build(story)
    print(f"PDF report generated: {output_pdf_path}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    csv_file = "data.csv"
    pdf_file = "report.pdf"

    try:
        summary, table_data = read_and_analyze_csv(csv_file)
        generate_pdf_report(pdf_file, csv_file, summary, table_data)
    except FileNotFoundError:
        print(f"ERROR: Could not find {csv_file}. Make sure it is in the same folder.")
    except KeyError:
        print("ERROR: CSV must have a 'value' column.")
    except ValueError as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
