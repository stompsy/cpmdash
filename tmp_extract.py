from pathlib import Path
from zipfile import ZipFile

from defusedxml.ElementTree import parse as ET_parse


def extract_docx_text(path: Path) -> str:
    with ZipFile(path) as zip_file, zip_file.open("word/document.xml") as doc_xml:
        tree = ET_parse(doc_xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for node in tree.iterfind(".//w:p", namespace):
        parts = [item.text for item in node.findall(".//w:t", namespace) if item.text]
        if parts:
            paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def main() -> None:
    base = Path("src/static/documents")
    selection = {
        "Combined_CPM_Master_Document.docx",
        "Annual Report 2024 Draft 1.docx",
        "Information Sharing Strategy.docx",
        "PORT Powerpoint.docx",
    }

    targets = []
    for path in sorted(base.glob("*.docx")):
        if path.name in selection or path.name.startswith("2023"):
            targets.append(path)

    for path in targets:
        print(f"--- {path.name} ---")
        text = extract_docx_text(path)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:120]:
            print(line)
        print()


if __name__ == "__main__":
    main()
