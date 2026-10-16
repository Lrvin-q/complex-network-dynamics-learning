#!/usr/bin/env python3
"""Create compact contact sheets from rendered PDF pages for visual QA."""

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
QA = ROOT / "paper" / "qa"
PAGES_PER_SHEET = 8
COLS = 4
THUMB_WIDTH = 260
LABEL_HEIGHT = 24
MARGIN = 12


def page_number(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[1])


def main() -> None:
    pages = sorted(QA.glob("page-*.png"), key=page_number)
    if not pages:
        raise SystemExit("No rendered pages found")

    for old in QA.glob("contact-*.png"):
        old.unlink()

    for start in range(0, len(pages), PAGES_PER_SHEET):
        group = pages[start : start + PAGES_PER_SHEET]
        prepared = []
        for path in group:
            with Image.open(path) as image:
                image = image.convert("RGB")
                height = round(image.height * THUMB_WIDTH / image.width)
                prepared.append((path, image.resize((THUMB_WIDTH, height), Image.Resampling.LANCZOS)))

        rows = (len(prepared) + COLS - 1) // COLS
        thumb_height = max(image.height for _, image in prepared)
        sheet = Image.new(
            "RGB",
            (
                MARGIN + COLS * (THUMB_WIDTH + MARGIN),
                MARGIN + rows * (LABEL_HEIGHT + thumb_height + MARGIN),
            ),
            "#d7d7d7",
        )
        draw = ImageDraw.Draw(sheet)
        for index, (path, image) in enumerate(prepared):
            row, col = divmod(index, COLS)
            x = MARGIN + col * (THUMB_WIDTH + MARGIN)
            y = MARGIN + row * (LABEL_HEIGHT + thumb_height + MARGIN)
            draw.text((x, y + 3), f"Page {page_number(path)}", fill="black")
            sheet.paste(image, (x, y + LABEL_HEIGHT))

        sheet_index = start // PAGES_PER_SHEET + 1
        sheet.save(QA / f"contact-{sheet_index:02d}.png", optimize=True)

    print(f"Created {(len(pages) + PAGES_PER_SHEET - 1) // PAGES_PER_SHEET} contact sheets")


if __name__ == "__main__":
    main()
