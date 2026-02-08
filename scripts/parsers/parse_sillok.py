"""조선왕조실록 XML 파서.

공공데이터포털에서 다운받은 실록 XML 파일들을 파싱하여
기사(article) 단위로 JSONL 파일에 저장한다.

Usage:
    python scripts/parsers/parse_sillok.py
    python scripts/parsers/parse_sillok.py --input data/raw/sillok --output data/parsed/sillok/articles.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

# 왕 코드 → 한글 이름 매핑 (파일명의 w{code}_ 부분)
KING_CODE_MAP: dict[str, str] = {
    "aa": "태조", "ba": "정종", "ca": "태종", "da": "세종",
    "ea": "문종", "fa": "단종", "ga": "세조", "ha": "예종",
    "ia": "성종", "ja": "연산군", "ka": "중종", "la": "인종",
    "ma": "명종", "na": "선조", "nb": "선조수정", "oa": "광해군(중초본)",
    "ob": "광해군(정초본)", "pa": "인조", "qa": "효종", "ra": "현종",
    "rb": "현종개수", "sa": "숙종", "sb": "숙종보궐정오", "ta": "경종",
    "tb": "경종수정", "ua": "영조", "va": "정조", "wa": "순조",
    "xa": "헌종", "ya": "철종", "za": "고종", "zb": "순종",
    "zc": "순종부록",
}


def extract_text(elem: ET.Element | None) -> str:
    """XML 요소에서 태그를 제거하고 텍스트만 추출한다."""
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def clean_original_text(text: str) -> str:
    """원문 텍스트를 정리한다.

    - 앞뒤 공백 제거
    - 연속 공백을 하나로
    - ○ 뒤 날짜 표기(간지/숫자日간지) 정리
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_date_info(front: ET.Element | None) -> dict[str, str]:
    """front/biblioData에서 날짜 정보를 추출한다."""
    info: dict[str, str] = {}
    if front is None:
        return info
    for date_elem in front.iter("dateOccured"):
        dtype = date_elem.get("type", "")
        dtext = date_elem.text or date_elem.get("date", "")
        if dtype and dtext:
            info[dtype] = dtext.strip()
    return info


def get_subject_classes(front: ET.Element | None) -> list[str]:
    """front/biblioData에서 주제 분류를 추출한다."""
    if front is None:
        return []
    return [
        sc.text.strip()
        for sc in front.iter("subjectClass")
        if sc.text and sc.text.strip()
    ]


def parse_king_code(filename: str) -> str:
    """파일명에서 왕 코드를 추출한다. 예: 2nd_waa_101.xml → aa"""
    m = re.match(r"2nd_w([a-z]{2})_", filename)
    return m.group(1) if m else ""


def parse_file_number(filename: str) -> int:
    """파일명에서 번호를 추출한다. 예: 2nd_waa_101.xml → 101"""
    m = re.match(r"2nd_w[a-z]{2}_(\d+)\.xml", filename)
    return int(m.group(1)) if m else -1


def get_sillok_name(root: ET.Element) -> str:
    """루트 요소에서 실록 이름을 추출한다."""
    for mt in root.iter("mainTitle"):
        text = (mt.text or "").strip()
        if text:
            return text
    return ""


def extract_articles_from_regular(
    root: ET.Element,
    king_code: str,
    sillok_name: str,
    filename: str,
) -> Iterator[dict]:
    """본문 파일(_1XX, _2XX+)에서 기사를 추출한다.

    구조: level2(root) > level3(월) > level4(일) > level5(기사)
    또는 부록의 경우: level2(root) > level3(기사) (level5 없이)
    """
    king_name = KING_CODE_MAP.get(king_code, king_code)

    # level5가 있는 경우 (일반 본문)
    for level5 in root.iter("level5"):
        article_id = level5.get("id", "")
        front = level5.find("front")
        biblio = front.find("biblioData") if front is not None else None

        # biblioData type="T"인 것만 기사
        if biblio is None or biblio.get("type") != "T":
            continue

        # 제목 (한국어 요약)
        title_elem = biblio.find("title")
        main_title = ""
        if title_elem is not None:
            mt = title_elem.find("mainTitle")
            main_title = extract_text(mt)

        # 원문
        content = level5.find(".//content")
        paragraphs = []
        if content is not None:
            for para in content.findall("paragraph"):
                text = extract_text(para)
                if text:
                    paragraphs.append(text)
            # postScript (사론 등)
            for ps in content.findall("postScript"):
                for para in ps.findall("paragraph"):
                    text = extract_text(para)
                    if text:
                        paragraphs.append(text)

        original = clean_original_text(" ".join(paragraphs))

        # 날짜
        date_info = get_date_info(front)

        # 주제 분류
        subjects = get_subject_classes(front)

        # 출처 (태백산사고본 정보)
        volume = ""
        page = ""
        for source in (biblio.findall("source") if biblio is not None else []):
            smt = source.find("mainTitle")
            if smt is not None and smt.get("type") == "태백산사고본":
                volume = extract_text(smt)
            pg = source.find("page")
            if pg is not None and not page:
                page = pg.get("begin", "")

        # 기사번호
        doc_no = ""
        dn = biblio.find("docNo") if biblio is not None else None
        if dn is not None and dn.text:
            doc_no = dn.text.strip()

        yield {
            "source": "sillok",
            "article_id": article_id,
            "king_code": king_code,
            "king": king_name,
            "subset": sillok_name,
            "volume": volume,
            "page": page,
            "doc_no": doc_no,
            "date": date_info.get("재위연도", ""),
            "date_western": date_info.get("서기", ""),
            "date_ganji": date_info.get("간지", ""),
            "title": main_title,
            "original": original,
            "subjects": subjects,
            "url": f"https://sillok.history.go.kr/id/{article_id}",
        }

    # level5가 없는 부록 파일: level3에서 직접 추출
    has_level5 = any(True for _ in root.iter("level5"))
    if not has_level5:
        for level3 in root.iter("level3"):
            article_id = level3.get("id", "")
            front = level3.find("front")
            biblio = front.find("biblioData") if front is not None else None

            if biblio is None or biblio.get("type") != "T":
                continue

            title_elem = biblio.find("title")
            main_title = ""
            if title_elem is not None:
                mt = title_elem.find("mainTitle")
                main_title = extract_text(mt)

            content = level3.find(".//content")
            paragraphs = []
            if content is not None:
                for para in content.findall("paragraph"):
                    text = extract_text(para)
                    if text:
                        paragraphs.append(text)

            original = clean_original_text(" ".join(paragraphs))
            date_info = get_date_info(front)
            subjects = get_subject_classes(front)

            yield {
                "source": "sillok",
                "article_id": article_id,
                "king_code": king_code,
                "king": king_name,
                "subset": sillok_name,
                "volume": "",
                "page": "",
                "doc_no": "",
                "date": date_info.get("재위연도", ""),
                "date_western": date_info.get("서기", ""),
                "date_ganji": date_info.get("간지", ""),
                "title": main_title,
                "original": original,
                "subjects": subjects,
                "url": f"https://sillok.history.go.kr/id/{article_id}",
            }


def extract_articles_from_chongseo(
    root: ET.Element,
    king_code: str,
    sillok_name: str,
) -> Iterator[dict]:
    """총서 파일(_000.xml)에서 기사를 추출한다.

    구조: level1(root) > level2(총서) > level3(기사)
    """
    king_name = KING_CODE_MAP.get(king_code, king_code)

    for level3 in root.iter("level3"):
        article_id = level3.get("id", "")
        front = level3.find("front")
        biblio = front.find("biblioData") if front is not None else None

        if biblio is None or biblio.get("type") != "T":
            continue

        title_elem = biblio.find("title")
        main_title = ""
        if title_elem is not None:
            mt = title_elem.find("mainTitle")
            main_title = extract_text(mt)

        content = level3.find(".//content")
        paragraphs = []
        if content is not None:
            for para in content.findall("paragraph"):
                text = extract_text(para)
                if text:
                    paragraphs.append(text)

        original = clean_original_text(" ".join(paragraphs))
        date_info = get_date_info(front)
        subjects = get_subject_classes(front)

        yield {
            "source": "sillok",
            "article_id": article_id,
            "king_code": king_code,
            "king": king_name,
            "subset": sillok_name,
            "volume": "",
            "page": "",
            "doc_no": "",
            "date": date_info.get("재위연도", ""),
            "date_western": date_info.get("서기", ""),
            "date_ganji": date_info.get("간지", ""),
            "title": main_title,
            "original": original,
            "subjects": subjects,
            "url": f"https://sillok.history.go.kr/id/{article_id}",
        }


def parse_xml_file(xml_path: Path) -> Iterator[dict]:
    """XML 파일 하나를 파싱하여 기사들을 yield한다."""
    filename = xml_path.name
    king_code = parse_king_code(filename)
    file_num = parse_file_number(filename)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    sillok_name = get_sillok_name(root)

    if file_num == 0:
        # _000.xml: 총서 (level1 > level2 > level3)
        yield from extract_articles_from_chongseo(root, king_code, sillok_name)
    else:
        # _1XX, _2XX+: 본문/부록 (level2 > ... > level5 또는 level3)
        yield from extract_articles_from_regular(
            root, king_code, sillok_name, filename
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="조선왕조실록 XML 파서")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/sillok"),
        help="XML 파일이 있는 디렉토리 (default: data/raw/sillok)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/parsed/sillok/articles.jsonl"),
        help="출력 JSONL 파일 경로 (default: data/parsed/sillok/articles.jsonl)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(input_dir.glob("2nd_w*.xml"))
    if not xml_files:
        print(f"[ERROR] XML 파일을 찾을 수 없습니다: {input_dir}")
        return

    print(f"[INFO] XML 파일 {len(xml_files)}개 발견")

    total_articles = 0
    king_counts: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as f:
        for i, xml_path in enumerate(xml_files, 1):
            try:
                count = 0
                for article in parse_xml_file(xml_path):
                    f.write(json.dumps(article, ensure_ascii=False) + "\n")
                    count += 1

                king_code = parse_king_code(xml_path.name)
                king_name = KING_CODE_MAP.get(king_code, king_code)
                king_counts[king_name] = king_counts.get(king_name, 0) + count
                total_articles += count

                if i % 50 == 0 or i == len(xml_files):
                    print(
                        f"  [{i}/{len(xml_files)}] {xml_path.name} → "
                        f"{count}건 (누적 {total_articles:,}건)"
                    )
            except ET.ParseError as e:
                print(f"  [WARN] XML 파싱 실패: {xml_path.name} - {e}")
            except Exception as e:
                print(f"  [ERROR] {xml_path.name} - {e}")

    print(f"\n[DONE] 총 {total_articles:,}건 → {output_path}")
    print("\n왕대별 기사 수:")
    for king, count in king_counts.items():
        print(f"  {king}: {count:,}건")


if __name__ == "__main__":
    main()
