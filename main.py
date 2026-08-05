"""Mini NPU Simulator.

외부 라이브러리 없이 2차원 배열의 MAC(Multiply-Accumulate) 연산을
직접 수행하고, 사용자 입력 또는 data.json 데이터를 판정한다.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


Matrix = List[List[float]]
EPSILON = 1e-9
BENCHMARK_REPEATS = 10
DATA_PATH = Path(__file__).with_name("data.json")


def normalize_label(label: object) -> str:
    """외부 라벨을 프로그램의 표준 라벨(Cross 또는 X)로 변환한다."""
    if not isinstance(label, str):
        raise ValueError("라벨은 문자열이어야 합니다.")

    normalized = label.strip().lower()
    if normalized in {"+", "cross"}:
        return "Cross"
    if normalized == "x":
        return "X"
    raise ValueError(f"알 수 없는 라벨: {label!r}")


def validate_matrix(matrix: object, expected_size: int, name: str) -> Matrix:
    """matrix가 expected_size x expected_size 숫자 배열인지 검사한다."""
    if not isinstance(matrix, list) or len(matrix) != expected_size:
        raise ValueError(
            f"{name}: 행 수가 {expected_size}개여야 합니다."
        )

    validated: Matrix = []
    for row_index, row in enumerate(matrix, start=1):
        if not isinstance(row, list) or len(row) != expected_size:
            raise ValueError(
                f"{name}: {row_index}번째 행의 열 수가 "
                f"{expected_size}개여야 합니다."
            )

        converted_row: List[float] = []
        for column_index, value in enumerate(row, start=1):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name}: ({row_index}, {column_index}) 값은 숫자여야 합니다."
                )
            converted_row.append(float(value))
        validated.append(converted_row)

    return validated


def mac(pattern: Sequence[Sequence[float]], filter_matrix: Sequence[Sequence[float]]) -> float:
    """같은 위치의 값을 곱해 모두 더한 MAC 점수를 반환한다."""
    if len(pattern) != len(filter_matrix):
        raise ValueError("패턴과 필터의 행 수가 다릅니다.")

    score = 0.0
    for row_index in range(len(pattern)):
        pattern_row = pattern[row_index]
        filter_row = filter_matrix[row_index]
        if len(pattern_row) != len(filter_row):
            raise ValueError("패턴과 필터의 열 수가 다릅니다.")
        for column_index in range(len(pattern_row)):
            score += pattern_row[column_index] * filter_row[column_index]
    return score


def decide(score_a: float, score_b: float, label_a: str, label_b: str) -> str:
    """epsilon보다 점수 차이가 작으면 UNDECIDED, 아니면 높은 점수 라벨을 반환한다."""
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def benchmark_mac(
    pattern: Sequence[Sequence[float]],
    filters: Sequence[Sequence[Sequence[float]]],
    repeats: int = BENCHMARK_REPEATS,
) -> float:
    """I/O를 제외하고 MAC 함수 한 번당 평균 실행 시간을 ms로 반환한다."""
    if repeats < 1 or not filters:
        raise ValueError("반복 횟수와 필터 수는 1 이상이어야 합니다.")

    start_ns = time.perf_counter_ns()
    for _ in range(repeats):
        for filter_matrix in filters:
            mac(pattern, filter_matrix)
    elapsed_ns = time.perf_counter_ns() - start_ns
    call_count = repeats * len(filters)
    return elapsed_ns / call_count / 1_000_000


def generate_pattern(size: int, label: str) -> Matrix:
    """홀수 크기의 Cross 또는 X 패턴을 만든다."""
    if size < 1 or size % 2 == 0:
        raise ValueError("패턴 크기는 1 이상의 홀수여야 합니다.")

    standard_label = normalize_label(label)
    center = size // 2
    matrix: Matrix = []
    for row in range(size):
        matrix_row: List[float] = []
        for column in range(size):
            if standard_label == "Cross":
                active = row == center or column == center
            else:
                active = row == column or row + column == size - 1
            matrix_row.append(1.0 if active else 0.0)
        matrix.append(matrix_row)
    return matrix


def read_matrix(name: str, size: int = 3) -> Matrix:
    """공백으로 구분된 숫자 행을 입력받고 잘못된 행만 다시 입력받는다."""
    print(f"{name} ({size}줄 입력, 공백 구분)")
    matrix: Matrix = []

    while len(matrix) < size:
        row_number = len(matrix) + 1
        raw = input(f"  {row_number}행: ").strip()
        values = raw.split()

        if len(values) != size:
            print(
                f"입력 형식 오류: 각 줄에 {size}개의 숫자를 "
                "공백으로 구분해 입력하세요."
            )
            continue

        try:
            row = [float(value) for value in values]
        except ValueError:
            print("입력 형식 오류: 숫자만 입력하세요.")
            continue

        matrix.append(row)

    return matrix


def print_section(number: int, title: str) -> None:
    print("\n" + "-" * 48)
    print(f"[{number}] {title}")
    print("-" * 48)


def print_performance_table(rows: Sequence[Tuple[int, float]]) -> None:
    print(f"{'크기':<12}{'평균 시간(ms)':>16}{'연산 횟수(N²)':>18}")
    print("-" * 46)
    for size, average_ms in rows:
        print(f"{size}×{size:<9}{average_ms:>16.6f}{size * size:>18}")


def run_manual_mode() -> None:
    print_section(1, "필터 입력")
    filter_a = read_matrix("필터 A")
    filter_b = read_matrix("필터 B")
    print("✓ 필터 A와 B 저장 완료")

    print_section(2, "패턴 입력")
    pattern = read_matrix("패턴")
    print("✓ 패턴 저장 완료")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    result = decide(score_a, score_b, "A", "B")
    average_ms = benchmark_mac(pattern, [filter_a, filter_b])

    print_section(3, "MAC 결과")
    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(평균/{BENCHMARK_REPEATS}회): {average_ms:.6f} ms")
    if result == "UNDECIDED":
        print(f"판정: 판정 불가 (|A-B| < {EPSILON:g})")
    else:
        print(f"판정: {result}")

    print_section(4, f"성능 분석 (평균/{BENCHMARK_REPEATS}회)")
    print_performance_table([(3, average_ms)])


def load_json_data(path: Path = DATA_PATH) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"data.json을 찾을 수 없습니다: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"data.json 형식 오류: {error.msg} (줄 {error.lineno})"
        ) from error

    if not isinstance(data, dict):
        raise ValueError("data.json의 최상위 값은 객체여야 합니다.")
    return data


def load_filters(raw_filters: object) -> Tuple[Dict[int, Dict[str, Matrix]], List[str]]:
    """필터 스키마를 검사하고 크기별 표준 라벨 필터를 반환한다."""
    loaded: Dict[int, Dict[str, Matrix]] = {}
    errors: List[str] = []

    if not isinstance(raw_filters, dict):
        return loaded, ["filters는 객체여야 합니다."]

    for size_key, filter_group in raw_filters.items():
        match = re.fullmatch(r"size_(\d+)", str(size_key))
        if match is None:
            errors.append(f"{size_key}: 필터 키는 size_N 형식이어야 합니다.")
            continue

        size = int(match.group(1))
        if not isinstance(filter_group, dict):
            errors.append(f"{size_key}: Cross/X 필터 객체가 필요합니다.")
            continue

        normalized_group: Dict[str, Matrix] = {}
        try:
            for raw_label, matrix in filter_group.items():
                label = normalize_label(raw_label)
                if label in normalized_group:
                    raise ValueError(f"{label} 필터가 중복되었습니다.")
                normalized_group[label] = validate_matrix(
                    matrix, size, f"{size_key}.{raw_label}"
                )

            missing = {"Cross", "X"} - normalized_group.keys()
            if missing:
                raise ValueError(f"필터 누락: {', '.join(sorted(missing))}")
        except ValueError as error:
            errors.append(f"{size_key}: {error}")
            continue

        loaded[size] = normalized_group

    return loaded, errors


def analyze_patterns(
    raw_patterns: object, filters: Mapping[int, Mapping[str, Matrix]]
) -> Tuple[int, int, List[Tuple[str, str]], Dict[int, Matrix]]:
    """각 JSON 패턴을 독립적으로 분석하여 한 케이스의 오류가 전체를 중단하지 않게 한다."""
    total = 0
    passed = 0
    failures: List[Tuple[str, str]] = []
    benchmark_patterns: Dict[int, Matrix] = {}

    if not isinstance(raw_patterns, dict):
        reason = "patterns는 객체여야 합니다."
        print(f"FAIL: {reason}")
        return 1, 0, [("patterns", reason)], benchmark_patterns

    for case_id, case in raw_patterns.items():
        total += 1
        print(f"\n--- {case_id} ---")

        try:
            match = re.fullmatch(r"size_(\d+)_(\d+)", str(case_id))
            if match is None:
                raise ValueError("패턴 키는 size_N_idx 형식이어야 합니다.")

            size = int(match.group(1))
            if size not in filters:
                raise ValueError(f"size_{size}에 사용할 유효한 필터가 없습니다.")
            if not isinstance(case, dict):
                raise ValueError("패턴 항목은 input과 expected를 가진 객체여야 합니다.")
            if "input" not in case or "expected" not in case:
                raise ValueError("input 또는 expected 키가 없습니다.")

            pattern = validate_matrix(case["input"], size, str(case_id))
            expected = normalize_label(case["expected"])
            cross_score = mac(pattern, filters[size]["Cross"])
            x_score = mac(pattern, filters[size]["X"])
            prediction = decide(cross_score, x_score, "Cross", "X")
            benchmark_patterns.setdefault(size, pattern)

            print(f"Cross 점수: {cross_score}")
            print(f"X 점수: {x_score}")
            if prediction == expected:
                passed += 1
                print(f"판정: {prediction} | expected: {expected} | PASS")
            else:
                if prediction == "UNDECIDED":
                    reason = "점수 차이가 epsilon보다 작아 UNDECIDED로 판정됨"
                else:
                    reason = f"판정 {prediction}이 expected {expected}와 다름"
                failures.append((str(case_id), reason))
                print(
                    f"판정: {prediction} | expected: {expected} | "
                    f"FAIL ({reason})"
                )
        except (KeyError, TypeError, ValueError) as error:
            reason = str(error)
            failures.append((str(case_id), reason))
            print(f"판정: FAIL ({reason})")

    return total, passed, failures, benchmark_patterns


def run_json_mode(path: Path = DATA_PATH) -> None:
    try:
        data = load_json_data(path)
    except ValueError as error:
        print(f"\n실행 오류: {error}")
        return

    print_section(1, "필터 로드")
    filters, filter_errors = load_filters(data.get("filters"))
    for size in sorted(filters):
        print(f"✓ size_{size} 필터 로드 완료 (Cross, X)")
    for error in filter_errors:
        print(f"✗ {error}")

    print_section(2, "패턴 분석 (라벨 정규화 적용)")
    total, passed, failures, benchmark_patterns = analyze_patterns(
        data.get("patterns"), filters
    )

    print_section(3, f"성능 분석 (평균/{BENCHMARK_REPEATS}회)")
    performance_rows: List[Tuple[int, float]] = []

    cross_3 = generate_pattern(3, "Cross")
    x_3 = generate_pattern(3, "X")
    performance_rows.append((3, benchmark_mac(cross_3, [cross_3, x_3])))

    for size in sorted(filters):
        pattern = benchmark_patterns.get(size, filters[size]["Cross"])
        average_ms = benchmark_mac(
            pattern, [filters[size]["Cross"], filters[size]["X"]]
        )
        performance_rows.append((size, average_ms))

    print_performance_table(performance_rows)

    print_section(4, "결과 요약")
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {total - passed}개")

    if failures:
        print("\n실패 케이스:")
        for case_id, reason in failures:
            print(f"- {case_id}: {reason}")
    else:
        print("실패 케이스가 없습니다.")


def choose_mode() -> str:
    print("=== Mini NPU Simulator ===")
    print("\n[모드 선택]\n")
    print("1. 사용자 입력 (3×3)")
    print("2. data.json 분석")

    while True:
        choice = input("선택: ").strip()
        if choice in {"1", "2"}:
            return choice
        print("입력 오류: 1 또는 2를 입력하세요.")


def main() -> None:
    mode = choose_mode()
    if mode == "1":
        run_manual_mode()
    else:
        run_json_mode()


if __name__ == "__main__":
    main()
