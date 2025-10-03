import csv
from pathlib import Path
from typing import Optional


class SubmissionValidator:
    """Валидатор для проверки submission файла"""

    def __init__(self, submission_file_path: Optional[str] = None) -> None:
        self.project_root = Path(__file__).parent.parent
        self.test_csv_path = self.project_root / "data" / "processed" / "test.csv"

        # Если путь не указан, используем data/processed/submission.csv
        if submission_file_path is None:
            self.submission_csv_path = self.project_root / "data" / "processed" / "submission.csv"
        else:
            self.submission_csv_path = Path(submission_file_path)

    def get_test_uids(self) -> set[str]:
        """Получить все uid из test.csv"""
        uids = set()
        with open(self.test_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # пропускаем заголовок
            for row in reader:
                if row and len(row) >= 1:
                    uids.add(row[0])
        return uids

    def validate_file_exists(self) -> bool:
        """Проверить, что файл submission существует"""
        if not self.submission_csv_path.exists():
            raise AssertionError(f"Файл {self.submission_csv_path} не найден")
        return True

    def validate_structure(self) -> bool:
        """Проверить структуру файла submission"""
        with open(self.submission_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)

            # Проверяем заголовок
            expected_header = ["uid", "type", "request"]
            if header != expected_header:
                raise AssertionError(f"Неверный заголовок. Ожидалось {expected_header}, получено {header}")
        return True

    def validate_uids_match_test(self) -> bool:
        """Проверить, что все uid из test.csv присутствуют в submission"""
        test_uids = self.get_test_uids()
        submission_uids = set()

        with open(self.submission_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # пропускаем заголовок

            for row in reader:
                if len(row) >= 1:
                    submission_uids.add(row[0])

        # Проверяем, что все uid из test.csv есть в submission
        missing_uids = test_uids - submission_uids
        if len(missing_uids) > 0:
            raise AssertionError(f"Отсутствуют uid в submission: {missing_uids}")

        # Проверяем, что нет лишних uid
        extra_uids = submission_uids - test_uids
        if len(extra_uids) > 0:
            raise AssertionError(f"Лишние uid в submission: {extra_uids}")

        # Проверяем общее количество
        if len(submission_uids) != len(test_uids):
            raise AssertionError(f"Количество uid не совпадает: {len(submission_uids)} vs {len(test_uids)}")

        return True

    def validate_row_count(self) -> bool:
        """Проверить, что количество строк в submission совпадает с test.csv"""

        def count_lines(filepath: Path) -> int:
            with open(filepath, encoding="utf-8") as f:
                return sum(1 for line in f)

        test_lines = count_lines(self.test_csv_path)
        submission_lines = count_lines(self.submission_csv_path)

        if submission_lines != test_lines:
            raise AssertionError(f"Количество строк не совпадает: submission {submission_lines}, test {test_lines}")

        return True

    def validate_type_values(self) -> bool:
        """Проверить, что все значения type являются валидными HTTP методами"""
        valid_http_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}

        invalid_types = []

        with open(self.submission_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # пропускаем заголовок

            for row_num, row in enumerate(reader, start=2):
                if len(row) >= 2:
                    type_value = row[1]
                    if type_value not in valid_http_methods:
                        invalid_types.append((row_num, type_value))

        if len(invalid_types) > 0:
            raise AssertionError(f"Найдены невалидные HTTP методы: {invalid_types}")

        return True

    def validate_request_values(self) -> bool:
        """Проверить, что все значения request являются валидными API путями"""
        invalid_requests = []

        with open(self.submission_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # пропускаем заголовок

            for row_num, row in enumerate(reader, start=2):
                if len(row) >= 3:
                    request_value = row[2]
                    # API путь должен начинаться с /
                    if not request_value.startswith("/"):
                        invalid_requests.append((row_num, request_value))

        if len(invalid_requests) > 0:
            raise AssertionError(f"Найдены невалидные API пути (не начинаются с /): {invalid_requests}")

        return True

    def validate_no_empty_values(self) -> bool:
        """Проверить, что нет пустых значений в обязательных полях"""
        empty_values = []

        with open(self.submission_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # пропускаем заголовок

            for row_num, row in enumerate(reader, start=2):
                for col_num, value in enumerate(row, start=1):
                    if not value.strip():  # пустое или только пробелы
                        empty_values.append((row_num, col_num, value))

        if len(empty_values) > 0:
            raise AssertionError(f"Найдены пустые значения: {empty_values}")

        return True

    def validate_uid_uniqueness(self) -> bool:
        """Проверить, что все uid уникальны"""
        uids = []
        duplicates = []

        with open(self.submission_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # пропускаем заголовок

            for row_num, row in enumerate(reader, start=2):
                if len(row) >= 1:
                    uid = row[0]
                    if uid in uids:
                        duplicates.append((row_num, uid))
                    uids.append(uid)

        if len(duplicates) > 0:
            raise AssertionError(f"Найдены дубликаты uid: {duplicates}")

        return True

    def run_all_validations(self) -> list[tuple[str, bool, Optional[str]]]:
        """Запустить все проверки валидации"""
        validations = [
            ("file_exists", self.validate_file_exists),
            ("structure", self.validate_structure),
            ("uids_match", self.validate_uids_match_test),
            ("row_count", self.validate_row_count),
            ("type_values", self.validate_type_values),
            ("request_values", self.validate_request_values),
            ("no_empty_values", self.validate_no_empty_values),
            ("uid_uniqueness", self.validate_uid_uniqueness),
        ]

        results = []
        for name, validation_func in validations:
            try:
                validation_func()
                results.append((name, True, None))
            except Exception as e:
                results.append((name, False, str(e)))

        return results
