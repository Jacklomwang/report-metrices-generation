from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Any

PHENOTYPE_BASE_PATH = Path('/export02/projects/LCS/05_phenotype/redcap_exports')
PHENOTYPE_REDCAP_PATH = PHENOTYPE_BASE_PATH / 'InvestigationOfTheLo_DATA.csv'
PHENOTYPE_REDCAP_DEFINITIONS_PATH = PHENOTYPE_BASE_PATH / 'REDCap_variables_definitions.xlsx'
PHENOTYPE_GROUP_INFO_CC_PATH = PHENOTYPE_BASE_PATH / 'Group_InfoSession_Data_CC.csv'
PHENOTYPE_GROUP_INFO_LC_PATH = PHENOTYPE_BASE_PATH / 'Group_InfoSession_Data_LC.csv'
PHENOTYPE_TESTING_SCHEDULE_PATH = PHENOTYPE_BASE_PATH / 'Testing_Schedule_Sheet1.csv'
PHENOTYPE_NOTES_SESSION_A_PATH = PHENOTYPE_BASE_PATH / 'LC_Experiments_Notes_v2_Session_A_Physio.csv'
PHENOTYPE_NOTES_SESSION_B_PATH = PHENOTYPE_BASE_PATH / 'LC_Experiments_Notes_v2_Session_B_MRI.csv'
PHENOTYPE_REDCAP_GLOB = 'InvestigationOfTheLo_DATA_*.csv'
PHENOTYPE_REDCAP_DEFINITIONS_GLOB = 'REDCap_variables_definitions_*.xlsx'
PHENOTYPE_GROUP_INFO_CC_GLOB = 'Group_InfoSession_Data_CC_*.csv'
PHENOTYPE_GROUP_INFO_LC_GLOB = 'Group_InfoSession_Data_LC_*.csv'
PHENOTYPE_TESTING_SCHEDULE_GLOB = 'Testing_Schedule_Sheet1_*.csv'
PHENOTYPE_NOTES_SESSION_A_GLOB = 'LC_Experiments_Notes_v2_Session_A_Physio_*.csv'
PHENOTYPE_NOTES_SESSION_B_GLOB = 'LC_Experiments_Notes_v2_Session_B_MRI_*.csv'
# No stable-name symlink exists for this export (unlike the others above), so we glob
# for the newest dated file each time instead of pointing at one fixed filename.
PHENOTYPE_INTAKE_GLOB = 'IntakeForm_DATA_*.csv'

CORE_NEUROPSYCH_FIELDS = [
    'DigitSpan_Forward',
    'DigitSpan_Backward',
    'RAVLT_DelayedRecall',
    'Category_Fluency_Category1_Total',
    'Category_Fluency_Category2_Total',
]

ECG_CONFIG_CUTOFF_DATE = '2025-10-31'
ECG_CONFIG_OLD_LABEL = 'Old'
ECG_CONFIG_NEW_LABEL = 'New'

MISSING_NOTE_TOKENS = {'', 'na', 'n/a', 'nan', 'none', 'null', '?', '-'}
SCHEDULE_CANCEL_KEYWORDS = ('cancel', 'no-show', 'withdrew', 'withdrawn', 'rescheduled')

XLSX_NS_MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
XLSX_NS_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG_REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
XLSX_NS = {'a': XLSX_NS_MAIN}
XLSX_CELL_REF_RE = re.compile(r'([A-Z]+)')


def normalize_participant_id(value: Any) -> str:
    if value is None:
        return ''
    token = str(value).strip().lower().replace(' ', '')
    if not token:
        return ''
    if token.startswith('sub-'):
        return token
    if token.startswith('sub'):
        return f"sub-{token[3:].lstrip('-_')}"
    return token


def infer_session_class(session_label: Any) -> str | None:
    if session_label is None:
        return None
    token = str(session_label).strip().lower().replace('_', '-')
    for prefix in ('session-', 'session', 'ses-'):
        if token.startswith(prefix):
            token = token[len(prefix):]
            break
    token = token.strip('- ')
    if token.startswith('a') or token in {'1', '01'}:
        return 'A'
    if token.startswith('b') or token in {'2', '02'}:
        return 'B'
    return None


def _read_rows_with_fallback(path: Path) -> list[list[str]]:
    for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            with path.open('r', encoding=encoding, newline='') as f:
                return list(csv.reader(f))
        except UnicodeDecodeError:
            continue
    with path.open('r', encoding='utf-8', errors='replace', newline='') as f:
        return list(csv.reader(f))


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in {'na', 'n/a', 'nan', 'null', '#value!', '#n/a', 'value!'}:
        return None
    if re.fullmatch(r'[-+]?\d+', text):
        try:
            return int(text)
        except Exception:
            return text
    if re.fullmatch(r'[-+]?\d*\.\d+', text):
        try:
            return float(text)
        except Exception:
            return text
    return text


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    text = str(value).strip()
    if not text or text.lower() in {'na', 'n/a', 'nan', 'null', '#value!', '#n/a'}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _sanitize_bmi_value(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None or parsed < 10 or parsed > 90:
        return None
    return round(parsed, 2)


def _sanitize_height_m(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None or parsed < 0.8 or parsed > 2.5:
        return None
    return parsed


def _sanitize_weight_kg(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None or parsed < 20 or parsed > 400:
        return None
    return parsed


def _normalized_var_token(value: Any) -> str:
    return re.sub(r'[^a-z0-9]', '', str(value or '').strip().lower())


def _resolve_bmi_from_group_entry(entry: dict[str, Any] | None) -> float | None:
    if not entry:
        return None
    for key, value in entry.items():
        if _normalized_var_token(key) == 'bmi':
            bmi = _sanitize_bmi_value(value)
            if bmi is not None:
                return bmi
    height_m = None
    weight_kg = None
    for key, value in entry.items():
        token = _normalized_var_token(key)
        if token in {'heightm', 'height'}:
            height_m = _safe_float(value)
        elif token in {'weightkg', 'weight'}:
            weight_kg = _safe_float(value)
    if height_m and height_m > 0 and weight_kg and weight_kg > 0:
        return _sanitize_bmi_value(weight_kg / (height_m * height_m))
    return None


def _clean_note(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_NOTE_TOKENS:
        return None
    return text or None


def _parse_iso_date(text: str) -> date | None:
    try:
        return datetime.strptime(text, '%Y-%m-%d').date()
    except Exception:
        return None


def _parse_datetime_from_string(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d %I:%M %p', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M'):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _parse_date_value(value: Any, default_year: int = 2025) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    month_fixes = {
        'fev': 'feb', 'fév': 'feb', 'aout': 'aug', 'août': 'aug', 'déc': 'dec',
        'avr': 'apr', 'mai': 'may', 'juin': 'jun', 'juillet': 'jul', 'sept': 'sep',
        'octobre': 'oct', 'novembre': 'novembre'.replace('embre','')
    }
    normalized = text.lower()
    for src, dst in month_fixes.items():
        normalized = normalized.replace(src, dst)
    normalized = normalized.replace('  ', ' ').strip()
    formats = [
        ('%Y-%m-%d', False), ('%Y/%m/%d', False), ('%d-%b-%Y', False), ('%d-%b-%y', False),
        ('%d-%b', True), ('%d/%m/%Y', False), ('%m/%d/%Y', False), ('%d/%m/%y', False),
        ('%m/%d/%y', False), ('%d-%m-%Y', False), ('%d-%m-%y', False),
    ]
    for fmt, needs_year in formats:
        try:
            dt = datetime.strptime(normalized, fmt)
            return date(default_year, dt.month, dt.day) if needs_year else dt.date()
        except Exception:
            continue
    return None


def _parse_time_value(value: Any) -> time | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?\s*[APMapm]{0,2})$', text)
    if match:
        text = match.group(1).strip()
    for fmt in ('%I:%M %p', '%I:%M%p', '%H:%M', '%H:%M:%S', '%I%p'):
        try:
            return datetime.strptime(text, fmt).time()
        except Exception:
            continue
    return None


def _compose_datetime(date_raw: Any, time_raw: Any, default_year: int = 2025) -> datetime | None:
    parsed_date = _parse_date_value(date_raw, default_year=default_year)
    if parsed_date is None:
        return None
    parsed_time = _parse_time_value(time_raw) or time(0, 0)
    return datetime.combine(parsed_date, parsed_time)


def _split_researchers(raw_values: list[str]) -> list[str]:
    canonical = {
        'jack': 'Jack', 'sophia': 'Sophia', 'oren': 'Oren', 'amanda': 'Amanda', 'andrew': 'Andrew',
        'michelle': 'Michelle', 'mary': 'Mary', 'nesrine': 'Nesrine', 'dina': 'Dina',
    }
    out: list[str] = []
    for raw in raw_values:
        text = _clean_note(raw)
        if not text:
            continue
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\band\b', '|', text, flags=re.IGNORECASE)
        for sep in ('/', '+', ',', ';'):
            text = text.replace(sep, '|')
        for token in (x.strip() for x in text.split('|')):
            if not token:
                continue
            normalized = canonical.get(token.lower(), token.title())
            if normalized not in out:
                out.append(normalized)
    return out[:2]


def _load_schedule_entries(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows = _read_rows_with_fallback(path)
    if not rows:
        return {}
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}

    def _value(row: list[str], key: str) -> str:
        pos = idx.get(key)
        if pos is None or pos >= len(row):
            return ''
        return str(row[pos]).strip()

    by_participant: dict[str, list[dict[str, Any]]] = {}
    for row in rows[1:]:
        participant = normalize_participant_id(_value(row, 'ID ') or _value(row, 'ID'))
        if not participant:
            continue
        session_raw = _value(row, 'Session')
        dt = _compose_datetime(_value(row, 'Date'), _value(row, 'Time'))
        entry = {
            'participant': participant,
            'session_raw': session_raw,
            'session_class': infer_session_class(session_raw),
            'datetime': dt,
            'date': dt.date() if dt else _parse_date_value(_value(row, 'Date')),
            'notes': _clean_note(_value(row, 'Notes')),
            'researchers': _split_researchers([
                _value(row, 'Slot 1 (if ses-A : Doppler/Equipment Setup)'),
                _value(row, 'Slot 2 (if ses-A : Computerized Tasks/Spirometry)'),
                _value(row, 'Slot 3 (Shadowing)'),
            ]),
        }
        by_participant.setdefault(participant, []).append(entry)
    return by_participant


def _is_cancelled_without_staff(entry: dict[str, Any]) -> bool:
    note = (entry.get('notes') or '').lower()
    return any(k in note for k in SCHEDULE_CANCEL_KEYWORDS) and not entry.get('researchers')


def _select_schedule_entry(entries: list[dict[str, Any]], session_class: str | None) -> dict[str, Any] | None:
    if not entries:
        return None
    candidates = entries
    if session_class:
        filtered = [e for e in entries if e.get('session_class') == session_class]
        if filtered:
            candidates = filtered
    ordered = sorted(candidates, key=lambda e: e.get('datetime') or datetime.min)
    for entry in reversed(ordered):
        if not _is_cancelled_without_staff(entry):
            return entry
    return ordered[-1] if ordered else None


def _find_header_row(rows: list[list[str]]) -> int | None:
    for i, row in enumerate(rows[:20]):
        for cell in row:
            if str(cell).strip().lower() == 'id':
                return i
    return None


def _load_experiment_notes_entries(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows = _read_rows_with_fallback(path)
    if not rows:
        return {}
    header_idx = _find_header_row(rows)
    if header_idx is None:
        return {}
    header = [str(x).strip() for x in rows[header_idx]]
    id_idx = next((i for i, col in enumerate(header) if col.lower() == 'id'), None)
    if id_idx is None:
        return {}
    by_participant: dict[str, list[dict[str, Any]]] = {}
    for row in rows[header_idx + 1:]:
        if len(row) < len(header):
            row = row + [''] * (len(header) - len(row))
        participant = normalize_participant_id(row[id_idx])
        if not participant:
            continue
        fields = {col: (str(row[i]).strip() if i < len(row) else '') for i, col in enumerate(header) if col}
        dt = _compose_datetime(fields.get('Date', ''), fields.get('Time', ''))
        entry = {
            'participant': participant,
            'datetime': dt,
            'date': dt.date() if dt else _parse_date_value(fields.get('Date', '')),
            'fields': fields,
            'researchers': _split_researchers([fields.get('Researchers', '')]),
        }
        by_participant.setdefault(participant, []).append(entry)
    return by_participant


def _select_notes_entry(entries: list[dict[str, Any]], target_date: date | None = None) -> dict[str, Any] | None:
    if not entries:
        return None
    candidates = entries
    if target_date:
        matched = [entry for entry in entries if entry.get('date') == target_date]
        if matched:
            candidates = matched
    ordered = sorted(candidates, key=lambda entry: (entry.get('datetime') or datetime.min, len(entry.get('fields', {}))))
    return ordered[-1] if ordered else None


def _load_redcap_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_rows_with_fallback(path)
    if not rows:
        return {}
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}

    def _value(row: list[str], key: str) -> str:
        pos = idx.get(key)
        if pos is None or pos >= len(row):
            return ''
        return str(row[pos]).strip()

    def _first_value(row: list[str], keys: list[str]) -> Any:
        for key in keys:
            candidate = _coerce_scalar(_value(row, key))
            if candidate is not None:
                return candidate
        return None

    out: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        participant = normalize_participant_id(_value(row, 'redcap_survey_identifier'))
        if not participant:
            continue
        demographics_dt = _parse_datetime_from_string(_value(row, 'demographics_timestamp'))
        consent_dt = _parse_datetime_from_string(_value(row, 'consent_form_timestamp'))
        out[participant] = {
            'age': _coerce_scalar(_value(row, 'age')),
            'bmi': _first_value(row, ['sb_bmi', 'bmi', 'body_mass_index', 'bmi_calc']),
            'height_m': _sanitize_height_m(_value(row, 'height_m')),
            'weight_kg': _sanitize_weight_kg(_value(row, 'weight_kg')),
            'sex_asab': _coerce_scalar(_value(row, 'asab')),
            'gender': _coerce_scalar(_value(row, 'gender')),
            'recording_datetime': demographics_dt or consent_dt,
        }
    return out


def _resolve_latest_intake_path(base_path: Path, pattern: str) -> Path | None:
    matches = sorted(base_path.glob(pattern))
    return matches[-1] if matches else None


def _resolve_source_path(preferred_path: Path, dated_pattern: str) -> Path:
    """Use a stable export name when present, otherwise the newest dated export."""
    if preferred_path.exists():
        return preferred_path
    matches = sorted(preferred_path.parent.glob(dated_pattern))
    return matches[-1] if matches else preferred_path


def _load_intake_form(base_path: Path, pattern: str) -> dict[str, dict[str, Any]]:
    path = _resolve_latest_intake_path(base_path, pattern)
    if path is None or not path.exists():
        return {}
    rows = _read_rows_with_fallback(path)
    if not rows:
        return {}
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}

    def _value(row: list[str], key: str) -> str:
        pos = idx.get(key)
        if pos is None or pos >= len(row):
            return ''
        return str(row[pos]).strip()

    out: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        participant = normalize_participant_id(_value(row, 'redcap_survey_identifier'))
        if not participant:
            continue
        out[participant] = {
            'age': _coerce_scalar(_value(row, 'agee')),
            'bmi': _sanitize_bmi_value(_value(row, 'bmi')),
            'height_m': _safe_float(_value(row, 'height_m')),
            'weight_kg': _safe_float(_value(row, 'weight_kg')),
        }
    return out


def _load_group_info(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_rows_with_fallback(path)
    if not rows:
        return {}
    header = rows[0]
    participant_cols: dict[int, str] = {}
    for idx, col_name in enumerate(header):
        if idx < 3:
            continue
        participant = normalize_participant_id(col_name)
        if participant:
            participant_cols[idx] = participant
    out: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        variable = str(row[1]).strip()
        if not variable:
            continue
        for idx, participant in participant_cols.items():
            if idx >= len(row):
                continue
            value = _coerce_scalar(row[idx])
            if value is not None:
                out.setdefault(participant, {})[variable] = value
    return out


def _xlsx_column_index_from_ref(cell_ref: str) -> int | None:
    match = XLSX_CELL_REF_RE.match(str(cell_ref or ''))
    if not match:
        return None
    letters = match.group(1)
    index = 0
    for char in letters:
        index = (index * 26) + (ord(char) - ord('A') + 1)
    return index - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value_node = cell.find(f'{{{XLSX_NS_MAIN}}}v')
    if value_node is None:
        inline_node = cell.find(f'{{{XLSX_NS_MAIN}}}is/{{{XLSX_NS_MAIN}}}t')
        return (inline_node.text or '').strip() if inline_node is not None else ''
    raw = (value_node.text or '').strip()
    if cell.attrib.get('t') == 's':
        try:
            return str(shared_strings[int(raw)]).strip()
        except Exception:
            return raw
    return raw


def _xlsx_row_to_cells(row: ET.Element, shared_strings: list[str]) -> dict[int, str]:
    values: dict[int, str] = {}
    fallback_index = 0
    for cell in row.findall(f'{{{XLSX_NS_MAIN}}}c'):
        col_index = _xlsx_column_index_from_ref(cell.attrib.get('r', ''))
        if col_index is None:
            col_index = fallback_index
        values[col_index] = _xlsx_cell_value(cell, shared_strings)
        fallback_index = max(fallback_index + 1, col_index + 1)
    return values


def _normalize_definition_field_name(field_name: str) -> str:
    return str(field_name or '').strip().lower()


def _normalize_choice_key(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value).strip().lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    if re.fullmatch(r'[-+]?\d+\.0+', text):
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text


def _field_matches_numeric_range(field_name: str, range_start: str, range_end: str) -> bool:
    field_match = re.fullmatch(r'([a-z_]+)(\d+)', field_name)
    start_match = re.fullmatch(r'([a-z_]+)(\d+)', range_start)
    end_match = re.fullmatch(r'([a-z_]+)(\d+)', range_end)
    if not (field_match and start_match and end_match):
        return False
    if not (field_match.group(1) == start_match.group(1) == end_match.group(1)):
        return False
    field_num = int(field_match.group(2))
    start_num = int(start_match.group(2))
    end_num = int(end_match.group(2))
    return min(start_num, end_num) <= field_num <= max(start_num, end_num)


def _lookup_redcap_definition_entry(field_name: str, definitions_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    key = _normalize_definition_field_name(field_name)
    if not key:
        return None
    if key in definitions_map:
        return definitions_map[key]
    for candidate_key, candidate_entry in definitions_map.items():
        if ' - ' not in candidate_key:
            continue
        left, right = [part.strip() for part in candidate_key.split(' - ', 1)]
        if _field_matches_numeric_range(key, left, right):
            return candidate_entry
    return None


def _load_redcap_definitions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with zipfile.ZipFile(path) as workbook_zip:
        workbook = ET.fromstring(workbook_zip.read('xl/workbook.xml'))
        rels = ET.fromstring(workbook_zip.read('xl/_rels/workbook.xml.rels'))
        rel_map = {rel.attrib['Id']: rel.attrib.get('Target', '') for rel in rels.findall(f'{{{PKG_REL_NS}}}Relationship')}
        shared_strings: list[str] = []
        if 'xl/sharedStrings.xml' in workbook_zip.namelist():
            shared_root = ET.fromstring(workbook_zip.read('xl/sharedStrings.xml'))
            for item in shared_root.findall('a:si', XLSX_NS):
                pieces = [text_node.text or '' for text_node in item.iter(f'{{{XLSX_NS_MAIN}}}t')]
                shared_strings.append(''.join(pieces))
        output: dict[str, dict[str, Any]] = {}
        for sheet in workbook.findall('a:sheets/a:sheet', XLSX_NS):
            sheet_name = sheet.attrib.get('name', '').strip()
            rid = sheet.attrib.get(f'{{{XLSX_NS_REL}}}id', '')
            target = rel_map.get(rid, '')
            if not target:
                continue
            sheet_path = target if target.startswith('xl/') else f'xl/{target}'
            if sheet_path not in workbook_zip.namelist():
                continue
            sheet_root = ET.fromstring(workbook_zip.read(sheet_path))
            rows = sheet_root.findall('.//a:sheetData/a:row', XLSX_NS)
            if not rows:
                continue
            header_cells = _xlsx_row_to_cells(rows[0], shared_strings)
            if not header_cells:
                continue
            max_col = max(header_cells.keys())
            headers = [str(header_cells.get(i, '')).strip() for i in range(max_col + 1)]
            var_idx = headers.index('Var') if 'Var' in headers else next((idx for idx, header in enumerate(headers) if 'Variable' in header), None)
            if var_idx is None:
                continue
            value_idx = headers.index('Value') if 'Value' in headers else None
            field_type_idx = headers.index('Field Type') if 'Field Type' in headers else None
            definition_idx = next((idx for idx, header in enumerate(headers) if 'Definition' in header), None)
            if definition_idx is None and len(headers) > 3:
                definition_idx = 3
            current_field = ''
            for row in rows[1:]:
                cells = _xlsx_row_to_cells(row, shared_strings)
                field_name = str(cells.get(var_idx, '')).strip()
                if field_name:
                    current_field = field_name
                if not current_field:
                    continue
                normalized_field = _normalize_definition_field_name(current_field)
                if not normalized_field:
                    continue
                entry = output.setdefault(normalized_field, {'source_sheet': sheet_name, 'field_type': '', 'choices': {}})
                field_type = str(cells.get(field_type_idx, '')).strip() if field_type_idx is not None else ''
                if field_type:
                    entry['field_type'] = field_type
                value = str(cells.get(value_idx, '')).strip() if value_idx is not None else ''
                definition = str(cells.get(definition_idx, '')).strip() if definition_idx is not None else ''
                if value and definition:
                    entry['choices'][_normalize_choice_key(value)] = definition
        return output


@lru_cache(maxsize=4)
def _load_redcap_definitions_cached(path_text: str) -> dict[str, dict[str, Any]]:
    return _load_redcap_definitions(Path(path_text))


def _interpret_redcap_value(field_name: str, value: Any, definitions_map: dict[str, dict[str, Any]]) -> str | None:
    if value is None:
        return None
    entry = _lookup_redcap_definition_entry(field_name, definitions_map)
    if not entry:
        return None
    return entry.get('choices', {}).get(_normalize_choice_key(value))


def _resolve_neuropsych_summary(participant: str, neuro_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = neuro_map.get(participant)
    if source is None:
        # Participant has no row at all in Group_Info_CC/LC, as opposed to having
        # a row where MoCA_Total itself is blank. Callers use `found` to tell those
        # two "missing" cases apart in the report.
        return {'found': False, 'NP_Date': None, 'MoCA_Total': None, 'MoCA_Subscores': {}, 'CoreTests': {}}
    moca_subscores = {k: v for k, v in source.items() if k.startswith('MoCA_') and k != 'MoCA_Total'}
    core_tests = {k: source[k] for k in CORE_NEUROPSYCH_FIELDS if k in source}
    return {
        'found': True,
        'NP_Date': source.get('NP_Date'),
        'MoCA_Total': source.get('MoCA_Total'),
        'MoCA_Subscores': moca_subscores,
        'CoreTests': core_tests,
    }


def _resolve_ecg_config(recording_date: date | None) -> str:
    if recording_date is None:
        return 'Unknown'
    cutoff = _parse_iso_date(ECG_CONFIG_CUTOFF_DATE)
    if cutoff is None:
        return 'Unknown'
    return ECG_CONFIG_NEW_LABEL if recording_date >= cutoff else ECG_CONFIG_OLD_LABEL


def _resolve_recording_date(schedule_entry: dict[str, Any] | None, notes_entry: dict[str, Any] | None, redcap_entry: dict[str, Any] | None) -> tuple[date | None, str]:
    if schedule_entry and schedule_entry.get('date'):
        return schedule_entry['date'], 'schedule'
    if notes_entry and notes_entry.get('date'):
        return notes_entry['date'], 'experiment_notes'
    if redcap_entry and redcap_entry.get('recording_datetime'):
        return redcap_entry['recording_datetime'].date(), 'redcap_demographics_timestamp'
    return None, 'unknown'


def build_subject_metadata(participant: str, session: str, task: str) -> dict[str, Any]:
    participant_id = normalize_participant_id(participant)
    session_class = infer_session_class(session)

    redcap_path = _resolve_source_path(PHENOTYPE_REDCAP_PATH, PHENOTYPE_REDCAP_GLOB)
    definitions_path = _resolve_source_path(
        PHENOTYPE_REDCAP_DEFINITIONS_PATH, PHENOTYPE_REDCAP_DEFINITIONS_GLOB
    )
    group_cc_path = _resolve_source_path(PHENOTYPE_GROUP_INFO_CC_PATH, PHENOTYPE_GROUP_INFO_CC_GLOB)
    group_lc_path = _resolve_source_path(PHENOTYPE_GROUP_INFO_LC_PATH, PHENOTYPE_GROUP_INFO_LC_GLOB)
    schedule_path = _resolve_source_path(
        PHENOTYPE_TESTING_SCHEDULE_PATH, PHENOTYPE_TESTING_SCHEDULE_GLOB
    )
    notes_a_path = _resolve_source_path(
        PHENOTYPE_NOTES_SESSION_A_PATH, PHENOTYPE_NOTES_SESSION_A_GLOB
    )
    notes_b_path = _resolve_source_path(
        PHENOTYPE_NOTES_SESSION_B_PATH, PHENOTYPE_NOTES_SESSION_B_GLOB
    )

    redcap_definitions = {}
    definitions_load_error = None
    try:
        redcap_definitions = _load_redcap_definitions_cached(str(definitions_path))
    except Exception as exc:
        definitions_load_error = str(exc)

    redcap_data = _load_redcap_metadata(redcap_path)
    group_cc = _load_group_info(group_cc_path)
    group_lc = _load_group_info(group_lc_path)
    intake_data = _load_intake_form(PHENOTYPE_BASE_PATH, PHENOTYPE_INTAKE_GLOB)
    schedule_data = _load_schedule_entries(schedule_path)
    notes_a_data = _load_experiment_notes_entries(notes_a_path)
    notes_b_data = _load_experiment_notes_entries(notes_b_path)

    neuro_map: dict[str, dict[str, Any]] = {}
    neuro_map.update(group_cc)
    neuro_map.update(group_lc)

    schedule_entry = _select_schedule_entry(schedule_data.get(participant_id, []), session_class=session_class)
    notes_map = notes_a_data if session_class == 'A' else notes_b_data
    notes_entry = _select_notes_entry(notes_map.get(participant_id, []), target_date=schedule_entry.get('date') if schedule_entry else None)

    redcap_entry = redcap_data.get(participant_id, {})
    group_cc_entry = group_cc.get(participant_id, {})
    group_lc_entry = group_lc.get(participant_id, {})
    intake_entry = intake_data.get(participant_id, {})

    # IntakeForm is the preferred source for age/BMI when the participant has a row
    # there; the group-info/REDCap chain below is only a fallback for participants
    # not yet in IntakeForm.
    bmi_value = intake_entry.get('bmi')
    bmi_source = 'intake_form' if bmi_value is not None else None
    if bmi_value is None:
        bmi_value = _resolve_bmi_from_group_entry(group_cc_entry)
        bmi_source = 'group_info_cc' if bmi_value is not None else bmi_source
    if bmi_value is None:
        bmi_value = _resolve_bmi_from_group_entry(group_lc_entry)
        bmi_source = 'group_info_lc' if bmi_value is not None else bmi_source
    if bmi_value is None:
        bmi_value = _sanitize_bmi_value(redcap_entry.get('bmi'))
        bmi_source = 'redcap' if bmi_value is not None else None

    age_value = intake_entry.get('age')
    age_source = 'intake_form' if age_value is not None else None
    if age_value is None:
        age_value = redcap_entry.get('age')
        age_source = 'redcap' if age_value is not None else None

    height_m = _sanitize_height_m(intake_entry.get('height_m'))
    height_source = 'intake_form' if height_m is not None else None
    if height_m is None:
        height_m = _sanitize_height_m(redcap_entry.get('height_m'))
        height_source = 'redcap' if height_m is not None else None

    weight_kg = _sanitize_weight_kg(intake_entry.get('weight_kg'))
    weight_source = 'intake_form' if weight_kg is not None else None
    if weight_kg is None:
        weight_kg = _sanitize_weight_kg(redcap_entry.get('weight_kg'))
        weight_source = 'redcap' if weight_kg is not None else None

    sex_asab_raw = redcap_entry.get('sex_asab')
    gender_raw = redcap_entry.get('gender')
    sex_asab_label = _interpret_redcap_value('asab', sex_asab_raw, redcap_definitions)
    gender_label = _interpret_redcap_value('gender', gender_raw, redcap_definitions)

    recording_date, recording_date_source = _resolve_recording_date(schedule_entry, notes_entry, redcap_entry)
    recording_datetime = None
    if schedule_entry and schedule_entry.get('datetime'):
        recording_datetime = schedule_entry['datetime']
    elif notes_entry and notes_entry.get('datetime'):
        recording_datetime = notes_entry['datetime']
    elif redcap_entry.get('recording_datetime'):
        recording_datetime = redcap_entry['recording_datetime']
    ecg_configuration = _resolve_ecg_config(recording_date)

    researchers: list[str] = []
    if schedule_entry and schedule_entry.get('researchers'):
        researchers = list(schedule_entry['researchers'])
    elif notes_entry and notes_entry.get('researchers'):
        researchers = list(notes_entry['researchers'])

    return {
        'participant': participant_id,
        'session': session,
        'session_class': session_class,
        'task': task,
        'recording_date': recording_date.isoformat() if recording_date else None,
        'recording_datetime': recording_datetime.isoformat(timespec='minutes') if recording_datetime else None,
        'recording_date_source': recording_date_source,
        'sex_asab': sex_asab_raw,
        'sex_asab_label': sex_asab_label,
        'gender': gender_raw,
        'gender_label': gender_label,
        'age': age_value,
        'age_source': age_source,
        'height_cm': round(height_m * 100.0, 1) if height_m is not None else None,
        'height_source': height_source,
        'weight_kg': round(weight_kg, 1) if weight_kg is not None else None,
        'weight_source': weight_source,
        'bmi': bmi_value,
        'bmi_source': bmi_source,
        'ecg_configuration': ecg_configuration,
        'researchers': researchers,
        'neuropsych': _resolve_neuropsych_summary(participant_id, neuro_map),
        'sources': {
            'redcap_found': participant_id in redcap_data,
            'redcap_definitions_found': bool(redcap_definitions),
            'redcap_definitions_error': definitions_load_error,
            'intake_found': participant_id in intake_data,
            'neuropsych_found': participant_id in neuro_map,
            'schedule_found': schedule_entry is not None,
            'session_notes_found': notes_entry is not None,
        },
        'definition_sources': {
            'redcap_path': str(redcap_path),
            'redcap_definitions_path': str(definitions_path),
            'schedule_path': str(schedule_path),
        },
    }
