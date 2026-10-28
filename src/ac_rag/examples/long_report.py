"""Deterministic, fictional long-document demo with known evidence positions."""

SOURCE = "Длинный пример — проект Aurora"

QUESTIONS = [
    "Когда запускается проект Aurora?",
    "Как называется резервный канал проекта Aurora?",
    "Какая группа проверяет резервный канал проекта Aurora?",
]
EXPECTED = ["14 ноября 2026", "Orion", "Delta"]
POSITIONS = ["начало", "середина", "связь середины и конца"]


def build_long_report(sections: int = 96) -> str:
    if sections < 96:
        raise ValueError("The complete demo requires 96 sections")
    rows = ["ОТЧЁТ ПО ПРОЕКТУ AURORA. Учебный документ с вымышленными сведениями."]
    for number in range(1, sections + 1):
        rows.append(f"Раздел {number}. Операционный журнал.")
        rows.append(
            f"Запись {number} описывает плановую сверку заявок, расписания и внутренних уведомлений. "
            "Ответственный сотрудник проверяет полноту формы, отмечает дату обработки и передаёт "
            "результат в общий журнал. Записи в этом разделе служат фоном для проверки поиска "
            "по длинному документу. Если обнаружено расхождение, сотрудник фиксирует его в журнале "
            "и передаёт координатору. Номер раздела помогает вернуться к первичному тексту."
        )
        if number == 2:
            rows.append("Ключевой факт: проект Aurora запускается 14 ноября 2026 года.")
        if number == 48:
            rows.append("Ключевой факт: резервный канал связи проекта Aurora называется Orion.")
        if number == 96:
            rows.append("Ключевой факт: систему Orion проверяет группа Delta каждую пятницу.")
    return "\n\n".join(rows)
