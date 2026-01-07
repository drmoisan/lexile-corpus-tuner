from pathlib import Path

f = Path("tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py")
text = f.read_text(encoding="utf-8")

replacements = [
    (
        'PlanTask(phase="P0", task_id="P0-T1", title="Task 1", checked=False)',
        'PlanTask(task_id="P0-T1", phase=0, task_num=1, title="Task 1", checked=False, line_index=1)',
    ),
    (
        'PlanTask(phase="P0", task_id="P0-T1", title="Task", checked=False)',
        'PlanTask(task_id="P0-T1", phase=0, task_num=1, title="Task", checked=False, line_index=1)',
    ),
    (
        'PlanTask(phase="P1", task_id="P1-T3", title="My task", checked=False)',
        'PlanTask(task_id="P1-T3", phase=1, task_num=3, title="My task", checked=False, line_index=3)',
    ),
    (
        'PlanTask(phase="P2", task_id="P2-T5", title="Important task", checked=False)',
        'PlanTask(task_id="P2-T5", phase=2, task_num=5, title="Important task", checked=False, line_index=5)',
    ),
]

for old, new in replacements:
    text = text.replace(old, new)

f.write_text(text, encoding="utf-8")
print("Fixed all PlanTask constructions")
