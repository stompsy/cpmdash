from __future__ import annotations

from typing import Any

from django import forms

from .models import Task


class TaskForm(forms.ModelForm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Allow omitting priority to use model default
        if "priority" in self.fields:
            self.fields["priority"].required = False
            try:
                from .models import Task as TaskModel

                self.fields["priority"].initial = TaskModel.Priority.MEDIUM
            except Exception:
                self.fields["priority"].initial = 2

    class Meta:
        model = Task
        fields = ["title", "description", "priority", "due_date", "assignee"]
        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60",
                    "placeholder": "Task title",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "rows": 3,
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60",
                    "placeholder": "Optional description",
                }
            ),
            "priority": forms.Select(
                attrs={
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-400/60",
                }
            ),
            "due_date": forms.DateInput(
                attrs={
                    "type": "date",
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60",
                }
            ),
            "assignee": forms.Select(
                attrs={
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-brand-400/60",
                }
            ),
        }


class TaskCompleteForm(forms.ModelForm):
    class Meta:
        model = Task
        fields = ["is_completed"]
