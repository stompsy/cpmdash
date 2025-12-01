from __future__ import annotations

from typing import Any

from django import forms

from .models import Task

# Modern Tailwind form styling with enhanced visual design
INPUT_CLASS = (
    "block w-full rounded-xl border-0 bg-white/10 px-4 py-3 text-base text-white "
    "placeholder:text-slate-400 shadow-sm ring-1 ring-inset ring-white/10 "
    "transition duration-200 "
    "hover:ring-white/20 "
    "focus:ring-2 focus:ring-inset focus:ring-brand-400 focus:outline-none "
    "disabled:cursor-not-allowed disabled:bg-white/5 disabled:text-slate-500"
)

TEXTAREA_CLASS = (
    "block w-full rounded-xl border-0 bg-white/10 px-4 py-3 text-base text-white "
    "placeholder:text-slate-400 shadow-sm ring-1 ring-inset ring-white/10 "
    "transition duration-200 resize-none "
    "hover:ring-white/20 "
    "focus:ring-2 focus:ring-inset focus:ring-brand-400 focus:outline-none"
)


class TaskForm(forms.ModelForm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    class Meta:
        model = Task
        fields = ["title", "description"]
        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": INPUT_CLASS,
                    "placeholder": "What needs to be done?",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "rows": 4,
                    "class": TEXTAREA_CLASS,
                    "placeholder": "Add more details about this task (optional)",
                }
            ),
        }


class TaskCompleteForm(forms.ModelForm):
    class Meta:
        model = Task
        fields = ["is_completed"]
