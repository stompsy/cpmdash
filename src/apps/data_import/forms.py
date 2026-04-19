from __future__ import annotations

from typing import Any, cast

from django import forms

from .models import DataImportBatch

FIELD_CLASS = (
    "block w-full rounded-2xl border border-white/10 bg-slate-900/60 px-4 py-3 text-sm "
    "text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 "
    "focus:ring-brand-400/60 focus:border-transparent"
)

FILE_CLASS = (
    "block w-full rounded-2xl border border-dashed border-white/20 bg-slate-900/40 px-4 py-3 "
    "text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-brand-500 "
    "file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white "
    "hover:file:bg-brand-400 cursor-pointer"
)

SELECT_CLASS = (
    "block w-full rounded-2xl border border-white/10 bg-slate-900/60 px-4 py-3 text-sm "
    "text-slate-100 focus:outline-none focus:ring-2 focus:ring-brand-400/60 "
    "focus:border-transparent appearance-none"
)


class DataUploadForm(forms.Form):
    """Upload form for new batches — always patients.csv (no dataset selector needed)."""

    file = forms.FileField(
        label="CSV File",
        widget=forms.ClearableFileInput(attrs={"class": FILE_CLASS, "accept": ".csv"}),
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": FIELD_CLASS + " min-h-[80px]",
                "rows": 2,
                "placeholder": "Optional notes about this import...",
            }
        ),
    )


class DataUploadToBatchForm(forms.Form):
    """Upload form for adding a dataset to an existing batch — dynamic file type choices."""

    file_type = forms.ChoiceField(
        choices=[],
        label="Dataset",
        widget=forms.Select(attrs={"class": SELECT_CLASS, "id": "id_file_type"}),
        help_text="Select the type of data you are importing.",
    )
    file = forms.FileField(
        label="CSV File",
        widget=forms.ClearableFileInput(attrs={"class": FILE_CLASS, "accept": ".csv"}),
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": FIELD_CLASS + " min-h-[80px]",
                "rows": 2,
                "placeholder": "Optional notes about this import...",
            }
        ),
    )

    def __init__(
        self, *args: Any, file_type_choices: list[tuple[str, str]] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if file_type_choices:
            file_type_field = cast(forms.ChoiceField, self.fields["file_type"])
            file_type_field.choices = file_type_choices


class BatchEditForm(forms.ModelForm):
    class Meta:
        model = DataImportBatch
        fields = ["notes"]
        widgets = {
            "notes": forms.Textarea(attrs={"class": FIELD_CLASS + " min-h-[80px]", "rows": 2}),
        }
