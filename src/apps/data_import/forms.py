from __future__ import annotations

from django import forms

from .models import DataImportBatch, DataImportFile

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
    """Single-file upload form — user picks a dataset type and uploads one CSV."""

    file_type = forms.ChoiceField(
        choices=DataImportFile.FileType.choices,
        label="Dataset",
        widget=forms.Select(attrs={"class": SELECT_CLASS}),
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


class BatchEditForm(forms.ModelForm):
    class Meta:
        model = DataImportBatch
        fields = ["notes"]
        widgets = {
            "notes": forms.Textarea(attrs={"class": FIELD_CLASS + " min-h-[80px]", "rows": 2}),
        }
