from __future__ import annotations

from typing import Any

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


class DataUploadForm(forms.Form):
    """Multi-file upload form for the 4 CSV types."""

    patients_file = forms.FileField(
        required=False,
        label="Patients CSV",
        widget=forms.ClearableFileInput(attrs={"class": FILE_CLASS, "accept": ".csv"}),
    )
    referrals_file = forms.FileField(
        required=False,
        label="Referrals CSV",
        widget=forms.ClearableFileInput(attrs={"class": FILE_CLASS, "accept": ".csv"}),
    )
    odreferrals_file = forms.FileField(
        required=False,
        label="OD Referrals CSV",
        widget=forms.ClearableFileInput(attrs={"class": FILE_CLASS, "accept": ".csv"}),
    )
    encounters_file = forms.FileField(
        required=False,
        label="Encounters CSV",
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

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        if cleaned is None:
            cleaned = {}
        # At least one file must be uploaded
        files = [
            cleaned.get("patients_file"),
            cleaned.get("referrals_file"),
            cleaned.get("odreferrals_file"),
            cleaned.get("encounters_file"),
        ]
        if not any(files):
            raise forms.ValidationError("Upload at least one CSV file.")
        return cleaned


class BatchEditForm(forms.ModelForm):
    class Meta:
        model = DataImportBatch
        fields = ["notes"]
        widgets = {
            "notes": forms.Textarea(attrs={"class": FIELD_CLASS + " min-h-[80px]", "rows": 2}),
        }
