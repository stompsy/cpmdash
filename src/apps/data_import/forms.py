from __future__ import annotations

import contextlib
from typing import Any, cast

from django import forms

from apps.core.models import Agency, County

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

    county = forms.ModelChoiceField(
        queryset=County.objects.none(),
        label="County",
        widget=forms.Select(attrs={"class": SELECT_CLASS}),
    )
    agency = forms.ModelChoiceField(
        queryset=Agency.objects.none(),
        label="Agency",
        widget=forms.Select(attrs={"class": SELECT_CLASS}),
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        county_field = cast(forms.ModelChoiceField, self.fields["county"])
        agency_field = cast(forms.ModelChoiceField, self.fields["agency"])
        county_field.queryset = County.objects.order_by("name")
        agency_field.queryset = Agency.objects.none()

        county_value = self.data.get("county") or self.initial.get("county")
        if county_value:
            with contextlib.suppress(TypeError, ValueError):
                agency_field.queryset = Agency.objects.filter(county_id=int(county_value)).order_by(
                    "name"
                )
                return

        agency_field.queryset = Agency.objects.select_related("county").order_by(
            "county__name", "name"
        )

    def clean(self) -> dict[str, Any]:
        cleaned: dict[str, Any] = super().clean() or {}
        county = cleaned.get("county")
        agency = cleaned.get("agency")
        if county and agency and agency.county_id != county.id:
            self.add_error("agency", "Selected agency does not belong to the selected county.")
        return cleaned


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
