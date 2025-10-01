from __future__ import annotations

from typing import Any

from django import forms
from django.utils import timezone
from django.utils.text import slugify

from .models import CaseStudy, Tag

FIELD_CLASS = (
    "block w-full rounded-2xl border border-white/10 bg-slate-900/60 px-4 py-3 text-sm "
    "text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-400/60 focus:border-transparent"
)

TEXTAREA_CLASS = FIELD_CLASS + " min-h-[180px]"

CHECKBOX_CLASS = "h-4 w-4 rounded border-slate-600 bg-slate-900 text-brand-400 focus:ring-brand-400/60 focus:ring-offset-0"


class CaseStudyForm(forms.ModelForm):
    published_at = forms.DateTimeField(
        required=False,
        widget=forms.DateTimeInput(
            attrs={"type": "datetime-local", "class": FIELD_CLASS}, format="%Y-%m-%dT%H:%M"
        ),
        input_formats=[
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ],
        help_text="Use local time; leave blank to publish later.",
    )
    tags = forms.ModelMultipleChoiceField(
        queryset=Tag.objects.order_by("name"),
        required=False,
        widget=forms.SelectMultiple(attrs={"class": FIELD_CLASS + " min-h-[200px]"}),
        help_text="Select one or more focus areas applied to this case study.",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fields["slug"].required = False
        published_at = self.instance.published_at if self.instance else None
        if published_at:
            localized = timezone.localtime(published_at)
            self.initial["published_at"] = localized.strftime("%Y-%m-%dT%H:%M")

    class Meta:
        model = CaseStudy
        fields = [
            "title",
            "slug",
            "excerpt",
            "content",
            "hero_image",
            "is_published",
            "published_at",
            "featured",
            "featured_rank",
            "tags",
        ]
        widgets = {
            "title": forms.TextInput(
                attrs={"class": FIELD_CLASS, "placeholder": "Case study title"}
            ),
            "slug": forms.TextInput(attrs={"class": FIELD_CLASS, "placeholder": "slug"}),
            "excerpt": forms.Textarea(
                attrs={
                    "class": TEXTAREA_CLASS,
                    "rows": 3,
                    "placeholder": "Optional short abstract for list views.",
                }
            ),
            "content": forms.Textarea(
                attrs={
                    "class": TEXTAREA_CLASS,
                    "rows": 16,
                    "placeholder": "Body content (supports Markdown/HTML as previously used).",
                }
            ),
            "hero_image": forms.ClearableFileInput(
                attrs={
                    "class": "block w-full text-sm text-slate-200 file:mr-4 file:rounded-full file:border-0 file:bg-brand-500/20 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-brand-100 hover:file:bg-brand-500/40",
                }
            ),
            "featured_rank": forms.NumberInput(attrs={"class": FIELD_CLASS, "min": 0}),
            "is_published": forms.CheckboxInput(attrs={"class": CHECKBOX_CLASS}),
            "featured": forms.CheckboxInput(attrs={"class": CHECKBOX_CLASS}),
        }
        help_texts = {
            "slug": "Leave empty to auto-generate from the title.",
            "featured_rank": "Lower values appear first when marked as featured.",
        }

    def clean_slug(self) -> str:
        slug = self.cleaned_data.get("slug")
        title = self.cleaned_data.get("title")
        if not slug and title:
            slug = slugify(title)
        if slug:
            existing = CaseStudy.objects.filter(slug=slug)
            if self.instance.pk:
                existing = existing.exclude(pk=self.instance.pk)
            if existing.exists():
                raise forms.ValidationError("Another case study already uses this slug.")
        if not slug:
            raise forms.ValidationError(
                "Provide a title to auto-generate a slug or enter one manually."
            )
        return slug

    def save(self, commit: bool = True) -> CaseStudy:
        instance = super().save(commit=False)
        # Ensure slug is present when cleaning wasn't triggered (e.g., slug provided manually).
        if not instance.slug:
            instance.slug = slugify(instance.title)
        if commit:
            instance.save()
            self.save_m2m()
        return instance
