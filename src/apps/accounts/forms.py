from __future__ import annotations

from django import forms

from .models import User


class ProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = [
            "first_name",
            "last_name",
            "email",
            "bio",
            "avatar",
        ]
        widgets = {
            "first_name": forms.TextInput(
                attrs={
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60"
                }
            ),
            "last_name": forms.TextInput(
                attrs={
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60"
                }
            ),
            "email": forms.EmailInput(
                attrs={
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60"
                }
            ),
            "bio": forms.Textarea(
                attrs={
                    "rows": 4,
                    "class": "w-full rounded-md border border-white/15 bg-white/5 px-3 py-2 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400/60",
                }
            ),
        }

    def clean_email(self) -> str:
        # Do not allow email changes via this form yet; keep existing value.
        if self.instance and self.instance.pk:
            return self.instance.email
        return self.cleaned_data.get("email", "").strip()
